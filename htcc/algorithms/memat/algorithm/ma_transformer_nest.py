import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from memat.algorithms.utils.util import check, init
from memat.algorithms.utils.transformer_act import discrete_autoregreesive_act
from memat.algorithms.utils.transformer_act import discrete_parallel_act
from memat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from memat.algorithms.utils.transformer_act import continuous_parallel_act


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class CMSResidualBlock(nn.Module):
    def __init__(self, n_embd, hidden_multiplier=2, delta_clip=1.0):
        super(CMSResidualBlock, self).__init__()
        hidden_dim = int(n_embd * hidden_multiplier)
        self.norm = nn.LayerNorm(n_embd)
        self.net = nn.Sequential(
            init_(nn.Linear(n_embd, hidden_dim), activate=True),
            nn.GELU(),
            init_(nn.Linear(hidden_dim, n_embd))
        )
        self.delta_clip = float(delta_clip)

    def forward(self, x):
        delta = self.net(self.norm(x))
        if self.delta_clip > 0:
            with torch.no_grad():
                norm = delta.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / self.delta_clip, min=1.0)
            delta = delta / scale
        return x + delta


class NestedTeamMemory(nn.Module):
    """
    NEST-MAT-Lite team memory.

    This is a low-intrusion CMS-style feature memory: it compresses the current team
    representation through multiple residual levels, then injects it back to each
    agent with a weak per-agent gate. It does not mutate parameters or keep episode
    state in this first version.
    """

    def __init__(
        self,
        n_embd,
        num_levels=3,
        hidden_multiplier=2,
        gate_bias=-2.0,
        alpha_init=0.05,
        delta_clip=1.0,
    ):
        super(NestedTeamMemory, self).__init__()
        self.n_embd = n_embd
        self.num_levels = max(1, int(num_levels))
        self.delta_clip = float(delta_clip)

        self.levels = nn.ModuleList([
            CMSResidualBlock(
                n_embd,
                hidden_multiplier=hidden_multiplier,
                delta_clip=delta_clip,
            )
            for _ in range(self.num_levels)
        ])

        self.memory_norm = nn.LayerNorm(n_embd * self.num_levels)
        self.memory_proj = init_(nn.Linear(n_embd * self.num_levels, n_embd))
        self.gate_norm = nn.LayerNorm(n_embd * 2)
        self.gate = init_(nn.Linear(n_embd * 2, n_embd))
        nn.init.constant_(self.gate.bias, float(gate_bias))
        self.out_norm = nn.LayerNorm(n_embd)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.latest_stats = {}

    def forward(self, agent_tokens):
        team_context = agent_tokens.mean(dim=1)
        current = team_context
        level_outputs = []
        for level in self.levels:
            current = level(current)
            level_outputs.append(current)

        memory = torch.cat(level_outputs, dim=-1)
        memory = self.memory_proj(self.memory_norm(memory)).unsqueeze(1)
        memory = memory.expand(-1, agent_tokens.size(1), -1)

        gate_input = torch.cat([agent_tokens, memory], dim=-1)
        gate = torch.sigmoid(self.gate(self.gate_norm(gate_input)))
        delta = gate * memory
        if self.delta_clip > 0:
            with torch.no_grad():
                norm = delta.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / self.delta_clip, min=1.0)
            delta = delta / scale

        alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
        rep_change = alpha * delta

        with torch.no_grad():
            self.latest_stats = {
                "nest/alpha": alpha.detach(),
                "nest/alpha_raw": self.alpha.detach(),
                "nest/gate_mean": gate.detach().mean(),
                "nest/gate_std": gate.detach().std(unbiased=False),
                "nest/gate_min": gate.detach().min(),
                "nest/gate_max": gate.detach().max(),
                "nest/memory_norm": memory.detach().norm(dim=-1).mean(),
                "nest/delta_norm": delta.detach().norm(dim=-1).mean(),
                "nest/rep_change_norm": rep_change.detach().norm(dim=-1).mean(),
            }

        return self.out_norm(agent_tokens + rep_change)

    def get_stats(self):
        stats = {}
        for key, value in self.latest_stats.items():
            if torch.is_tensor(value):
                stats[key] = float(value.detach().cpu().item())
            else:
                stats[key] = float(value)
        return stats


class Encoder(nn.Module):

    def __init__(
        self,
        state_dim,
        obs_dim,
        n_block,
        n_embd,
        n_head,
        n_agent,
        encode_state,
        nest_cms_levels=3,
        nest_cms_hidden_mult=2,
        nest_gate_bias=-2.0,
        nest_alpha_init=0.05,
        nest_delta_clip=1.0,
    ):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.nested_team_memory = NestedTeamMemory(
            n_embd,
            num_levels=nest_cms_levels,
            hidden_multiplier=nest_cms_hidden_mult,
            gate_bias=nest_gate_bias,
            alpha_init=nest_alpha_init,
            delta_clip=nest_delta_clip,
        )
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        rep_base = self.blocks(self.ln(x))
        rep = self.nested_team_memory(rep_base)
        v_loc = self.head(rep)

        return v_loc, rep

    def get_memory_stats(self):
        return self.nested_team_memory.get_stats()


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, action, obs_rep, obs):
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)

        return logit


class MultiAgentTransformer(nn.Module):

    def __init__(
        self,
        state_dim,
        obs_dim,
        action_dim,
        n_agent,
        n_block,
        n_embd,
        n_head,
        encode_state=False,
        device=torch.device("cpu"),
        action_type='Discrete',
        dec_actor=False,
        share_actor=False,
        nest_cms_levels=3,
        nest_cms_hidden_mult=2,
        nest_gate_bias=-2.0,
        nest_alpha_init=0.05,
        nest_delta_clip=1.0,
    ):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        state_dim = 37

        self.encoder = Encoder(
            state_dim,
            obs_dim,
            n_block,
            n_embd,
            n_head,
            n_agent,
            encode_state,
            nest_cms_levels=nest_cms_levels,
            nest_cms_hidden_mult=nest_cms_hidden_mult,
            nest_gate_bias=nest_gate_bias,
            nest_alpha_init=nest_alpha_init,
            nest_delta_clip=nest_delta_clip,
        )
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot

    def get_memory_stats(self):
        return self.encoder.get_memory_stats()
