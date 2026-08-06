import torch
import torch.nn as nn
import numpy as np

from memat.algorithms.utils.util import check
from memat.algorithms.utils.transformer_act import discrete_autoregreesive_act
from memat.algorithms.utils.transformer_act import discrete_parallel_act
from memat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from memat.algorithms.utils.transformer_act import continuous_parallel_act
from memat.algorithms.memat.algorithm.ma_transformer_nest import (
    CMSResidualBlock,
    Decoder,
    EncodeBlock,
    init_,
)


def _parse_clock_periods(periods, num_levels):
    if isinstance(periods, str):
        raw = [p.strip() for p in periods.replace(";", ",").split(",") if p.strip()]
        parsed = [int(p) for p in raw]
    else:
        parsed = [int(p) for p in periods]

    if not parsed:
        parsed = [1]
    parsed = [max(1, p) for p in parsed]
    if len(parsed) < num_levels:
        parsed = parsed + [parsed[-1]] * (num_levels - len(parsed))
    return parsed[:num_levels]


class ClockedNestedTeamMemory(nn.Module):
    """
    NEST-MAT-Clock team memory.

    Compared with NEST-MAT-Lite, CMS feature levels are updated with explicit
    multi-timescale clock rates. The default soft clock is a pure function of the
    current features, so it does not introduce an untracked recurrent policy
    state. A hard clock is kept for ablations.
    """

    def __init__(
        self,
        n_embd,
        num_levels=3,
        hidden_multiplier=2,
        gate_bias=-2.0,
        alpha_init=0.05,
        delta_clip=1.0,
        clock_periods="1,2,4",
        clock_mode="soft",
        clock_min_rate=0.25,
        clock_warmup=0,
        clock_beta=1.0,
    ):
        super(ClockedNestedTeamMemory, self).__init__()
        self.n_embd = n_embd
        self.num_levels = max(1, int(num_levels))
        self.delta_clip = float(delta_clip)
        self.clock_mode = str(clock_mode).lower()
        if self.clock_mode not in ("soft", "hard"):
            raise ValueError("clock_mode must be 'soft' or 'hard'")
        self.clock_min_rate = float(clock_min_rate)
        self.clock_warmup = max(0, int(clock_warmup))
        self.clock_beta = float(clock_beta)
        if self.clock_beta < 0.0 or self.clock_beta > 1.0:
            raise ValueError("clock_beta must be in [0, 1]")

        periods = _parse_clock_periods(clock_periods, self.num_levels)
        self.register_buffer("clock_periods", torch.tensor(periods, dtype=torch.long), persistent=False)

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

    def _clock_rates(self, batch_size, device, dtype):
        periods = self.clock_periods.to(device=device)
        if self.clock_mode == "soft":
            clock_rates = 1.0 / periods.to(dtype=dtype)
            # beta interpolates between NEST-MAT-Lite and the current Clock:
            # beta=0.0 -> Lite-like full-rate CMS levels [1, 1, 1]
            # beta=1.0 -> current Clock rates, e.g. periods 1,2,4 -> [1, .5, .25]
            rates = (1.0 - self.clock_beta) + self.clock_beta * clock_rates
            min_rate = torch.tensor(self.clock_min_rate, device=device, dtype=dtype)
            rates = torch.clamp(rates, min=min_rate, max=1.0)
            return rates.unsqueeze(0).expand(batch_size, -1)

        positions = torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(-1)
        if self.clock_warmup > 0:
            active = positions >= int(self.clock_warmup)
        else:
            active = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        active = active & ((positions % periods.view(1, -1)) == 0)
        hard_rates = active.to(dtype=dtype)
        return (1.0 - self.clock_beta) + self.clock_beta * hard_rates

    def forward(self, agent_tokens):
        team_context = agent_tokens.mean(dim=1)
        current = team_context
        level_outputs = []
        clock_rates = self._clock_rates(agent_tokens.size(0), agent_tokens.device, agent_tokens.dtype)

        for level_idx, level in enumerate(self.levels):
            candidate = level(current)
            level_delta = candidate - current
            rate = clock_rates[:, level_idx].unsqueeze(-1)
            current = current + rate * level_delta
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
            level_norms = torch.stack([
                level_output.detach().norm(dim=-1).mean()
                for level_output in level_outputs
            ])
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
                "nest/clock_beta": torch.tensor(self.clock_beta, device=agent_tokens.device),
                "nest/clock_rate_mean": clock_rates.detach().mean(),
                "nest/level_norm_mean": level_norms.mean(),
            }
            for idx in range(self.num_levels):
                self.latest_stats["nest/clock_level{}_rate".format(idx)] = clock_rates[:, idx].detach().mean()
                self.latest_stats["nest/level{}_norm".format(idx)] = level_norms[idx].detach()

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
        nest_clock_periods="1,2,4",
        nest_clock_mode="soft",
        nest_clock_min_rate=0.25,
        nest_clock_warmup=0,
        nest_clock_beta=1.0,
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
        self.nested_team_memory = ClockedNestedTeamMemory(
            n_embd,
            num_levels=nest_cms_levels,
            hidden_multiplier=nest_cms_hidden_mult,
            gate_bias=nest_gate_bias,
            alpha_init=nest_alpha_init,
            delta_clip=nest_delta_clip,
            clock_periods=nest_clock_periods,
            clock_mode=nest_clock_mode,
            clock_min_rate=nest_clock_min_rate,
            clock_warmup=nest_clock_warmup,
            clock_beta=nest_clock_beta,
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
        nest_clock_periods="1,2,4",
        nest_clock_mode="soft",
        nest_clock_min_rate=0.25,
        nest_clock_warmup=0,
        nest_clock_beta=1.0,
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
            nest_clock_periods=nest_clock_periods,
            nest_clock_mode=nest_clock_mode,
            nest_clock_min_rate=nest_clock_min_rate,
            nest_clock_warmup=nest_clock_warmup,
            nest_clock_beta=nest_clock_beta,
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
