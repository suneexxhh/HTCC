import os
import time
import wandb
import numpy as np
from functools import reduce
import torch
from memat.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class HandsRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(HandsRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = \
                    self.envs.step(torch.tensor(actions.transpose(1, 0, 2)))
                obs = _t2n(obs)
                share_obs = _t2n(share_obs)
                rewards = _t2n(rewards)
                dones = _t2n(dones)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.task,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)
                    self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, total_num_steps)
                    done_episodes_rewards = []

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = _t2n(share_obs).copy()
        self.buffer.obs[0] = _t2n(obs).copy()
        self.buffer.rnn_states[0] = 0
        self.buffer.rnn_states_critic[0] = 0
        self.buffer.masks[0] = 1
        self.buffer.active_masks[0] = 1

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, None, active_masks,
                           None)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_deterministic = os.environ.get("HANDS_EVAL_DETERMINISTIC", "1") != "0"
        eval_diagnostics = os.environ.get("HANDS_EVAL_DIAGNOSTICS", "0") == "1"

        while len(eval_episode_rewards) < self.all_args.eval_episodes:
            eval_obs, eval_share_obs, _ = self.eval_envs.reset()
            eval_obs = _t2n(eval_obs)
            eval_share_obs = _t2n(eval_share_obs)

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            running_rewards = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)
            batch_size = min(self.n_eval_rollout_threads,
                             self.all_args.eval_episodes - len(eval_episode_rewards))
            batch_finished = np.zeros(batch_size, dtype=bool)
            first_episode_rewards = np.full(self.n_eval_rollout_threads, np.nan, dtype=np.float32)
            first_done_steps = np.full(self.n_eval_rollout_threads, -1, dtype=np.int32)
            eval_step = 0

            while not np.all(batch_finished):
                eval_step += 1
                self.trainer.prep_rollout()
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            deterministic=eval_deterministic)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

                eval_obs, eval_share_obs, eval_rewards, eval_dones, _, _ = \
                    self.eval_envs.step(torch.tensor(eval_actions.transpose(1, 0, 2)))
                eval_obs = _t2n(eval_obs)
                eval_share_obs = _t2n(eval_share_obs)
                eval_rewards = _t2n(eval_rewards)
                eval_dones = _t2n(eval_dones)

                running_rewards += np.mean(eval_rewards, axis=1).reshape(-1)
                eval_dones_env = np.all(eval_dones, axis=1).reshape(-1)
                first_done = eval_dones_env & np.isnan(first_episode_rewards)
                first_episode_rewards[first_done] = running_rewards[first_done]
                first_done_steps[first_done] = eval_step

                eval_rnn_states[eval_dones_env] = 0
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env] = 0

                for eval_i in range(batch_size):
                    if eval_dones_env[eval_i] and not batch_finished[eval_i]:
                        eval_episode_rewards.append(float(running_rewards[eval_i]))
                        batch_finished[eval_i] = True

            if eval_diagnostics:
                completed_ids = np.flatnonzero(first_done_steps >= 0)
                completion_order = completed_ids[np.argsort(first_done_steps[completed_ids], kind='stable')]
                earliest_ids = completion_order[:batch_size]
                fixed_ids = np.arange(batch_size)
                print("eval_diagnostics fixed_ids={} fixed_rewards={} earliest_ids={} earliest_rewards={}."
                      .format(fixed_ids.tolist(), first_episode_rewards[fixed_ids].tolist(),
                              earliest_ids.tolist(), first_episode_rewards[earliest_ids].tolist()))

        eval_env_infos = {
            'eval_average_episode_rewards': [np.mean(eval_episode_rewards)],
            'eval_max_episode_rewards': [np.max(eval_episode_rewards)]
        }
        self.log_env(eval_env_infos, total_num_steps)
        eval_mode = "deterministic" if eval_deterministic else "stochastic"
        print("eval_average_episode_rewards ({}) is {}.".format(eval_mode, np.mean(eval_episode_rewards)))

        if self.eval_envs is self.envs:
            self.warmup()
