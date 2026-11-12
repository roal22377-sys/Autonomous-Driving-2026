from collections import deque
import numpy as np
import torch
import os

EPS = 1e-8 

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

class RolloutBuffer:
    def __init__(
            self, device:torch.device,
            obs_dim:int,
            action_dim:int,
            discount_factor:float,
            gae_coeff:float, 
            n_envs:int,
            n_steps:int,
            minibatch_size:int) -> None:
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.gae_coeff = gae_coeff
        self.n_envs = n_envs
        self.n_steps_per_env = int(n_steps/n_envs)
        self.minibatch_size = minibatch_size
        self.initialize()

    ################
    # Public Methods
    ################

    def initialize(self):
        self.storage = [deque(maxlen=self.n_steps_per_env) for _ in range(self.n_envs)]

    def addTransition(self, states, actions, log_probs, rewards, dones, fails, next_states):
        for env_idx in range(self.n_envs):
            self.storage[env_idx].append([
                states[env_idx], actions[env_idx], log_probs[env_idx], rewards[env_idx], 
                dones[env_idx], fails[env_idx], next_states[env_idx]
            ])

    @torch.no_grad()
    def getMiniBatches(self, states_tensor, actions_tensor, log_probs_tensor, reward_targets_tensor, advantages_tensor):
        batch_size = states_tensor.shape[0]
        assert batch_size == actions_tensor.shape[0] == reward_targets_tensor.shape[0] == advantages_tensor.shape[0]

        # shuffle index
        indices = np.random.permutation(batch_size)
        index_list = list_chunk(indices, self.minibatch_size)    # [[indices], [indices], [indices], ...]

        minibatches = []
        for index in index_list:
            minibatch = []
            minibatch.append(states_tensor[index])
            minibatch.append(actions_tensor[index])
            minibatch.append(log_probs_tensor[index])
            minibatch.append(reward_targets_tensor[index])
            minibatch.append(advantages_tensor[index])

            minibatches.append(minibatch)

        return minibatches
        
    @torch.no_grad()
    def getBatches(self, obs_rms, reward_rms, reward_critic):
        states_list = []
        actions_list = []
        log_probs_list = []
        reward_targets_list = []
        advantages_list = []

        for env_idx in range(self.n_envs):
            env_trajs = list(self.storage[env_idx])
            states = np.array([traj[0] for traj in env_trajs])
            actions = np.array([traj[1] for traj in env_trajs])
            log_probs = np.array([traj[2] for traj in env_trajs])
            rewards = np.array([traj[3] for traj in env_trajs])
            dones = np.array([traj[4] for traj in env_trajs])
            fails = np.array([traj[5] for traj in env_trajs])
            next_states = np.array([traj[6] for traj in env_trajs])

            # normalize 
            states = obs_rms.normalize(states)
            next_states = obs_rms.normalize(next_states)
            rewards = reward_rms.normalize(rewards)

            states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)

            # get values
            next_reward_values = reward_critic(next_states_tensor).squeeze(-1).detach().cpu().numpy()
            reward_values = reward_critic(states_tensor).squeeze(-1).detach().cpu().numpy()

            # get targets & advantages
            # # Option 1. Monte Carlo estimate of returns
            # #   NOTE: does not bootstrap at time-limit truncation or at the end of the rollout,
            # #   and a truncated episode's return leaks into the next episode (only fails cut it).
            # discounted_reward = 0.0
            # reward_targets = np.zeros_like(rewards)
            # for t in reversed(range(len(reward_targets))):
            #     discounted_reward = rewards[t] + self.discount_factor*(1.0 - fails[t])*discounted_reward
            #     reward_targets[t] = discounted_reward
            # advantages = reward_targets - reward_values

            # # Option 2. 1-step TD
            # reward_targets = rewards + self.discount_factor*(1.0 - fails)*next_reward_values
            # advantages = reward_targets - reward_values

            # Option 3. GAE (used): lambda-return target, bootstraps with V(s') unless the episode failed,
            # and stops propagating at episode boundaries (dones). lambda = gae_coeff.
            reward_delta = 0.0
            reward_targets = np.zeros_like(rewards)
            for t in reversed(range(len(reward_targets))):
                reward_targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_reward_values[t] \
                                + self.discount_factor*(1.0 - dones[t])*reward_delta
                reward_delta = self.gae_coeff*(reward_targets[t] - reward_values[t])
            advantages = reward_targets - reward_values

            # append
            states_list.append(states)
            actions_list.append(actions)
            log_probs_list.append(log_probs)
            reward_targets_list.append(reward_targets)
            advantages_list.append(advantages)

        # convert to tensor
        states_tensor = torch.tensor(np.concatenate(states_list, axis=0), device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(np.concatenate(actions_list, axis=0), device=self.device, dtype=torch.float32)
        log_probs_tensor = torch.tensor(np.concatenate(log_probs_list, axis=0), device=self.device, dtype=torch.float32)
        reward_targets_tensor = torch.tensor(np.concatenate(reward_targets_list, axis=0), device=self.device, dtype=torch.float32)
        advantages_tensor = torch.tensor(np.concatenate(advantages_list, axis=0), device=self.device, dtype=torch.float32)

        # normalize GAE
        advantages_tensor = (advantages_tensor - advantages_tensor.mean())/(advantages_tensor.std() + EPS)

        # convert to minibatches
        minibatches = self.getMiniBatches(states_tensor, actions_tensor, log_probs_tensor, reward_targets_tensor, advantages_tensor)

        return minibatches, states_tensor
    
