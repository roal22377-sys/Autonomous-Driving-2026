from algorithm.common import *
from utils.color import cprint

from .storage import RolloutBuffer

import os
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.kl import kl_divergence

EPS = 1e-8
        

class Agent(AgentBase):
    def __init__(self, args):
        super().__init__(args)

        # for training
        self.lr = args.lr
        self.minibatch_size = args.minibatch_size
        self.train_epochs = args.train_epochs
        self.critic_coeff = args.critic_coeff
        self.gae_coeff = args.gae_coeff
        self.ent_coeff = args.ent_coeff
        self.max_grad_norm = args.max_grad_norm
        self.n_envs = args.n_envs
        self.n_steps = args.n_steps

        # for trust region
        self.max_kl = args.max_kl
        self.kl_tolerance = args.kl_tolerance
        self.adaptive_lr_ratio = args.adaptive_lr_ratio
        self.clip_ratio = args.clip_ratio
        
        # for model
        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)

        # for buffer
        self.replay_buffer = RolloutBuffer(
            self.device, self.obs_dim, self.action_dim, self.discount_factor, 
            self.gae_coeff, self.n_envs, self.n_steps, self.minibatch_size)

    
    def step(self, rewards, dones, fails, next_states):
        self.replay_buffer.addTransition(self.state, self.action, self.log_prob, rewards, dones, fails, next_states)

        # update statistics
        if self.norm_obs:
            self.obs_rms.update(self.state)
        if self.norm_reward:
            self.reward_rms.update(rewards)

    def train(self, steps):
        # logs
        epochs = 0

        # backup old policy
        with torch.no_grad():
            _, total_states_tensor = self.replay_buffer.getBatches(self.obs_rms, self.reward_rms, self.critic)
            total_old_action_dists = self.actor(total_states_tensor)

        # ============================ implement here ============================ #
        for _ in range(self.train_epochs):
            epochs += 1

            # get batches
            with torch.no_grad():
                minibatches, _ = self.replay_buffer.getBatches(self.obs_rms, self.reward_rms, self.critic)

            for i, minibatch in enumerate(minibatches):
                states_tensor, actions_tensor, old_log_probs_tensor, reward_targets_tensor, reward_gaes_tensor = minibatch

                # critic loss
                ##############
                ##   eq.3   ##
                ##############
                # L^VF = E[(V(s_t) - R_t)^2]
                values_tensor = self.critic(states_tensor).squeeze(-1)
                critic_loss = torch.mean(torch.square(values_tensor - reward_targets_tensor))

                # actor loss
                ##############
                ##  eq.5-6  ##
                ##############
                # r_t = pi_theta(a|s) / pi_old(a|s) = exp(log pi_theta - log pi_old)
                action_dists = self.actor(states_tensor)
                log_probs_tensor = torch.sum(action_dists.log_prob(actions_tensor), dim=-1)
                prob_ratios_tensor = torch.exp(log_probs_tensor - old_log_probs_tensor)
                clipped_ratios_tensor = torch.clamp(prob_ratios_tensor, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)

                ##############
                ##   eq.7   ##
                ##############
                # L^CLIP = E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]
                actor_objective = torch.mean(torch.minimum(
                    prob_ratios_tensor*reward_gaes_tensor,
                    clipped_ratios_tensor*reward_gaes_tensor))
                actor_loss = -actor_objective
                entropy = torch.mean(torch.sum(action_dists.entropy(), dim=-1))

                # total loss
                total_loss = actor_loss + self.critic_coeff*critic_loss - self.ent_coeff*entropy

                # update
                self.optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # KL(pi_old || pi_new) averaged over the whole rollout, for early stopping & adaptive lr
            with torch.no_grad():
                total_new_action_dists = self.actor(total_states_tensor)
                kl = torch.mean(torch.sum(kl_divergence(total_old_action_dists, total_new_action_dists), dim=-1))
                entropy = torch.mean(torch.sum(total_new_action_dists.entropy(), dim=-1))
        # ======================================================================== #

            if kl > self.max_kl*self.kl_tolerance: break

        # reset buffer
        self.replay_buffer.initialize()

        # adjust learning rate based on KL divergence
        if kl > self.max_kl*self.kl_tolerance:
            self.lr /= self.adaptive_lr_ratio
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        elif kl < self.max_kl/self.kl_tolerance:
            self.lr *= self.adaptive_lr_ratio
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        
        train_results = {
            'critic_loss':critic_loss.item(),
            'actor_loss':actor_loss.item(),
            'kl':kl.item(),
            'entropy':entropy.item(),
            'epochs':epochs,
            'lr': self.lr
            }
        return train_results

    def save(self, model_num=None, log=True):
        if model_num is None:
            checkpoint_file = f"{self.checkpoint_dir}/model.pt"
            # for sim2real
            self.obs_rms.save(self.sim2real_dir)
            self.reward_rms.save(self.sim2real_dir)
            torch.save(self.actor, f"{self.sim2real_dir}/actor.pt")
        else:
            checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"

        # save rms
        self.obs_rms.save(self.save_dir)
        self.reward_rms.save(self.save_dir)

        # save models
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optim': self.optimizer.state_dict(),
            }, checkpoint_file)
        if log: cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def load(self, model_num=None):
        # load rms
        self.obs_rms.load(self.save_dir)
        self.reward_rms.load(self.save_dir)

        # load models
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.sim2real_dir)
        if model_num is None:
            checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        else:
            checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            cprint(f'[{self.name}_{self.algo_idx}] load success.', bold=True, color="blue")
        else:
            self.actor.initialize()
            self.critic.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")