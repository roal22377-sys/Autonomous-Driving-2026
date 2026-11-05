from algorithm.common import *
from utils.color import cprint

from .storage import RolloutBuffer

import numpy as np
import torch
import os

EPS = 1e-8


class Agent(AgentBase):
    def __init__(self, args):
        super().__init__(args)

        # for expert rollout
        self.total_steps = args.total_steps

        # for training
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.max_grad_norm = args.max_grad_norm

        # for model
        self.actor = Actor(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        # for buffer
        self.rollout_dataset = RolloutBuffer(self.device, self.total_steps, self.batch_size)


    def step(self, states, actions):
        # update statistics
        if self.norm_obs:
            self.obs_rms.update(states)
            states = self.obs_rms.normalize(states)

        self.rollout_dataset.addBatch(states, actions)

    def train(self):
        states_tensor, actions_tensor = self.rollout_dataset.getBatches()
        action_dists = self.actor(states_tensor)

        # ============================ implement here ============================ #
        MSE_Loss = torch.nn.MSELoss()
        action_mean = action_dists.loc      # deterministic action (tanh output, normalized to [-1, 1])
        actor_loss = MSE_Loss(action_mean, actions_tensor)      # behavior cloning: regress expert actions
        # ======================================================================== #
        
        # update
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

        train_results = {
            'actor_loss':actor_loss.item(),
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
            self.optimizer.load_state_dict(checkpoint['optim'])
            cprint(f'[{self.name}_{self.algo_idx}] load success.', bold=True, color="blue")
        else:
            self.actor.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")