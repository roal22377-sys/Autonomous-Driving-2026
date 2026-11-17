import numpy as np
import pickle
import torch
import os

class RunningMeanStd(object):
    def __init__(self, name:str, state_dim:int, limit_cnt:float=np.inf):
        self.name = name
        self.limit_cnt = limit_cnt
        self.mean = np.zeros(state_dim, np.float32)
        self.var = np.ones(state_dim, np.float32)
        self.count = 0.0

    def update(self, raw_data):
        arr = raw_data.reshape(-1, self.mean.shape[0])
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        return

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if self.count >= self.limit_cnt: return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        return

    def normalize(self, observations, mean=0.0, std=1.0):
        return_torch = False
        if isinstance(observations, torch.Tensor):
            return_torch = True
            device, dtype = observations.device, observations.dtype
            observations = observations.clone().detach().cpu().numpy()

        norm_obs = (observations - self.mean)/np.sqrt(self.var + 1e-8)
        if return_torch:
            return torch.tensor(norm_obs, dtype=dtype, device=device)
        else:
            return norm_obs * std + mean
    
    def load(self, save_dir, required=False):
        file_name = f"{save_dir}/{self.name}_scale.pkl"
        if required and not os.path.exists(file_name):
            # running a trained actor on unnormalized observations drives blind; never do it silently
            raise FileNotFoundError(f'observation normalizer missing: {file_name}')
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                self.mean, self.var, self.count = pickle.load(f)
            print(f'[load] {self.name} success.')
        else:
            print(f'[load] {self.name} fail.')

    def save(self, save_dir):
        file_name = f"{save_dir}/{self.name}_scale.pkl"
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'wb') as f:
            pickle.dump([self.mean, self.var, self.count], f)