'''
    This F1Wrapper is for 
        1. usage issue (gymnasium style)
        2. define observation & action space
        3. observation processing (concatenate lidar + other properties)
        
        ->  outputs of step function should have (n_agents, dim) shape
            ex) observations, rewards, terminates, truncates, infos = env.step([agent_1_action, 
                                                                                agent_2_action,
                                                                                ,,,
                                                                                agent_n_action])
'''

from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.observation import get_obs_config, make_observation, add_obs_noise

EPS = 1e-8

def unnormalize_speed(value, minimum, maximum):
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
    temp_a = np.ones_like(value)*temp_a
    temp_b = np.ones_like(value)*temp_b
    return temp_a*value + temp_b


class F1Wrapper(gym.Wrapper):
    def __init__(self, args, maps, render_mode=None, obs_noise=False) -> None:
        '''
            obs_noise: add lidar noise/dropout and speed reading error (see `lidar`,
                       `speed_obs` in f1tenth.yaml). Use it only for collecting training data.
        '''

        self._env = gym.make("f1tenth_gym:f1tenth-v0", 
                            args=args,
                            maps=maps,
                            render_mode=render_mode)
        super().__init__(self._env)
        self.show_centerline = args.show_centerline

        # for control
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer

        # for spaces
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.observation_space = spaces.Box(-np.inf*np.ones(self.obs_dim), np.inf*np.ones(self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)

        # for observation (lidar + speed), checks obs_dim
        self.obs_cfg = get_obs_config(args)
        self.obs_noise = obs_noise
        self.noise_rng = np.random.default_rng(args.get('seed', None))
        self.speed_scale = 1.0

    def _reset_pose(self, obs_dict):
        # collision
        self.collision = obs_dict['collisions'][0]
        
        # cartesian coordinate pose
        poses_x, poses_y, poses_theta = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = obs_dict['poses_theta'][0]
        
        # frenet coordinate pose
        poses_s, poses_d, poses_phi = 0, 0, 0
        s, ey, phi = self._env.track.cartesian_to_frenet2(poses_x, poses_y, poses_theta)
        poses_s = s
        poses_d = ey
        poses_phi = phi
        self.position_frenet = np.stack([poses_s, poses_d]).T
        self.yaw_frenet = poses_phi

    def _step_pose(self, obs_dict):
        # collision
        self.collision = obs_dict['collisions'][0]
        
        # cartesian coordinate pose
        poses_x, poses_y, poses_theta = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = obs_dict['poses_theta'][0]
        
        poses_s, poses_d, poses_phi = 0, 0, 0
        s, ey, phi = self._env.track.cartesian_to_frenet2(poses_x, poses_y, poses_theta)
        poses_s = s
        poses_d = ey
        poses_phi = phi
        
        # assume that vehicle can't complete 1 cycle in single step
        self.delta_s = poses_s - self.position_frenet[0]
        total_track_s = self._env.track.centerline.spline.s[-1]
        if abs(self.delta_s) > total_track_s/2.0 and self.delta_s < 0:
            # if vehicle moved from spline endpoint to spline startpoint
            self.delta_s += total_track_s
                        
        # frenet coordinate pose
        self.position_frenet = np.stack([poses_s, poses_d]).T
        self.yaw_frenet = poses_phi

    def reset(self, **kwargs):
        self.history = deque(maxlen=10)
        if self.obs_noise:
            # per-episode speed reading error, like an imperfect odom calibration
            self.speed_scale = self.noise_rng.uniform(*self.obs_cfg['speed']['scale_range'])
        obs_dict, info = self._env.reset(**kwargs)
        if self.show_centerline:
            self._env.unwrapped.add_render_callback(self._env.track.centerline.render_waypoints)
        self._reset_pose(obs_dict)
        obs = self.getObs(obs_dict, reset=True)
        info['obs_dict'] = obs_dict
        return obs, info
    
    # ============================ implement here ============================ #
    # TODO
    # You need to implement your own reward function.
    def calc_reward(self):        
        # forward reward
        forward_reward = 1.0

        # collison cost
        collision_cost = 10.0 if self.collision else 0.0 
        
        # final reward
        reward = forward_reward - collision_cost
        reward_dict = {'forward_reward': forward_reward,
                       'collision_cost': collision_cost}
        return reward, reward_dict
    # ======================================================================== #
        
    def step(self, action:np.array):
        '''
            Original env's action : [desired steer, desired speed]
                                    steer : -1(left) ~ 1(right)
                                    speed : ~inf ~ inf

            Changed to:             [desired steer, normalized speed]
                                    steer : -1(left) ~ 1(right)
                                    speed : -1 ~ 1              -> multiplied by self.max_speed
        '''
        _action = action.copy()
        _action[0] *= self.max_steer
        _action[1] = unnormalize_speed(_action[1], self.min_speed, self.max_speed)
        obs_dict, _, terminate, truncate, info = self._env.step(_action)
        self._step_pose(obs_dict)
        obs = self.getObs(obs_dict)
        reward, reward_dict = self.calc_reward()
        info['obs_dict'] = obs_dict
        info.update(reward_dict)
        return obs, reward, terminate, truncate, info
    
    # ============================ implement here ============================ #
    # TODO
    # You need to implement your own observation.
    def getObs(self, obs_dict, reset=False):
        '''
            Original observation is dictionary with keys:
                === key ====     === shape ===
                'ego_idx'       | int
                'scans'         | (num_agents, 1080)
                'poses_x'       | (num_agents,)
                'poses_y'       | (num_agents,)
                'poses_theta'   | (num_agents,)
                'linear_vels_x' | (num_agents,)
                'linear_vels_y' | (num_agents,)
                'ang_vels_z'    | (num_agents,)
                'collisions'    | (num_agents,)
                'lap_times'     | (num_agents,)
                'lap_counts'    | (num_agents,)
            
            ==> Changed to [downsampled lidar (lidar.num_beams), speed (if speed_obs.enabled)]
                built by utils/observation.py, the same code agent_node.py uses.
        '''
        scan = np.array(obs_dict['scans']).reshape(-1)      # raw lidar value (1080,)
        speed = float(obs_dict['linear_vels_x'][0])         # longitudinal speed [m/s]
        if self.obs_noise:
            scan, speed = add_obs_noise(scan, speed, self.noise_rng, self.obs_cfg, self.speed_scale)
        observation = make_observation(scan, speed, self.obs_cfg)

        return observation
    # ======================================================================== #

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
