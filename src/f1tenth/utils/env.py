'''
    This F1Wrapper is for
        1. usage issue (gymnasium style)
        2. define observation & action space
        3. observation processing (concatenate lidar + other properties)
        4. sim-to-real gaps measured on the car (sim_mode='train' / 'eval'):
             steering / speed command delay, real acceleration limit, friction,
             lidar + odom noise, randomized tracks (width, open sides, both directions)

        ->  outputs of step function should have (n_agents, dim) shape
            ex) observations, rewards, terminates, truncates, infos = env.step([agent_1_action,
                                                                                agent_2_action,
                                                                                ,,,
                                                                                agent_n_action])
'''

from collections import deque, OrderedDict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.observation import get_obs_config, make_observation, add_obs_noise, ActionHistory

EPS = 1e-8
MAX_TRACK_CACHE = 24

def unnormalize_speed(value, minimum, maximum):
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
    temp_a = np.ones_like(value)*temp_a
    temp_b = np.ones_like(value)*temp_b
    return temp_a*value + temp_b

def normalize_speed(speed, minimum, maximum):
    return (speed - (maximum + minimum)/2.0) / ((maximum - minimum)/2.0)


class F1Wrapper(gym.Wrapper):
    def __init__(self, args, maps, render_mode=None, obs_noise=False, sim_mode='ideal', tracks=None) -> None:
        '''
            obs_noise: add lidar noise/dropout and speed reading error (see `lidar`,
                       `speed_obs` in f1tenth.yaml).
            sim_mode:  'ideal' - the plain simulator (old behaviour)
                       'train' - sim2real gaps sampled per episode from `sim2real` ranges
                       'eval'  - sim2real gaps fixed at their nominal values
            tracks:    list of generated track files (utils/track_gen.py); None = from `maps`
                       when sim2real.generated_tracks is on, else the original maps
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
        self.timestep = args.timestep
        self.laps = int(args.get('laps', 1))

        # for spaces
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.observation_space = spaces.Box(-np.inf*np.ones(self.obs_dim), np.inf*np.ones(self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)

        # for observation (lidar + speed + action history), checks obs_dim
        self.obs_cfg = get_obs_config(args)
        self.obs_noise = obs_noise
        self.noise_rng = np.random.default_rng(args.get('seed', None))
        self.speed_scale = 1.0
        self.lidar_override = {}
        self.history = ActionHistory(self.obs_cfg['history']['length'])

        # sim2real
        self.sim_mode = sim_mode
        self.s2r = dict(args.get('sim2real', None) or {})
        self.base_params = dict(self._env.unwrapped.params)
        self.steer_delay_steps, self.speed_delay_steps = 0, 0
        self.track_files = None
        if sim_mode != 'ideal' and self.s2r.get('generated_tracks', False):
            from utils.track_gen import list_variants
            self.track_files = tracks if tracks is not None else list_variants(maps)
            if len(self.track_files) == 0:
                raise FileNotFoundError('no generated tracks: run `python -m utils.track_gen 6 16` in src/f1tenth')
        elif tracks is not None:
            self.track_files = tracks
        self._track_cache = OrderedDict()
        self.base_max_episode_steps = self._env.unwrapped.max_episode_steps
        # per-episode speed cap (configs/task/f1tenth.yaml speed_limit), applied like agent_node.py
        self.sl_cfg = dict(args.get('speed_limit', None) or {})
        self.speed_cap = self.max_speed
        self.fixed_speed_cap = None         # set by evaluation code to force a cap

    # ---------------------------------------------------------------- sim2real
    def _rand(self, key, default):
        '''sample a range (train) or take the nominal value (eval) of a sim2real entry.'''
        if self.sim_mode == 'train' and key in self.s2r:
            lo, hi = self.s2r[key]
            return float(self.noise_rng.uniform(lo, hi))
        return float(self.s2r.get(f'{key}_nominal', default))

    def _apply_sim2real(self):
        params = dict(self.base_params)
        if self.sim_mode == 'ideal':
            self.steer_delay_steps = self.speed_delay_steps = 0
        else:
            self.steer_delay_steps = int(round(self._rand('steer_delay', 0.0) / self.timestep))
            self.speed_delay_steps = int(round(self._rand('speed_delay', 0.0) / self.timestep))
            params['a_max'] = self._rand('a_max', params['a_max'])
            if self.sim_mode == 'train' and 'mu_scale' in self.s2r:
                params['mu'] = params['mu'] * float(self.noise_rng.uniform(*self.s2r['mu_scale']))
        self._env.unwrapped.sim.update_params(params)
        self.params = params
        self.steer_queue = deque([0.0] * self.steer_delay_steps)
        self.speed_queue = deque([0.0] * self.speed_delay_steps)

    def _load_track(self, path):
        if path in self._track_cache:
            self._track_cache.move_to_end(path)
            return self._track_cache[path]
        from utils.track_gen import to_track
        track = to_track(path)
        self._track_cache[path] = track
        if len(self._track_cache) > MAX_TRACK_CACHE:
            self._track_cache.popitem(last=False)
        return track

    # ---------------------------------------------------------------- pose
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
        self.delta_s = 0.0

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
        elif abs(self.delta_s) > total_track_s/2.0 and self.delta_s > 0:
            # moved backwards across the start point
            self.delta_s -= total_track_s
        if abs(self.delta_s) > 1.0:
            # a jump to another part of the centerline (projection glitch), not real motion
            self.delta_s = 0.0

        # frenet coordinate pose
        self.position_frenet = np.stack([poses_s, poses_d]).T
        self.yaw_frenet = poses_phi

    def reset(self, **kwargs):
        if self.obs_noise:
            # per-episode speed reading error, like an imperfect odom calibration
            self.speed_scale = self.noise_rng.uniform(*self.obs_cfg['speed']['scale_range'])
            lid = self.obs_cfg['lidar']
            k = float(self.noise_rng.uniform(*self.s2r.get('noise_scale', [1.0, 1.0])))
            rot = lid.get('rotation_deg', 0.0)
            self.lidar_override = {
                'noise_std': lid['noise_std']*k, 'dropout_prob': lid['dropout_prob']*k,
                'spike_short_prob': lid.get('spike_short_prob', 0.0)*k,
                'spike_long_prob': lid.get('spike_long_prob', 0.0)*k,
                'rotation_beams': int(round(self.noise_rng.uniform(-rot, rot) / 0.25)) if rot > 0 else 0,
            }
        # track: a generated variant (random or chosen with options={'track': path})
        options = kwargs.pop('options', None)
        options = dict(options) if options is not None else None
        if self.track_files is not None and (options is None or 'map' not in options):
            path = (options or {}).pop('track', None) or self.track_files[self.noise_rng.integers(len(self.track_files))]
            options = dict(options or {})
            options['map'] = self._load_track(path)
            self.track_file = path
        else:
            self.track_file = None
        self._apply_sim2real()
        self._sample_speed_cap()
        self.history.reset()
        self.last_action = np.zeros(2, dtype=np.float32)
        self.prev_cmd = np.zeros(2, dtype=np.float32)
        if self.track_file is not None and 'poses' not in options:
            options['poses'] = self._start_pose(options['map'])

        # laps are counted from progress along the track (progress_laps), not from the start-line
        # band of f110_env, which misses laps when the start is off-center or the track is wide
        self.progress_laps = self.sim_mode != 'ideal'
        self._env.unwrapped.num_laps = 10**6 if self.progress_laps else self.laps
        self.progress, self.elapsed, self.success, self.lap_times = 0.0, 0.0, False, []
        obs_dict, info = self._env.reset(options=options, **kwargs)
        # enough steps for all laps even at a slow 1.6 m/s
        if self.s2r.get('laps_steps', False) and self.sim_mode != 'ideal':
            length = float(self._env.unwrapped.track.centerline.spline.s[-1])
            self._env.unwrapped.max_episode_steps = int(self.laps*length/1.6/self.timestep) + 400
        else:
            self._env.unwrapped.max_episode_steps = self.base_max_episode_steps
        self.max_episode_steps = self._env.unwrapped.max_episode_steps
        if self.show_centerline:
            self._env.unwrapped.add_render_callback(self._env.track.centerline.render_waypoints)
        self._reset_pose(obs_dict)
        obs = self.getObs(obs_dict, reset=True)
        info['obs_dict'] = obs_dict
        info['privileged'] = self.privileged()
        return obs, info

    # ============================ implement here ============================ #
    # TODO
    # You need to implement your own reward function.
    PROGRESS_COEFF = 10.0     # reward per meter of centerline progress (8 m/s * 0.025 s -> ~2.0/step)
    COLLISION_COST = 10.0
    STEER_RATE_COST = 0.0     # per (change of normalized steering command)^2; PPO fine-tuning sets it

    def calc_reward(self, action=None):
        # forward reward: progress along the centerline (frenet s) this step [m].
        # Rewards speed, not mere survival; driving backwards gives a negative reward.
        forward_reward = self.PROGRESS_COEFF*float(self.delta_s)

        # collison cost
        collision_cost = self.COLLISION_COST if self.collision else 0.0

        # smoothness: penalize steering command jumps (weaving costs time and is fragile on the car)
        steer_cost = 0.0
        if action is not None and self.STEER_RATE_COST > 0:
            steer_cost = self.STEER_RATE_COST*float(action[0] - self.prev_cmd[0])**2

        # final reward
        reward = forward_reward - collision_cost - steer_cost
        reward_dict = {'forward_reward': forward_reward,
                       'collision_cost': collision_cost,
                       'steer_cost': steer_cost}
        return reward, reward_dict
    # ======================================================================== #

    def step(self, action:np.array):
        '''
            Original env's action : [desired steer, desired speed]
                                    steer : -1(left) ~ 1(right)
                                    speed : ~inf ~ inf

            Changed to:             [desired steer, normalized speed]
                                    steer : -1(left) ~ 1(right)
                                    speed : -1 ~ 1              -> mapped linearly to [min_speed, max_speed]

            With sim2real delays, the command reaches the car steer_delay / speed_delay later.
        '''
        cmd = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:2], -1.0, 1.0)
        self.prev_cmd = self.last_action
        self.last_action = cmd.copy()

        steer = float(cmd[0])*self.max_steer
        speed = min(float(unnormalize_speed(cmd[1], self.min_speed, self.max_speed)), self.speed_cap)
        # history = what was actually sent (after the cap), exactly like agent_node.py
        self.history.push([cmd[0], normalize_speed(speed, self.min_speed, self.max_speed)])
        if self.steer_delay_steps > 0:
            self.steer_queue.append(steer)
            steer = self.steer_queue.popleft()
        if self.speed_delay_steps > 0:
            self.speed_queue.append(speed)
            speed = self.speed_queue.popleft()
        _action = np.array([steer, speed], dtype=np.float32)

        obs_dict, _, terminate, truncate, info = self._env.step(_action)
        self._step_pose(obs_dict)
        self.elapsed += self.timestep
        if self.progress_laps:
            length = float(self._env.track.centerline.spline.s[-1])
            self.progress += float(self.delta_s)
            if self.progress >= (len(self.lap_times) + 1) * length:
                self.lap_times.append(self.elapsed)
            if self.progress >= self.laps * length and not self.collision:
                self.success, terminate = True, True
        else:
            self.success = bool(self._env.unwrapped.toggle_list[0] >= 2 * self.laps)
            self.lap_times = list(self._env.unwrapped.lap_split_times)
        obs = self.getObs(obs_dict)
        reward, reward_dict = self.calc_reward(cmd)
        info['obs_dict'] = obs_dict
        info.update(reward_dict)
        info['lap_count'] = len(self.lap_times)
        info['lap_split_times'] = list(self.lap_times)
        info['success'] = self.success
        info['privileged'] = self.privileged()
        return obs, reward, terminate, truncate, info

    START_MAX_HEADING = 0.35      # [rad] heading error at random starts
    start_mode = None              # None: random offset in 'train', centerline otherwise; or 'center' / 'random'

    def _start_pose(self, track):
        '''
            Start pose on a generated track. Random starts (train, or start_mode='random') put the
            car anywhere across the drivable width with a heading error, so the policy learns to
            merge onto the race line from far away; otherwise on the centerline, heading along it.
        '''
        xy = np.stack([track.centerline.xs, track.centerline.ys], axis=1).astype(np.float64)
        n = len(xy)
        i = int(self.noise_rng.integers(n))
        t = xy[(i + 1) % n] - xy[i - 1]
        yaw = float(np.arctan2(t[1], t[0]))
        mode = self.start_mode or ('random' if self.sim_mode == 'train' else 'center')
        x, y, th = xy[i, 0], xy[i, 1], yaw
        if mode == 'random' and hasattr(track, 'w_left'):
            dh = float(self.noise_rng.uniform(-self.START_MAX_HEADING, self.START_MAX_HEADING))
            # keep the 0.30 x 0.60 body clear of the wall it points at: half width 0.15 + 0.15, plus
            # the sideways reach of the front corner (~0.8 m ahead incl. the first moments of driving)
            margin = 0.3 + 0.8*abs(np.sin(dh))
            lo, hi = -(float(track.w_right[i]) - margin), float(track.w_left[i]) - margin
            off = float(self.noise_rng.uniform(lo, hi)) if hi > lo else 0.0
            x, y = x - off*np.sin(yaw), y + off*np.cos(yaw)
            th = yaw + dh
        return np.array([[x, y, th]])

    def _sample_speed_cap(self):
        if self.fixed_speed_cap is not None:
            self.speed_cap = float(self.fixed_speed_cap)
        elif not self.sl_cfg.get('enabled', False) or self.sim_mode == 'ideal':
            self.speed_cap = self.max_speed
        elif self.sim_mode == 'train':
            if self.noise_rng.random() < self.sl_cfg.get('uncapped_prob', 0.0):
                self.speed_cap = self.max_speed
            else:
                self.speed_cap = float(self.noise_rng.uniform(*self.sl_cfg.get('range', [self.max_speed]*2)))
        else:
            self.speed_cap = float(self.sl_cfg.get('nominal', self.max_speed))
        self.speed_cap = min(self.speed_cap, self.max_speed)

    PRIV_DIM = 14

    def privileged(self):
        '''
            Simulator-only state for an asymmetric critic (PPO): never given to the actor.
            [ey, heading error, speed, steer, yaw rate, slip, steer delay, a_max, mu, speed cap,
             centerline curvature 1 / 3 / 6 / 10 m ahead]
        '''
        st = self.vehicle_state()
        cl = self._env.unwrapped.track.centerline.spline
        L = float(cl.s[-1])
        s0 = float(self.position_frenet[0])
        curv = [float(cl.calc_curvature((s0 + d) % L)) for d in (1.0, 3.0, 6.0, 10.0)]
        yaw_err = (float(self.yaw_frenet) + np.pi) % (2*np.pi) - np.pi
        return np.array([float(self.position_frenet[1]), yaw_err, st[3], st[2], st[5], st[6],
                         self.steer_delay_steps*self.timestep, self.params['a_max'], self.params['mu'],
                         self.speed_cap] + curv, dtype=np.float32)

    # ---------------------------------------------------------------- expert support
    def pending_commands(self):
        '''commands already sent but not yet applied (oldest first): steer [rad], speed [m/s]'''
        return list(self.steer_queue), list(self.speed_queue)

    def vehicle_state(self):
        '''true simulator state [x, y, steer, v, yaw, yaw_rate, slip]'''
        return np.array(self._env.unwrapped.sim.agents[0].state, dtype=np.float64)

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

            ==> Changed to [downsampled lidar (lidar.num_beams), speed (if speed_obs.enabled),
                            action history (2*action_history.length)]
                built by utils/observation.py, the same code agent_node.py uses.
        '''
        scan = np.array(obs_dict['scans']).reshape(-1)      # raw lidar value (1080,)
        speed = float(obs_dict['linear_vels_x'][0])         # longitudinal speed [m/s]
        if self.obs_noise:
            scan, speed = add_obs_noise(scan, speed, self.noise_rng, self.obs_cfg, self.speed_scale,
                                        lidar_override=self.lidar_override)
        observation = make_observation(scan, speed, self.obs_cfg, self.history, self.speed_cap)

        return observation
    # ======================================================================== #

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
