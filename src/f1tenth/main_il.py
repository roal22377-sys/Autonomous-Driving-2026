# backup
from shutil import copyfile

from copy import deepcopy
import argparse
import os
import time
import random
import numpy as np
import torch
import wandb
import pickle
from ruamel.yaml import YAML
from easydict import EasyDict

import gymnasium as gym

from utils.env import F1Wrapper
from utils.logger import Logger
from utils.video import create_video
from utils.rule_based import PurePursuitPlanner
from utils.color import cprint

from algorithm import algo_dict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'        # for debug
EPS = 1e-8

MAPS = ['Sochi', 'Spa', 'Nuerburgring', 'Monza', 'Melbourne', 'Austin', 
        'Silverstone', 'Sakhir', 'IMS', 'Budapest', 'Montreal', 'Sepang', 
        'Oschersleben', 'YasMarina', 'MoscowRaceway', 'Zandvoort', 'Catalunya', 
        'BrandsHatch', 'Shanghai', 'Hockenheim', 'SaoPaulo', 'Spielberg', 'MexicoCity']

EVAL_MAPS = ['Sochi', 'IMS', 'Catalunya', 'Zandvoort', 'Silverstone', 'SaoPaulo',       # easy
             'YasMarina', 'MoscowRaceway', 'Shanghai', 'MexicoCity', 'Montreal']        # hard

PARAM = {
    "mass": 3.463388126201571,
    "lf": 0.15597534362552312,
    "tlad": 0.82461887897713965,
    "vgain": 0.5,
}

def getParser():
    parser = argparse.ArgumentParser(description='F1tenth.')
    # mode
    parser.add_argument('--test',  action='store_true', help='for test.')
    # algorithm
    parser.add_argument('--name', type=str, default=None, help='algo name.')
    parser.add_argument('--algo_idx', type=int, default=1, help='algo index.')
    parser.add_argument('--model_num', type=int, default=None, help='for checkpoint num.')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--comment', type=str, default=None, help='wandb comment saved in run name.')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--log_freq', type=int, default=int(1e4), help='# of time steps for logging.')
    parser.add_argument('--save_freq', type=int, default=int(5e5), help='# of time steps for save.')
    # train
    parser.add_argument('--epochs', type=int, default=50, help='# of epochs for update.')
    parser.add_argument('--updates_per_epoch', type=int, default=1, help='# of gradient updates per epoch.')
    parser.add_argument('--threshold', type=int, default=200, help='minimum steps of expert rollout.')
    parser.add_argument('--total_steps', type=int, default=int(1e5), help='total interaction steps.')
    # for eval
    parser.add_argument('--eval_num', type=int, default=5, help='# of evaluation epochs.')
    parser.add_argument('--eval_freq', type=int, default=int(5e4), help='# of time steps for eval.')
    parser.add_argument('--save_video', action='store_true', help='save video.')
    parser.add_argument('--live_video', action='store_true', help='live video.')
    return parser

def eval(args, env, agent, name, eval_num=1):
    cprint('[Evaluation]\t{:^15}\tLap Time'.format('Map'), color='cyan')

    steps, scores, frames = [], [], []
    for _ in range(eval_num):
        step, score, done, _frames = 0, 0, False, []
        observation, info = env.reset()
        while not done and step < env._env.max_episode_steps:
            step += 1

            with torch.no_grad():
                clipped_action_tensor = agent.getAction(observation, deterministic=True)
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
                observation, reward, terminate, truncate, info = env.step(clipped_action)
            score += reward
            done = terminate or truncate
            _frames.append(env.render())

        if truncate or info['checkpoint_done']:
            print('[Success]   \t{:^15}\t{:>5s}'.format(env._env.map, f'{env._env.lap_times[0]:.2f}'))
        else:
            print('[Fail]      \t{:^15}\t{:>5s}'.format(env._env.map, f'{env._env.lap_times[0]:.2f}'))

        steps.append(step)
        scores.append(score)
        frames.append(_frames)

    max_score_idx = np.argmax(scores)
    step_mean, score_mean = np.mean(steps), np.mean(scores)
    cprint(f'[Eval]-{name} \t steps: {step_mean:.2f} \t score: {score_mean:.2f}', color='cyan')
    frames = frames[max_score_idx]
    if args.save_video:
        create_video(frames, output_name=f'{args.save_dir}/video/{name}')
    return step_mean, score_mean

def train(args):
    # backup
    copyfile('configs/task/f1tenth.yaml', f'{args.backup_dir}/f1tenth.yaml')
    copyfile('configs/task/dynamic.yaml', f'{args.backup_dir}/dynamic.yaml')
    copyfile(f'configs/algorithm/imitation.yaml', f'{args.backup_dir}/imitation.yaml')

    # for random seed
    np.random.seed(args.seed)
    random.seed(args.seed)    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define train environment (lidar noise only for expert data collection)
    train_env = F1Wrapper(args=args, maps=MAPS, scan_noise=True)

    # define eval environment
    render_mode = None
    if args.live_video:
        render_mode = "human_fast"
    if args.save_video:
        render_mode = "rgb_array"
    eval_env = F1Wrapper(args=args, maps=EVAL_MAPS, render_mode=render_mode)

    # set args value for env
    args.action_bound_min = train_env.action_space.low
    args.action_bound_max = train_env.action_space.high

    # define agent
    args.train_frequency = args.batch_size      # for code compatibility
    agent = algo_dict['imitation'](args)
    agent.load(args.model_num)

    # wandb
    if args.wandb:
        wandb.init(project='[F1tenth] Baselines', config=args)
        if args.comment is not None:
            wandb.run.name = f"{args.name}/{args.comment}"
        else:
            wandb.run.name = f"{args.name}/seed_{args.algo_idx}"

    # for log
    log_list = deepcopy(agent.log_list)
    rollout_logger = {arg: Logger(args.log_dir, arg) for arg in log_list['rollout']}
    train_logger = {arg: Logger(args.log_dir, arg) for arg in log_list['train']}

    # ======= collect trajectories ======= #
    # cache is tied to the observation settings so data made with different lidar
    # processing (e.g. the old raw 1080-dim expert_data.pkl) is never reused
    obs_config = {'obs_dim': args.obs_dim, 'lidar': train_env.lidar_cfg}
    expert_data_path = f'expert_data_obs{args.obs_dim}.pkl'
    expert_data = None
    if os.path.isfile(expert_data_path):
        with open(expert_data_path, 'rb') as f:
            expert_data = pickle.load(f)
        if expert_data.get('obs_config') != obs_config:
            cprint(f'\n[Rollout] {expert_data_path} was collected with different lidar settings. re-collecting.', color='red')
            expert_data = None

    if expert_data is not None:
        cprint(f'\n[Rollout] load {expert_data_path}.', bold=True, underline=True)
        observations = expert_data['observations']
        actions = expert_data['actions']
    else:
        cprint(f'\n[Rollout] start.', bold=True, underline=True)
        total_step = 0
        observations, actions = [], []
        while total_step < args.total_steps:

            observation, info = train_env.reset()
            track = train_env._env.unwrapped.track
            planner = PurePursuitPlanner(track=track, wb=0.17145 + 0.15875)
            ep_observations, ep_actions = [], []
            ep_step, ep_reward = 0, 0

            terminate = False
            while ep_step < args.max_episode_steps and not terminate:
                speed, steer = planner.plan(info['obs_dict'], PARAM)
                action = np.clip([steer/(train_env.max_steer+EPS), speed/train_env.max_speed], -np.ones(2), np.ones(2))
                ep_observations.append(observation)
                ep_actions.append(action)

                observation, reward, terminate, truncate, info = train_env.step(action)
                ep_step += 1
                ep_reward += reward

            if ep_step > args.threshold:
                observations.append(ep_observations)
                actions.append(ep_actions)
                total_step += ep_step
                print(f'[Total step] {total_step}: {ep_step}/{ep_reward:.2f}')

        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        with open(expert_data_path, 'wb') as f:
            pickle.dump({'observations':observations, 'actions':actions, 'obs_config':obs_config}, f)
    agent.step(observations, actions)
    train_env.close()
    # ==================================== #

    # ======= train ======= #
    best_score = -np.inf
    cprint(f'\n[Train] start.', bold=True, underline=True)
    for epoch in range(1, args.epochs+1):
        total_step = epoch*args.batch_size*args.updates_per_epoch
        losses = [agent.train()['actor_loss'] for _ in range(args.updates_per_epoch)]
        train_results = {'actor_loss': float(np.mean(losses))}
        cprint(f'[Train] epoch {epoch} \t actor_loss: {train_results["actor_loss"]:.5f}', color='yellow')

        # logging
        for key, value in train_results.items():
            if key in log_list['train']:
                train_logger[key].write(epoch, value)

        log_data = {
                'rollout/step': total_step,
                'train/best': 0         # if best model updated, becomes True
            }
        
        # evaluation
        eval_len, eval_score = eval(args, eval_env, agent, name=epoch, eval_num=args.eval_num)
        rollout_logger['score'].write(eval_len, eval_score)
        rollout_logger['ep_len'].write(eval_len, eval_len)

        # save best model
        if eval_score > best_score:
            cprint(f'[{args.name}] save best model.', bold=True, color="green")
            agent.save(log=False)
            best_score = eval_score
            log_data['train/best'] = 1

        # logger
        for arg, logger in rollout_logger.items():
            if arg == 'step': continue
            log_data[f'rollout/{arg}'] = logger.get_avg()
        for arg, logger in train_logger.items():
            log_data[f'train/{arg}'] = logger.get_avg()
        print(log_data)
        if args.wandb:
            wandb.log(log_data)

        # save
        agent.save(model_num=int(total_step))
        for logger in rollout_logger.values():
            logger.save()
        for logger in train_logger.values():
            logger.save()

    eval_env.close()


def test(args):
    # for random seed
    np.random.seed(args.seed)
    random.seed(args.seed)    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define Environment
    render_mode = None
    if args.live_video:
        render_mode = "human_fast"
    if args.save_video:
        render_mode = "rgb_array"
    args.max_episode_steps = 10000
    test_env = F1Wrapper(args=args, maps=args.map if args.map else EVAL_MAPS, render_mode=render_mode)

    # set args value for env
    args.action_bound_min = test_env.action_space.low
    args.action_bound_max = test_env.action_space.high

    # define agent
    args.train_frequency = args.batch_size      # for code compatibility
    agent = algo_dict['imitation'](args)
    agent.load(args.model_num)

    # test
    eval(args, test_env, agent, name='test', eval_num=args.eval_num)


if __name__ == "__main__":
    # base configurations
    parser = getParser()
    args = parser.parse_args()
    if args.name == None:
        args.name = 'imitation'
    args = EasyDict(vars(args))

    # load configurations & merge them
    with open('configs/task/f1tenth.yaml', 'r') as f:
        task_args = EasyDict(YAML().load(f))
    with open('configs/task/dynamic.yaml', 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))
    with open(f'configs/algorithm/imitation.yaml', 'r') as f:
        agent_args = EasyDict(YAML().load(f))
    args.update(task_args)
    args.update(dynamic_args)
    args.update(agent_args)
    args.save_dir = f"results/{args.name}/{args.algo_idx}"

    # ==== processing args ==== #
    # directory
    args.log_dir = f'{args.save_dir}/logs'
    args.video_dir = f'{args.save_dir}/video'
    args.backup_dir = f'{args.save_dir}/backup'
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.backup_dir, exist_ok=True)
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_idx}"
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device('cuda:0')
        cprint('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        cprint('[torch] cpu is used.')
    args.device = device

    if args.test:
        test(args)
    else:
        train(args)