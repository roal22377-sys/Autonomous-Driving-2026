#!/usr/bin/env python

import os
import sys
import time
sim_path = 'src/f1tenth'
sys.path.append(sim_path)
from utils.vectorize import RunningMeanStd
from utils.observation import get_obs_config, make_observation, ActionHistory

import rclpy
import rclpy.duration
from rclpy.time import Time
from rclpy.node import Node
import torch
from easydict import EasyDict
from ruamel.yaml import YAML
from collections import deque

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

ODOM_TIMEOUT = 0.5      # [s] don't drive if the speed reading is older than this
VESC_SERVO_RANGE = (0.0, 1.0)   # VESC firmware clamps the servo output to this, whatever vesc.yaml says

def steering_command_limits(vesc_config):
    '''
      Steering commands the servo can actually reach: servo = gain*cmd + offset, clipped by
      vesc_driver to [servo_min, servo_max] and by the VESC to [0, 1]. With the straight-ahead
      offset 0.558 the left end is 1.0 - 0.558 = 0.442, less than the policy's 0.497.
      Returns (min_cmd, max_cmd), or None if the file can't be read.
    '''
    with open(vesc_config, 'r') as f:
      cfg = YAML().load(f)
    p = next(iter(cfg.values()))['ros__parameters']
    gain = float(p['steering_angle_to_servo_gain'])
    offset = float(p['steering_angle_to_servo_offset'])
    lo = max(float(p.get('servo_min', 0.0)), VESC_SERVO_RANGE[0])
    hi = min(float(p.get('servo_max', 1.0)), VESC_SERVO_RANGE[1])
    a, b = (lo - offset) / gain, (hi - offset) / gain
    # small margin: the command is float32, and 0.442 + 0.558 can land a hair above 1.0
    margin = 5e-4
    return min(a, b) + margin, max(a, b) - margin

def normalize_speed(speed, minimum, maximum):
    return (speed - (maximum + minimum)/2.0) / ((maximum - minimum)/2.0)

def unnormalize_speed(value, minimum, maximum):
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
    temp_a = np.ones_like(value)*temp_a
    temp_b = np.ones_like(value)*temp_b
    return temp_a*value + temp_b

class AgentNode(Node):
  def __init__(self):
    super().__init__('agent_node')
    self.declare_parameter('name', 'il?ppo?sac?')
    self.declare_parameter('algo_idx', 1)
    self.declare_parameter('scan_topic', 'scan')
    self.declare_parameter('drive_topic', 'drive')
    self.declare_parameter('odom_topic', 'odom')
    self.declare_parameter('speed_limit', 0.0)     # [m/s] cap on commanded speed for practice runs; <= 0 disables
    # [m/s] low-speed test mode (> 0 enables, needs speed_limit): the policy's capped range
    # [min_speed, speed_limit] is stretched to [speed_min, speed_limit] on the car, and /odom is mapped
    # back the other way, so the policy keeps seeing the speeds it was trained on. <= 0 disables.
    self.declare_parameter('speed_min', 0.0)
    # servo calibration, to clip steering to what the servo can reach (relative to ~/ros2_ws like sim_path)
    self.declare_parameter('vesc_config', 'src/f1tenth_ros/f1tenth_stack/config/vesc.yaml')

    self.scan_topic = self.get_parameter('scan_topic').value
    self.drive_topic = self.get_parameter('drive_topic').value
    self.odom_topic = self.get_parameter('odom_topic').value
    self.speed_limit = float(self.get_parameter('speed_limit').value)
    self.speed_min = float(self.get_parameter('speed_min').value)
    if self.speed_limit > 0:
      self.get_logger().info(f'[drive] speed limited to {self.speed_limit} m/s')

    # steering limits the servo can reach. Commands beyond them would be clipped by the VESC anyway;
    # clipping here keeps the action history (model input) equal to what the car really did.
    vesc_config = self.get_parameter('vesc_config').value
    try:
      self.steer_min, self.steer_max = steering_command_limits(vesc_config)
      self.get_logger().info(f'[drive] steering command limits from {vesc_config}: '
                             f'right {self.steer_min:+.3f}, left {self.steer_max:+.3f}')
    except Exception as e:
      self.steer_min, self.steer_max = -np.inf, np.inf
      self.get_logger().warn(f'[drive] could not read servo limits from {vesc_config} ({e}); '
                             'steering is not clipped here (the VESC still clips at the servo end)')

    # fix seed
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device
    if torch.cuda.is_available():
      self.device = torch.device('cuda:0')
      self.get_logger().info('[torch] cuda is used.')
    else:
      self.device = torch.device('cpu')
      self.get_logger().info('[torch] cpu is used.')
    
    # load parameters
    name = self.get_parameter('name').value
    algo_idx = self.get_parameter('algo_idx').value
    save_dir = os.path.join(sim_path, f"results/{name}/{algo_idx}/sim2real")
    backup_dir = os.path.join(sim_path, f"results/{name}/{algo_idx}/backup")
    with open(f'{backup_dir}/f1tenth.yaml', 'r') as f:
      self.args = EasyDict(YAML().load(f))

    # low-speed test mode: policy speed v in [min_speed, cap] <-> car speed in [speed_min, cap]
    self.remap = None
    if self.speed_min > 0:
      lo, cap = float(self.args.min_speed), self.speed_limit
      if cap > lo and 0 < self.speed_min < lo:
        self.remap = (lo, cap, self.speed_min)
        self.get_logger().warn(f'[drive] SPEED REMAP on: policy {lo:.2f}-{cap:.2f} m/s -> car '
                               f'{self.speed_min:.2f}-{cap:.2f} m/s (odom mapped back for the policy)')
      else:
        self.get_logger().error(f'[drive] speed_min {self.speed_min} ignored: needs 0 < speed_min < model '
                                f'min_speed {lo} < speed_limit {cap}')

    # observation
    '''
      obs_dim, lidar processing and speed input come from the model's backup
      f1tenth.yaml, so the real car builds observations exactly like in training.
    '''
    self.obs_dim = self.args.obs_dim
    self.obs_cfg = get_obs_config(self.args)       # raises if obs_dim doesn't match
    self.use_speed = self.obs_cfg['speed']['enabled']
    lidar_cfg = self.obs_cfg['lidar']
    self.get_logger().info(f"[obs] lidar 1080 -> {lidar_cfg['num_beams']} ({lidar_cfg['pooling']} pooling), "
                           f"max_range {lidar_cfg['max_range']} m, "
                           f"speed input: {'/' + self.odom_topic if self.use_speed else 'off'}, "
                           f"action history: {self.obs_cfg['history']['length']}, "
                           f"speed-limit input: {'on' if self.obs_cfg['speed_limit']['enabled'] else 'off'} (obs_dim {self.obs_dim})")
    # last commands this node sent (normalized), same as ActionHistory in utils/env.py
    self.history = ActionHistory(self.obs_cfg['history']['length'])

    # load algorithm
    try:
      self.obs_rms = RunningMeanStd('agent_obs', self.obs_dim)
      self.obs_rms.load(save_dir, required=True)
      self.actor = torch.load(f'{save_dir}/actor.pt').to(self.device)
      self.get_logger().info('[model] agent load success')
    except:
      self.get_logger().info('[model] agent load failed. directory does not exist')
      raise ValueError

    # subscribe
    self.laser_sub = self.create_subscription(LaserScan, self.scan_topic, self.scanCallback, 10)
    self.scan_queue = deque(maxlen=2)
    self.odom_speed, self.odom_time = None, None
    if self.use_speed:
      self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odomCallback, 10)

    # publish
    self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
    
    self.speed = 0.0
    self.steering_angle = 0.0

  def odomCallback(self, odom_msg: Odometry):
    self.odom_speed = odom_msg.twist.twist.linear.x
    self.odom_time = time.monotonic()

  def scanCallback(self, scan_msg: LaserScan):
    # the model needs the current speed; without a fresh reading, don't drive
    if self.use_speed and (self.odom_time is None or time.monotonic() - self.odom_time > ODOM_TIMEOUT):
      self.get_logger().warn(f'no /{self.odom_topic} in the last {ODOM_TIMEOUT}s, not publishing drive. '
                             'is vesc_to_odom_node running?', throttle_duration_sec=1.0)
      self.history.reset()
      return

    # make obs
    raw_scan = np.array(scan_msg.ranges[:1080])
    obs = self.getObs(raw_scan, self.odom_speed)

    # model inference
    with torch.no_grad():
      state_tensor = torch.tensor(self.obs_rms.normalize(obs), dtype=torch.float32, device=self.device)
      action = self.actor(state_tensor).loc.cpu().numpy()
    
    # transform action
    # policy angle [rad] -> servo command: calibration says command = angle * steer_cmd_gain
    action[0] *= self.args.max_steer * float(self.args.get('steer_cmd_gain', 1.0))
    # the servo can't go past these (left end is short of the policy's range with offset 0.558)
    action[0] = np.clip(action[0], self.steer_min, self.steer_max)
    action[1] = unnormalize_speed(action[1], self.args.min_speed, self.args.max_speed)
    if self.speed_limit > 0:
      # cap after unnormalizing, so the model's output mapping stays as trained
      action[1] = min(action[1], self.speed_limit)
    
    # publish drive (low-speed mode: the car gets the stretched speed, the history keeps the policy's)
    msg = AckermannDriveStamped()
    msg.header.stamp = scan_msg.header.stamp
    msg.drive.steering_angle = float(action[0])
    msg.drive.speed = float(self.policy_to_car_speed(action[1]))
    self.drive_pub.publish(msg)

    # remember what was actually sent (after the steering clip and speed cap), normalized like the policy output
    self.history.push([action[0] / (self.args.max_steer * float(self.args.get('steer_cmd_gain', 1.0))),
                       normalize_speed(action[1], self.args.min_speed, self.args.max_speed)])

  def policy_to_car_speed(self, v):
    '''policy [min_speed, cap] -> car [speed_min, cap], linear; identity when the remap is off'''
    if self.remap is None:
      return v
    lo, cap, vmin = self.remap
    return vmin + (v - lo) * (cap - vmin) / (cap - lo)

  def car_to_policy_speed(self, s):
    '''
      /odom speed -> the speed the policy believes it is driving. Inverse of policy_to_car_speed above
      speed_min; below it (starting from rest) 0..speed_min maps to 0..min_speed so the reading stays
      continuous and 0 stays 0.
    '''
    if self.remap is None:
      return s
    lo, cap, vmin = self.remap
    if s < vmin:
      return max(s, 0.0) * lo / vmin
    return lo + (s - vmin) * (cap - lo) / (cap - vmin)

  def getObs(self, raw_scan: np.ndarray, speed=None):
    '''
      Same processing as getObs in src/f1tenth/utils/env.py (without sim noise).
      speed: /odom linear.x [m/s], used only if the model has a speed input.
    '''
    # models trained with a speed-limit input are told the cap (agent.yaml speed_limit; <= 0 = none)
    speed = self.car_to_policy_speed(speed) if speed is not None else 0.0
    observation = make_observation(raw_scan, speed, self.obs_cfg, self.history,
                                   self.speed_limit if self.speed_limit > 0 else None)
    return observation

def main(args=None):
    rclpy.init(args=args)
    p = AgentNode()
    rclpy.spin(p)

if __name__ == '__main__':
    main()
