#!/usr/bin/env python

import os
import sys
sim_path = 'src/f1tenth'
sys.path.append(sim_path)
from utils.vectorize import RunningMeanStd
from utils.lidar import get_lidar_config, process_scan

import rclpy
import rclpy.duration
from rclpy.time import Time
from rclpy.node import Node
import torch
from easydict import EasyDict
from ruamel.yaml import YAML
from collections import deque

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

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

    self.scan_topic = self.get_parameter('scan_topic').value
    self.drive_topic = self.get_parameter('drive_topic').value

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

    # observation
    '''
      obs_dim and lidar processing come from the model's backup f1tenth.yaml,
      so the real car always processes scans exactly like the model was trained.
    '''
    self.obs_dim = self.args.obs_dim
    self.lidar_cfg = get_lidar_config(self.args)
    if self.obs_dim != self.lidar_cfg['num_beams']:
      raise ValueError(f"obs_dim ({self.obs_dim}) != lidar.num_beams ({self.lidar_cfg['num_beams']}) in {backup_dir}/f1tenth.yaml")
    self.get_logger().info(f"[obs] lidar 1080 -> {self.lidar_cfg['num_beams']} ({self.lidar_cfg['pooling']} pooling), "
                           f"max_range {self.lidar_cfg['max_range']} m")

    # load algorithm
    try:
      self.obs_rms = RunningMeanStd('agent_obs', self.obs_dim)
      self.obs_rms.load(save_dir)
      self.actor = torch.load(f'{save_dir}/actor.pt').to(self.device)
      self.get_logger().info('[model] agent load success')
    except:
      self.get_logger().info('[model] agent load failed. directory does not exist')
      raise ValueError

    # subscribe
    self.laser_sub = self.create_subscription(LaserScan, self.scan_topic, self.scanCallback, 10)
    self.scan_queue = deque(maxlen=2)
    
    # publish
    self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
    
    self.speed = 0.0
    self.steering_angle = 0.0

  def scanCallback(self, scan_msg: LaserScan):
    # make obs
    raw_scan = np.array(scan_msg.ranges[:1080])
    obs = self.getObs(raw_scan)

    # model inference
    with torch.no_grad():
      state_tensor = torch.tensor(self.obs_rms.normalize(obs), dtype=torch.float32, device=self.device)
      action = self.actor(state_tensor).loc.cpu().numpy()
    
    # transform action
    action[0] *= self.args.max_steer
    action[1] = unnormalize_speed(action[1], self.args.min_speed, self.args.max_speed)
    
    # publish drive
    msg = AckermannDriveStamped()
    msg.header.stamp = scan_msg.header.stamp
    msg.drive.steering_angle = float(action[0])
    msg.drive.speed = float(action[1])
    self.drive_pub.publish(msg)

  def getObs(self, raw_scan: np.ndarray):
    '''
      Same processing as getObs in src/f1tenth/utils/env.py (without sim noise).
    '''
    observation = process_scan(raw_scan, **self.lidar_cfg)
    return observation

def main(args=None):
    rclpy.init(args=args)
    p = AgentNode()
    rclpy.spin(p)

if __name__ == '__main__':
    main()
