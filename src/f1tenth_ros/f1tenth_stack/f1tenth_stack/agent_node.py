#!/usr/bin/env python

import os
import sys
import time
sim_path = 'src/f1tenth'
sys.path.append(sim_path)
from utils.vectorize import RunningMeanStd
from utils.observation import get_obs_config, make_observation, ActionHistory, get_kick_config, apply_start_kick

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
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

ODOM_TIMEOUT = 0.5      # [s] don't drive if the speed reading is older than this
TELEOP_HOLD = 0.25      # [s] a /teleop message newer than this = the keyboard (mux priority 100, timeout 0.2 s) has the car
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
    self.declare_parameter('teleop_topic', 'teleop')   # keyboard commands (mux, higher priority than the agent)
    self.declare_parameter('speed_limit', 0.0)     # [m/s] cap on commanded speed for practice runs; <= 0 disables
    # [m/s] low-speed test mode (> 0 enables, needs speed_limit): the policy's capped range
    # [min_speed, speed_limit] is stretched to [speed_min, speed_limit] on the car, and /odom is mapped
    # back the other way, so the policy keeps seeing the speeds it was trained on. <= 0 disables.
    self.declare_parameter('speed_min', 0.0)
    # extra steering on the car, per side (1.0 = as calibrated). The real car turns less than the sim
    # for the same command (11/23 log: left 0.74, right 0.89 of the sim's yaw rate), so the command is
    # multiplied by these before the servo clip. The action history gets the command divided back, so
    # the policy still sees the steering it asked for.
    self.declare_parameter('steer_gain_left', 1.0)
    self.declare_parameter('steer_gain_right', 1.0)
    # servo calibration, to clip steering to what the servo can reach (relative to ~/ros2_ws like sim_path)
    self.declare_parameter('vesc_config', 'src/f1tenth_ros/f1tenth_stack/config/vesc.yaml')
    # plan B (MAP_TOOL.md): models with the track-preview input need the course and its map (tools/map_session.sh)
    self.declare_parameter('course', '')          # e.g. src/f1tenth/installation/maps_real/Practice1126.npz
    self.declare_parameter('map_dir', '')         # e.g. maps/run_1126_1030 (map.yaml / map.pgm of the same run)
    self.declare_parameter('laser_x', 0.13)       # [m] base_link -> laser (bringup_launch.py static transform)
    self.declare_parameter('loc_stride', 3)       # every n-th beam for the scan matching (4 if a step takes > 10 ms)
    # localization lost longer than this [s] -> speed 0 (brake, the model keeps steering) until it is valid again.
    # The 11/25 preview model drives poorly on LiDAR alone (IL 13 r3, whole run without localization: 9 % of the
    # eval tracks at cap 8, 53 % of the short courses). <= 0: keep driving
    self.declare_parameter('lost_stop', 0.3)
    # localization lost longer than lost_switch [s] -> a LiDAR-only model drives instead (fallback_name / _algo_idx, at
    # most fallback_speed_limit) while the localization keeps searching; back to this model once the pose is valid again.
    # Set -> lost_stop is not used. '' = no fallback (lost_stop applies). 11/26: PPO 12 u190 drove 143 m clean at cap 4
    self.declare_parameter('fallback_name', '')
    self.declare_parameter('fallback_algo_idx', 0)
    self.declare_parameter('fallback_speed_limit', 4.0)
    self.declare_parameter('lost_switch', 0.3)

    self.scan_topic = self.get_parameter('scan_topic').value
    self.drive_topic = self.get_parameter('drive_topic').value
    self.odom_topic = self.get_parameter('odom_topic').value
    self.speed_limit = float(self.get_parameter('speed_limit').value)
    if 0 < self.speed_limit < 2.0:
      # trained with caps 2-8 m/s (speed-limit input) and a 2.0 m/s start kick: below 2.0 both are outside training
      self.get_logger().warn(f'[drive] speed_limit {self.speed_limit} < 2.0 m/s: outside the trained caps (2-8) and below '
                             'the start kick (2.0 m/s); use speed_min (low-speed mode) for slower runs')
    self.speed_min = float(self.get_parameter('speed_min').value)
    self.gain_left = float(self.get_parameter('steer_gain_left').value)
    self.gain_right = float(self.get_parameter('steer_gain_right').value)
    if (self.gain_left, self.gain_right) != (1.0, 1.0):
      self.get_logger().warn(f'[drive] STEER GAIN on: left x{self.gain_left:.2f}, right x{self.gain_right:.2f}')
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
    # policy angle [rad] -> servo command, per side (models from 11/23 on); older models: one gain
    g = float(self.args.get('steer_cmd_gain', 1.0))
    self.cmd_gain = (float(self.args.get('steer_cmd_gain_left', g)), float(self.args.get('steer_cmd_gain_right', g)))
    self.get_logger().info(f'[drive] max_steer {self.args.max_steer} rad, command gain left {self.cmd_gain[0]:.3f} '
                           f'right {self.cmd_gain[1]:.3f}')
    # start kick from the model's config (same as the sim); models without it (up to IL run 8) get none
    self.kick_cfg = get_kick_config(self.args)
    if self.kick_cfg:
      self.get_logger().info(f"[drive] start kick: commands < {self.kick_cfg['speed']} m/s become "
                             f"{self.kick_cfg['speed']} m/s while odom < {self.kick_cfg['below']} m/s")
    self.obs_cfg = get_obs_config(self.args)       # raises if obs_dim doesn't match
    self.use_speed = self.obs_cfg['speed']['enabled']
    lidar_cfg = self.obs_cfg['lidar']
    self.get_logger().info(f"[obs] lidar 1080 -> {lidar_cfg['num_beams']} ({lidar_cfg['pooling']} pooling"
                           f"{', zones ' + str(lidar_cfg['zones']) if lidar_cfg.get('zones') else ''}), "
                           f"max_range {lidar_cfg['max_range']} m, "
                           f"speed input: {'/' + self.odom_topic if self.use_speed else 'off'}, "
                           f"action history: {self.obs_cfg['history']['length']}, "
                           f"speed-limit input: {'on' if self.obs_cfg['speed_limit']['enabled'] else 'off'} (obs_dim {self.obs_dim})")
    # last commands this node sent (normalized), same as ActionHistory in utils/env.py
    self.history = ActionHistory(self.obs_cfg['history']['length'])

    # track preview (plan B): localization on the course map -> race line / speed plan ahead (utils/localizer.py)
    self.carpv, self.odom_pose = None, None
    self.loc_pub = None
    if self.obs_cfg['preview']['enabled']:
      course, map_dir = self.get_parameter('course').value, self.get_parameter('map_dir').value
      if not course or not map_dir:
        raise RuntimeError('this model uses the track preview: set the course (.npz) and map_dir parameters '
                           '(agent_preview.yaml, MAP_TOOL.md)')
      from utils.localizer import CarPreview
      dyn = os.path.join(backup_dir, 'dynamic.yaml')
      lidar_off = 0.17
      if os.path.exists(dyn):
        with open(dyn) as f:
          lidar_off = float(YAML().load(f).get('lidar_pos_offset', 0.17))
      t0 = time.monotonic()
      self.carpv = CarPreview(course, map_dir, self.obs_cfg['preview'], laser_x=float(self.get_parameter('laser_x').value),
                              lidar_pos_offset=lidar_off, log=self.get_logger().info,
                              stride=int(self.get_parameter('loc_stride').value))
      self.get_logger().info(f'[preview] course {course}, map {map_dir} ({len(self.carpv.loc.pts)} wall cells, '
                             f'{time.monotonic() - t0:.1f} s); the start pose is searched at the first scan')
      self.loc_pub = self.create_publisher(PoseStamped, 'loc_pose', 10)
      # RViz "2D Pose Estimate" (or tools/loc_gui.py on a PC without ROS): search the start pose again, near this pose
      self.pose_hint = None
      self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.onInitialPose, 10)
      self.loc_last_report = time.monotonic()
      self.start_try = -1e9               # last start search (they take 0.3 s+ each: at most every 0.5 s)
      self.lost_since = None
      self.lost_stop = float(self.get_parameter('lost_stop').value)

    # load algorithm
    try:
      self.obs_rms = RunningMeanStd('agent_obs', self.obs_dim)
      self.obs_rms.load(save_dir, required=True)
      self.actor = torch.load(f'{save_dir}/actor.pt').to(self.device)
      self.get_logger().info('[model] agent load success')
    except Exception as e:
      self.get_logger().error(f'[model] agent load failed from {save_dir}: {e!r}')
      raise
    # one forward pass now: the first real call took 1.8-3.1 s on the car (11/24 logs) while ~10 scans queued up
    with torch.no_grad():
      self.actor(torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device))
    self.prim = EasyDict(args=self.args, obs_cfg=self.obs_cfg, history=self.history, rms=self.obs_rms, actor=self.actor,
                         cmd_gain=self.cmd_gain, name=f'{name}/{algo_idx}')
    self.fb, self.fb_on = None, False
    fb_name = self.get_parameter('fallback_name').value
    if self.carpv is not None and fb_name:
      self.fb = self.loadPolicy(fb_name, int(self.get_parameter('fallback_algo_idx').value))
      self.fb_cap = float(self.get_parameter('fallback_speed_limit').value)
      self.lost_switch = float(self.get_parameter('lost_switch').value)
      self.get_logger().info(f'[fallback] localization lost > {self.lost_switch:.1f} s -> {self.fb.name} drives '
                             f'(LiDAR only, cap {self.fb_cap:.1f} m/s) until the pose is valid again; lost_stop not used')

    # subscribe
    # newest scan only (a slow callback must not work through a backlog of old scans)
    scan_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
    self.laser_sub = self.create_subscription(LaserScan, self.scan_topic, self.scanCallback, scan_qos)
    self.scan_checked = False
    # while the keyboard holds or drives the car, the commands sent here are not executed: keep the action history
    # empty then (like a simulator reset) instead of filling it with commands the car never ran. 11/24 run 1515: behind
    # the hold the history fed back on itself (steering 0 -> 0.42 rad with the car standing) and the car set off at 0.40 rad
    self.teleop_time = None
    self.teleop_sub = self.create_subscription(AckermannDriveStamped, self.get_parameter('teleop_topic').value,
                                               self.teleopCallback, 10)
    self.odom_speed, self.odom_time = None, None
    if self.use_speed or self.obs_cfg['preview']['enabled']:
      self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odomCallback, 10)

    # publish
    self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
    
    self.speed = 0.0
    self.steering_angle = 0.0

  def loadPolicy(self, name, algo_idx):
    '''a second, LiDAR-only model (no preview input) with its own observation config, normalizer and history'''
    save_dir = os.path.join(sim_path, f'results/{name}/{algo_idx}/sim2real')
    backup_dir = os.path.join(sim_path, f'results/{name}/{algo_idx}/backup')
    with open(f'{backup_dir}/f1tenth.yaml', 'r') as f:
      args = EasyDict(YAML().load(f))
    obs_cfg = get_obs_config(args)
    if obs_cfg['preview']['enabled']:
      raise RuntimeError(f'fallback {name}/{algo_idx} uses the track preview: it must be a LiDAR-only model')
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load(save_dir, required=True)
    actor = torch.load(f'{save_dir}/actor.pt').to(self.device)
    with torch.no_grad():
      actor(torch.zeros(args.obs_dim, dtype=torch.float32, device=self.device))
    g = float(args.get('steer_cmd_gain', 1.0))
    cmd_gain = (float(args.get('steer_cmd_gain_left', g)), float(args.get('steer_cmd_gain_right', g)))
    self.get_logger().info(f'[fallback] loaded {name}/{algo_idx} (obs_dim {args.obs_dim})')
    return EasyDict(args=args, obs_cfg=obs_cfg, history=ActionHistory(obs_cfg['history']['length']), rms=rms,
                    actor=actor, cmd_gain=cmd_gain, name=f'{name}/{algo_idx}')

  def teleopCallback(self, msg: AckermannDriveStamped):
    self.teleop_time = time.monotonic()

  def odomCallback(self, odom_msg: Odometry):
    self.odom_speed = odom_msg.twist.twist.linear.x
    self.odom_time = time.monotonic()
    if self.carpv is not None:
      q = odom_msg.pose.pose.orientation
      pose = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y,
              float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))
      if np.all(np.isfinite(pose)):          # a broken message must not reach the localization
        self.odom_pose = pose

  def scanCallback(self, scan_msg: LaserScan):
    # the model needs the current speed; without a fresh reading, don't drive
    if self.use_speed and (self.odom_time is None or time.monotonic() - self.odom_time > ODOM_TIMEOUT):
      self.get_logger().warn(f'no /{self.odom_topic} in the last {ODOM_TIMEOUT}s, not publishing drive. '
                             'is vesc_to_odom_node running?', throttle_duration_sec=1.0)
      self.resetHistories()
      return

    if not self.scan_checked:
      # the observation bins assume the UST-10LX layout: 1081 beams from -135 deg in 0.25 deg steps
      if (abs(scan_msg.angle_min + 0.75 * np.pi) > 0.01 or abs(scan_msg.angle_increment - np.radians(0.25)) > 1e-4
              or len(scan_msg.ranges) < 1080):
        self.get_logger().error(f'[scan] unexpected layout: angle_min {scan_msg.angle_min:.4f}, increment '
                                f'{scan_msg.angle_increment:.6f}, {len(scan_msg.ranges)} beams (expected -2.3562, 0.004363, 1081)')
      self.scan_checked = True
    teleop_active = self.teleop_time is not None and time.monotonic() - self.teleop_time < TELEOP_HOLD
    if teleop_active:
      self.resetHistories()

    # make obs
    raw_scan = np.array(scan_msg.ranges[:1080])
    preview = None
    if self.carpv is not None:
      preview = self.trackPreview(scan_msg)
      if preview is None:
        self.resetHistories()
        return                    # no start pose yet: don't drive
    # localization lost too long and a LiDAR-only fallback is loaded: it drives (the localization keeps searching)
    lost_for = (time.monotonic() - self.lost_since) if (self.carpv is not None and self.lost_since is not None) else 0.0
    fb_now = self.fb is not None and lost_for > self.lost_switch
    if fb_now != self.fb_on:
      self.fb_on = fb_now
      if fb_now:
        self.get_logger().warn(f'[fallback] localization lost {lost_for:.1f} s: {self.fb.name} drives (LiDAR only, '
                               f'cap {self.fb_cap:.1f} m/s); still searching the pose')
      else:
        self.get_logger().info(f'[fallback] localization valid again: {self.prim.name} drives')
    pol = self.fb if fb_now else self.prim
    cap = self.speed_limit
    if fb_now:
      cap = min(cap, self.fb_cap) if cap > 0 else self.fb_cap
      speed = self.odom_speed if self.odom_speed is not None else 0.0
      obs = make_observation(raw_scan, speed, pol.obs_cfg, pol.history, cap if cap > 0 else None, None)
    else:
      obs = self.getObs(raw_scan, self.odom_speed, preview)

    # model inference
    with torch.no_grad():
      state_tensor = torch.tensor(pol.rms.normalize(obs), dtype=torch.float32, device=self.device)
      action = pol.actor(state_tensor).loc.cpu().numpy()
    
    # transform action
    # policy angle [rad] -> servo command: calibration says command = angle * steer_cmd_gain
    cmd_gain = pol.cmd_gain[0] if action[0] > 0 else pol.cmd_gain[1]
    action[0] *= pol.args.max_steer * cmd_gain
    # per-side extra steering (steer_gain_left/right), then the servo limits
    # (the left end is short of the policy's range with offset 0.558)
    side_gain = self.gain_left if action[0] > 0 else self.gain_right
    action[0] = np.clip(action[0] * side_gain, self.steer_min, self.steer_max)
    action[1] = unnormalize_speed(action[1], pol.args.min_speed, pol.args.max_speed)
    if cap > 0:
      # cap after unnormalizing, so the model's output mapping stays as trained
      action[1] = min(action[1], cap)
    # plan B without a fallback: localization lost too long -> brake (speed 0), keep the model's steering
    if self.fb is None and self.carpv is not None and self.lost_stop > 0 and lost_for > self.lost_stop:
      action[1] = 0.0
    # starting from rest: the VESC can't start the motor with a small command (same as utils/env.py)
    action[1] = apply_start_kick(action[1], self.odom_speed, self.kick_cfg)
    
    # publish drive (low-speed mode: the car gets the stretched speed, the history keeps the policy's)
    msg = AckermannDriveStamped()
    msg.header.stamp = scan_msg.header.stamp
    msg.drive.steering_angle = float(action[0])
    msg.drive.speed = float(self.policy_to_car_speed(action[1]))
    self.drive_pub.publish(msg)

    # remember what was actually sent (after the steering clip and speed cap), normalized like the policy output
    if teleop_active:
      return                    # not executed: the history stays empty until the keyboard lets go
    self.pushHistories(action[0] / side_gain, action[1])

  def resetHistories(self):
    self.history.reset()
    if getattr(self, 'fb', None) is not None:
      self.fb.history.reset()

  def pushHistories(self, angle_cmd, speed):
    '''the command actually sent (steering before the per-side gain), normalized for each model as its own output:
    the model that is not driving then starts from the true recent commands when it takes over'''
    for pol in ([self.prim, self.fb] if getattr(self, 'fb', None) is not None else [self.prim]):
      g = pol.cmd_gain[0] if angle_cmd > 0 else pol.cmd_gain[1]
      pol.history.push([angle_cmd / (pol.args.max_steer * g),
                        normalize_speed(speed, pol.args.min_speed, pol.args.max_speed)])

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

  def onInitialPose(self, m):
    '''a pose given by hand (map frame): forget the current pose and search the start again around it'''
    q = m.pose.pose.orientation
    self.pose_hint = (float(m.pose.pose.position.x), float(m.pose.pose.position.y), float(2 * np.arctan2(q.z, q.w)))
    self.carpv.started = False
    self.start_try = -1e9
    self.get_logger().info(f'[preview] initial pose from {m.header.frame_id or "?"}: x {self.pose_hint[0]:.2f} '
                           f'y {self.pose_hint[1]:.2f} yaw {np.degrees(self.pose_hint[2]):.0f} deg -> searching near it')

  def trackPreview(self, scan_msg):
    '''localization + preview features (utils/localizer.CarPreview); None until the start pose is found'''
    if self.odom_pose is None:
      self.get_logger().warn('[preview] no odom pose yet', throttle_duration_sec=1.0)
      return None
    ranges = np.asarray(scan_msg.ranges, np.float64)
    angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
    if not self.carpv.started:
      if time.monotonic() - self.start_try < 0.5:
        return None
      t0 = time.monotonic()
      ok, msg = self.carpv.start(ranges, angles, self.odom_pose, hint=self.pose_hint)
      self.start_try = time.monotonic()
      if ok and self.pose_hint is not None:
        msg += ' (near the given pose)'
        self.pose_hint = None
      # one call site per severity: rclpy (Foxy) raises if the same line logs at another severity (11/25 review)
      if ok:
        self.get_logger().info(f'[preview] {msg} ({time.monotonic() - t0:.2f} s)')
      else:
        self.get_logger().error(f'[preview] {msg} ({time.monotonic() - t0:.2f} s)', throttle_duration_sec=2.0)
      return None             # this scan is old now (the search took a while): drive from the next one
    t0 = time.monotonic()
    cap = self.speed_limit if self.speed_limit > 0 else None
    feat, pose, valid, info = self.carpv.features(ranges, angles, self.odom_pose, cap)
    dt = time.monotonic() - t0
    if info.get('switched'):
      self.get_logger().warn(f"[preview] {info['switched']}")
    if pose is not None:
      m = PoseStamped()
      m.header.stamp = scan_msg.header.stamp
      m.header.frame_id = 'map' if valid else 'map_lost'
      m.pose.position.x, m.pose.position.y = float(pose[0]), float(pose[1])
      m.pose.position.z = float(info.get('res', 0.0)) if np.isfinite(info.get('res', np.inf)) else -1.0   # ICP residual [m]
      m.pose.orientation.z, m.pose.orientation.w = float(np.sin(pose[2] / 2)), float(np.cos(pose[2] / 2))
      self.loc_pub.publish(m)
    if valid:
      self.lost_since = None
    elif self.lost_since is None:
      self.lost_since = time.monotonic()
    if not valid:
      if getattr(self, 'fb', None) is not None:
        msg = f': LiDAR-only fallback after {self.lost_switch:.1f} s, searching the pose'
      elif self.lost_stop > 0:
        msg = f': stopping after {self.lost_stop:.1f} s'
      else:
        msg = ': driving on LiDAR only (preview valid = 0)'
      self.get_logger().warn('[preview] localization lost' + msg, throttle_duration_sec=1.0)
    if time.monotonic() - self.loc_last_report > 5.0:
      self.loc_last_report = time.monotonic()
      self.get_logger().info(f'[preview] valid {self.carpv.n_valid / max(self.carpv.n, 1) * 100:.1f}% of {self.carpv.n} scans, '
                             f'e_y {feat[0]:+.2f} m, step {dt * 1e3:.1f} ms')
    return feat

  def getObs(self, raw_scan: np.ndarray, speed=None, preview=None):
    '''
      Same processing as getObs in src/f1tenth/utils/env.py (without sim noise).
      speed: /odom linear.x [m/s], used only if the model has a speed input.
    '''
    # models trained with a speed-limit input are told the cap (agent.yaml speed_limit; <= 0 = none)
    speed = self.car_to_policy_speed(speed) if speed is not None else 0.0
    observation = make_observation(raw_scan, speed, self.obs_cfg, self.history,
                                   self.speed_limit if self.speed_limit > 0 else None, preview)
    return observation

def main(args=None):
    rclpy.init(args=args)
    p = AgentNode()
    rclpy.spin(p)

if __name__ == '__main__':
    main()
