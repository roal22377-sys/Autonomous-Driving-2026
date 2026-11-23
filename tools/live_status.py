#!/usr/bin/env python3
'''
Live driving status on the car, one line every 0.25 s (instead of the raw 40 Hz topic echo):

  time  MODE    cmd_v  odom_v  steer  | front  left  right | agent_hz
  MODE: MANUAL (keyboard /teleop is driving, it has priority in the mux), AUTO (agent /drive),
        IDLE (nothing within the mux timeout)
  front / left / right: closest lidar return within +-30 deg / 30-100 deg left / right [m]
  a '!' after front marks < 0.5 m

usage (on the car, ROS env sourced):  python3 tools/live_status.py
'''
import time

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

MUX_TIMEOUT = 0.2   # src/f1tenth_ros/f1tenth_stack/config/mux.yaml


class LiveStatus(Node):
    def __init__(self):
        super().__init__('live_status')
        self.t0 = time.monotonic()
        self.last = {'drive': None, 'teleop': None}
        self.cmd = None                  # last /ackermann_cmd (what the VESC gets)
        self.odom_v = None
        self.dist = (np.nan, np.nan, np.nan)
        self.n_drive = 0
        self.create_subscription(AckermannDriveStamped, 'drive', lambda m: self.on_src('drive'), 10)
        self.create_subscription(AckermannDriveStamped, 'teleop', lambda m: self.on_src('teleop'), 10)
        self.create_subscription(AckermannDriveStamped, 'ackermann_cmd', self.on_cmd, 10)
        self.create_subscription(Odometry, 'odom', self.on_odom, 10)
        self.create_subscription(LaserScan, 'scan', self.on_scan, 10)
        self.create_timer(0.25, self.show)
        self.last_show = time.monotonic()
        print('  time  MODE    cmd_v  odom_v  steer  | front  left  right | agent_hz', flush=True)

    def on_src(self, name):
        self.last[name] = time.monotonic()
        if name == 'drive':
            self.n_drive += 1

    def on_cmd(self, m):
        self.cmd = (m.drive.speed, m.drive.steering_angle)

    def on_odom(self, m):
        self.odom_v = m.twist.twist.linear.x

    def on_scan(self, m):
        r = np.asarray(m.ranges, dtype=np.float32)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment
        r = np.where(np.isfinite(r) & (r > 0.05), r, np.inf)
        deg = np.degrees(a)

        def mn(mask):
            v = r[mask]
            return float(v.min()) if v.size and np.isfinite(v.min()) else np.nan
        self.dist = (mn(np.abs(deg) < 30), mn((deg >= 30) & (deg <= 100)), mn((deg <= -30) & (deg >= -100)))

    def show(self):
        now = time.monotonic()
        fresh = {k: v is not None and now - v < MUX_TIMEOUT for k, v in self.last.items()}
        mode = 'MANUAL' if fresh['teleop'] else 'AUTO' if fresh['drive'] else 'IDLE'
        hz = self.n_drive / max(now - self.last_show, 1e-3)
        self.n_drive, self.last_show = 0, now
        v, s = self.cmd if self.cmd else (np.nan, np.nan)
        ov = self.odom_v if self.odom_v is not None else np.nan
        f, l, r = self.dist
        warn = '!' if f < 0.5 else ' '
        print(f'{now - self.t0:6.1f}  {mode:6s}  {v:5.2f}  {ov:5.2f}  {s:+.3f} | {f:5.2f}{warn}{l:5.2f} {r:5.2f} | {hz:4.0f}',
              flush=True)


def main():
    rclpy.init()
    n = LiveStatus()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
