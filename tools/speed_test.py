#!/usr/bin/env python3
'''
Straight-line speed test toward a flat wall: is /odom (speed_to_erpm_gain) the true speed?

The car drives straight (steering 0) at one constant speed toward a wall. The lidar range straight
ahead shrinks at exactly the true speed, so no model or scan matching is involved.
tools/speed_analysis.py compares that with /odom.

Sequence: 3 s countdown -> --speed (kick to 2 m/s first if lower) -> brake to 0 when the wall is
closer than the stopping distance (v^2 / (2 * --brake) + --margin) -> done.
Publishes on /drive (mux "navigation"); the keyboard (priority 100) still overrides: s = stop.

usage (on the car, bringup running with use_agent:=false; tools/speed_series.sh runs several speeds):
    python3 tools/speed_test.py --speed 3.0 --dry_run
    python3 tools/speed_test.py --speed 3.0
'''
import argparse
import math
import time

import numpy as np

KICK = 2.0          # [m/s] start-up command below this speed (VESC cannot start with less)


def stop_dist(v, brake, margin):
    return v * v / (2 * brake) + margin


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speed', type=float, required=True, help='[m/s] test speed')
    p.add_argument('--accel', type=float, default=2.5, help='[m/s^2] assumed acceleration (for the plan)')
    p.add_argument('--brake', type=float, default=2.5, help='[m/s^2] assumed braking (for the stop distance)')
    p.add_argument('--margin', type=float, default=1.0, help='[m] extra distance to the wall when stopped')
    p.add_argument('--min_hold', type=float, default=1.5, help='[s] constant-speed time needed (plan check)')
    p.add_argument('--rate', type=float, default=50.0)
    p.add_argument('--no_prompt', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()

    v = a.speed
    d_acc = v * v / (2 * a.accel)
    d_hold = v * a.min_hold
    d_stop = stop_dist(v, a.brake, a.margin)
    need = d_acc + d_hold + d_stop
    print(f'[plan] {v:.1f} m/s straight: accelerate ~{d_acc:.1f} m, hold >= {a.min_hold:.1f} s ({d_hold:.1f} m), '
          f'brake when the wall is {d_stop:.1f} m ahead -> start at least {need:.1f} m from a flat wall')
    if a.dry_run:
        return

    import select
    import sys
    import rclpy
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    node = Node('speed_test')
    pub = node.create_publisher(AckermannDriveStamped, 'drive', 10)
    st = {'odom': None, 'front': None}

    def on_odom(m):
        st['odom'] = m.twist.twist.linear.x

    def on_scan(m):
        r = np.asarray(m.ranges, dtype=np.float32)
        ang = m.angle_min + np.arange(len(r)) * m.angle_increment
        f = r[(np.abs(ang) < math.radians(10)) & np.isfinite(r) & (r > 0.05)]
        st['front'] = float(f.min()) if len(f) else float('inf')

    node.create_subscription(Odometry, 'odom', on_odom, 10)
    node.create_subscription(LaserScan, 'scan', on_scan, 10)

    def send(speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.drive.steering_angle = 0.0
        msg.drive.speed = float(speed)
        pub.publish(msg)

    def hold_zero(sec):
        t_end = time.monotonic() + sec
        while time.monotonic() < t_end:
            send(0.0)
            time.sleep(1.0 / a.rate)

    try:
        t0 = time.monotonic()
        while st['odom'] is None or st['front'] is None:
            send(0.0)
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - t0 > 10:
                print('[error] no /odom or /scan for 10 s. Is bringup running?')
                return
        print(f'[ready] wall straight ahead at {st["front"]:.1f} m (need {need:.1f} m). TELEOP tab: r (released).')
        if st['front'] < need:
            print(f'[warn] closer than the plan needs: the constant-speed part will be short')
        if not a.no_prompt:
            print('Enter to start (Ctrl+C = stop) ', end='', flush=True)
            while not select.select([sys.stdin], [], [], 1.0 / a.rate)[0]:
                send(0.0)
                rclpy.spin_once(node, timeout_sec=0.0)
            sys.stdin.readline()
        print('[start] driving in 3 s')
        hold_zero(3.0)
        print(f'[drive] {v:.1f} m/s straight', flush=True)
        t_start = time.monotonic()
        while True:
            rclpy.spin_once(node, timeout_sec=0.0)
            cur = st['odom'] or 0.0
            if st['front'] < stop_dist(max(cur, 0.5), a.brake, a.margin):
                print(f'[brake] wall at {st["front"]:.2f} m (odom {cur:.2f} m/s, '
                      f'{time.monotonic() - t_start:.1f} s after start)')
                break
            if time.monotonic() - t_start > 30:
                print('[stop] 30 s limit')
                break
            cmd = KICK if (cur < 0.3 and v < KICK) else v
            send(cmd)
            time.sleep(1.0 / a.rate)
        hold_zero(1.5)
        print('[done] stopped.')
    except KeyboardInterrupt:
        print('\n[Ctrl+C] stopping')
        hold_zero(1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
