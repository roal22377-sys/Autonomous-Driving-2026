#!/usr/bin/env python3
'''
Straight-line alignment (steering trim) on the car: find the steering command that drives straight,
then put it into vesc.yaml as steering_angle_to_servo_offset.

Each run: 3 s countdown -> drive at --speed with steering = trim for --duration seconds -> stop.
It stops on time (the lidar is not needed); it also stops if the front range (+-5 deg) is below
--front_stop (0 = off). Publishes on /drive (mux "navigation"); the keyboard still overrides: s = stop.

After a run, enter how far the car ended up to the side of a straight floor line (tape), in cm
(+ = left of the line, - = right), and the distance along the line in m (Enter = odom distance).
From that the script suggests the next trim:
    drift y over distance L  ->  curvature 2y/L^2  ->  steering 2y*WB/L^2 [rad]  ->  trim -= that
When the drift is a few cm over 4-5 m in a few runs in a row, quit: it prints the new
steering_angle_to_servo_offset (offset + trim * servo gain) for config/vesc.yaml.

usage (on the car, bringup running with use_agent:=false; tools/drive_session.sh --straight opens it):
    python3 tools/straight_test.py                 # 1.5 m/s for 2.5 s (about 4-5 m)
    python3 tools/straight_test.py --trim -0.01    # start from a trim [rad] (+ = left)
'''
import argparse
import math
import os
import re
import sys
import time

import numpy as np

KICK = 2.0          # [m/s] start-up command below this speed (VESC cannot start with less)
WB = 0.34           # [m] wheelbase
VESC_YAML = os.path.expanduser('~/ros2_ws/src/f1tenth_ros/f1tenth_stack/config/vesc.yaml')


def servo_params():
    try:
        txt = open(VESC_YAML).read()
        off = float(re.search(r'steering_angle_to_servo_offset:\s*([-\d.]+)', txt).group(1))
        gain = float(re.search(r'steering_angle_to_servo_gain:\s*([-\d.]+)', txt).group(1))
        return off, gain
    except (OSError, AttributeError, ValueError):
        return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speed', type=float, default=1.5, help='[m/s]')
    p.add_argument('--duration', type=float, default=2.5, help='[s] driving time per run')
    p.add_argument('--trim', type=float, default=0.0, help='[rad] starting steering trim (+ = left)')
    p.add_argument('--front_stop', type=float, default=0.5, help='[m] stop if the front range is below this (0 = off)')
    p.add_argument('--rate', type=float, default=50.0)
    a = p.parse_args()

    import rclpy
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    node = Node('straight_test')
    pub = node.create_publisher(AckermannDriveStamped, 'drive', 10)
    st = {'odom': None, 'front': None}

    def on_odom(m):
        st['odom'] = m.twist.twist.linear.x

    def on_scan(m):
        r = np.asarray(m.ranges, dtype=np.float32)
        ang = m.angle_min + np.arange(len(r)) * m.angle_increment
        f = r[(np.abs(ang) < math.radians(5)) & np.isfinite(r) & (r > 0.05)]
        st['front'] = float(np.median(f)) if len(f) else float('inf')

    node.create_subscription(Odometry, 'odom', on_odom, 10)
    node.create_subscription(LaserScan, 'scan', on_scan, 10)

    def send(speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        pub.publish(msg)

    def hold(sec, steer):
        t_end = time.monotonic() + sec
        while time.monotonic() < t_end:
            send(0.0, steer)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(1.0 / a.rate)

    def drive(trim):
        print('[start] driving in 3 s', flush=True)
        hold(3.0, trim)
        print(f'[drive] {a.speed:.1f} m/s, steering {trim:+.4f} rad for {a.duration:.1f} s', flush=True)
        t0 = time.monotonic()
        last = t0
        dist = 0.0
        why = 'time'
        while time.monotonic() - t0 < a.duration:
            rclpy.spin_once(node, timeout_sec=0.0)
            cur = st['odom'] or 0.0
            now = time.monotonic()
            dist += max(cur, 0.0) * (now - last)
            last = now
            if a.front_stop > 0 and st['front'] is not None and st['front'] < a.front_stop:
                why = f'front {st["front"]:.2f} m'
                break
            send(KICK if (cur < 0.3 and a.speed < KICK) else a.speed, trim)
            time.sleep(1.0 / a.rate)
        # coast-down distance while stopping
        t1 = time.monotonic()
        while time.monotonic() - t1 < 1.5:
            send(0.0, trim)
            rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()
            dist += max(st['odom'] or 0.0, 0.0) * (now - last)
            last = now
            time.sleep(1.0 / a.rate)
        print(f'[done] stopped ({why}). odom distance {dist:.2f} m', flush=True)
        return dist

    def ask(prompt):
        # keep publishing zero while waiting for the keyboard, so the mux stays on a stopped command
        import select
        print(prompt, end='', flush=True)
        while not select.select([sys.stdin], [], [], 1.0 / a.rate)[0]:
            send(0.0, trim)
            rclpy.spin_once(node, timeout_sec=0.0)
        return sys.stdin.readline().strip()

    trim = a.trim
    history = []
    try:
        t0 = time.monotonic()
        while st['odom'] is None:
            send(0.0, trim)
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - t0 > 10:
                print('[error] no /odom for 10 s. Is bringup running?')
                return
        print(f'[ready] about {a.speed * a.duration * 1.3 + 1:.0f} m of straight floor needed. '
              'Lay a straight tape line, put the car on it, nose along the line. TELEOP tab: r (released).')
        while True:
            print(f'\n==== trim {trim:+.4f} rad ====')
            ans = ask('Enter = drive | number = set trim [rad] | q = quit: ')
            if ans == 'q':
                break
            if ans:
                try:
                    trim = float(ans)
                except ValueError:
                    print('  not a number')
                continue
            dist = drive(trim)
            y = ask('  drift from the line [cm, + = left, - = right] (Enter = skip): ')
            if not y:
                continue
            try:
                y = float(y) / 100.0
            except ValueError:
                print('  not a number, skipped')
                continue
            L = ask(f'  distance along the line [m] (Enter = odom {dist:.2f}): ')
            try:
                L = float(L) if L else dist
            except ValueError:
                L = dist
            if L < 1.0:
                print('  too short to judge (< 1 m), skipped')
                continue
            delta = 2.0 * y * WB / (L * L)
            history.append((trim, y, L, delta))
            new = trim - delta
            print(f'  -> the car steers {delta:+.4f} rad {"left" if delta > 0 else "right"} at this trim '
                  f'(radius {abs(WB / delta) if abs(delta) > 1e-6 else float("inf"):.0f} m). next trim {new:+.4f}')
            trim = new
    except KeyboardInterrupt:
        print('\n[Ctrl+C] stopping')
    finally:
        hold(1.0, trim)
        node.destroy_node()
        rclpy.shutdown()

    if history:
        print('\nruns: trim [rad], drift [cm], distance [m], steering error [rad]')
        for h in history:
            print(f'  {h[0]:+.4f}  {h[1] * 100:+6.1f}  {h[2]:5.2f}  {h[3]:+.4f}')
    off, gain = servo_params()
    print(f'\nfinal trim {trim:+.4f} rad')
    if off is not None:
        print(f'-> config/vesc.yaml steering_angle_to_servo_offset: {off:.3f} -> {off + trim * gain:.3f} '
              f'(gain {gain}); then trim 0 drives straight')


if __name__ == '__main__':
    main()
