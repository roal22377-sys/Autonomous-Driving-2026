#!/usr/bin/env python3
'''
Constant-steering circle test: one steering command, a ladder of speeds. No policy, no keyboard.
Measures (with tools/circle_analysis.py on the bag):
  - steering: real wheel angle for this command (circle radius at low lateral acceleration)
  - grip:     the lateral acceleration where the circle stops shrinking with steering = tire limit
  - speed:    true speed (lidar) vs /odom (speed_to_erpm_gain)
  - low speed: whether the VESC holds speeds below the 2 m/s policy minimum once rolling

Sequence (each step held --hold seconds, or --low_hold for the low-speed steps):
    kick at --kick m/s  ->  speed ladder (--speeds, ascending)  ->  low ladder (--low, descending)  ->  stop
(low steps last: if the VESC cannot hold one, the stall stop ends the run after the grip data is in)
The kick is needed because the VESC cannot start the motor from rest with small commands
(11/23: 0.5-1.5 m/s from standstill -> 30 A spikes, fault 4).

Publishes on /drive (mux "navigation"), so the keyboard (priority 100) still overrides it: keep the
teleop tab ready, s = stop.

Safety:
  - waits for /odom and /scan, then a countdown
  - stops if anything is closer than --stop_dist in front (+-25 deg)
  - stops if the measured speed stays < 0.3 m/s for 1.5 s while >= 1 m/s is commanded (stalled / stuck)
  - Ctrl+C or the end: speed 0 for 1 s
  - --dry_run prints the circle sizes the plan needs (kinematic, and with the sim's grip limit)

usage (on the car, bringup running with use_agent:=false; tools/circle_test.sh does all of it):
    python3 tools/circle_test.py --steer 0.3 --dry_run
    python3 tools/circle_test.py --steer 0.3 --speeds 2.0 2.5 3.0 3.5
    python3 tools/circle_test.py --steer -0.3 ...        # right turn
'''
import argparse
import math
import time

import numpy as np

WHEELBASE = 0.34
STEER_CMD_GAIN = 1.031      # command = wheel angle * gain (configs/task/f1tenth.yaml)
SIM_GRIP = 10.0             # [m/s^2] roughly mu * g in the current sim (dynamic.yaml mu 1.05)
REAL_GRIP_GUESS = 5.5       # [m/s^2] from the 11/23 logs (lateral accel saturated at 4-6)


def build_plan(a):
    '''list of (duration, label, speed)'''
    plan = [(a.kick_time, 'kick', a.kick)]
    plan += [(a.hold, f'speed {v:.1f}', v) for v in sorted(a.speeds)]
    plan += [(a.low_hold, f'low {v:.1f}', v) for v in sorted(a.low, reverse=True)]
    return plan


def dry_run(plan, a):
    delta = abs(a.steer) / STEER_CMD_GAIN
    R_kin = WHEELBASE / math.tan(delta) if delta > 1e-3 else float('inf')
    print(f'[plan] steering command {a.steer:+.3f} ({"left" if a.steer > 0 else "right"}), '
          f'kinematic radius {R_kin:.2f} m (diameter {2 * R_kin:.1f} m)')
    total = 0.0
    for dur, label, v in plan:
        ay_kin = v * v / R_kin
        R_grip = max(R_kin, v * v / REAL_GRIP_GUESS)
        laps = dur * v / (2 * math.pi * R_grip)
        total += dur
        print(f'   {label:10s} {v:3.1f} m/s for {dur:3.1f} s: lateral accel {ay_kin:4.1f} m/s^2 if no slip; '
              f'with ~{REAL_GRIP_GUESS:.1f} m/s^2 grip the circle grows to diameter {2 * R_grip:.1f} m '
              f'({laps:.1f} laps)')
    v_max = max(p[2] for p in plan)
    print(f'[plan] {total:.0f} s; needs a free area of about {2 * max(R_kin, v_max ** 2 / REAL_GRIP_GUESS) + 1.0:.1f} m '
          f'across (sim grip would allow {SIM_GRIP:.0f} m/s^2)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steer', type=float, required=True,
                   help='steering command [rad], + left / - right (servo limits: about +0.44 / -0.55)')
    p.add_argument('--speeds', type=float, nargs='+', default=[2.0, 2.5, 3.0, 3.5])
    p.add_argument('--low', type=float, nargs='*', default=[1.5, 1.2, 1.0, 0.8],
                   help='low-speed hold check at the end, descending (empty to skip)')
    p.add_argument('--kick', type=float, default=2.0, help='[m/s] start speed')
    p.add_argument('--kick_time', type=float, default=1.5)
    p.add_argument('--hold', type=float, default=5.0, help='[s] per speed step (>= 1 lap is best)')
    p.add_argument('--low_hold', type=float, default=3.0, help='[s] per low-speed step')
    p.add_argument('--stop_dist', type=float, default=0.6, help='[m] emergency stop distance ahead')
    p.add_argument('--rate', type=float, default=50.0, help='[Hz] publish rate')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    plan = build_plan(a)
    dry_run(plan, a)
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
    node = Node('circle_test')
    pub = node.create_publisher(AckermannDriveStamped, 'drive', 10)
    state = {'odom': None, 'front': None}

    def on_odom(m):
        state['odom'] = m.twist.twist.linear.x

    def on_scan(m):
        r = np.asarray(m.ranges, dtype=np.float32)
        ang = m.angle_min + np.arange(len(r)) * m.angle_increment
        front = r[(np.abs(ang) < math.radians(25)) & np.isfinite(r) & (r > 0.05)]
        state['front'] = float(front.min()) if len(front) else float('inf')

    node.create_subscription(Odometry, 'odom', on_odom, 10)
    node.create_subscription(LaserScan, 'scan', on_scan, 10)

    def send(steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        pub.publish(msg)

    def stop(seconds=1.0):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            send(0.0, 0.0)
            time.sleep(1.0 / a.rate)

    try:
        print('[wait] /odom and /scan ...')
        t0 = time.monotonic()
        while state['odom'] is None or state['front'] is None:
            send(0.0, 0.0)      # vesc_to_odom publishes /odom only after it has seen a servo command
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - t0 > 10:
                print('[error] no /odom or /scan for 10 s. Is bringup running?')
                return
        print(f'[ready] free space ahead {state["front"]:.1f} m. In the keyboard tab press r (release), not s.')
        print('Put the car in the middle of the free area, then press Enter to start (Ctrl+C = stop) ',
              end='', flush=True)
        while not select.select([sys.stdin], [], [], 1.0 / a.rate)[0]:
            send(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.0)
        sys.stdin.readline()
        t_cd = time.monotonic() + 3.0
        for i in (3, 2, 1):
            print(f'  {i}...')
            while time.monotonic() < t_cd - (i - 1):
                send(0.0, 0.0)
                time.sleep(1.0 / a.rate)

        aborted = False
        for k, (dur, label, speed) in enumerate(plan):
            print(f'[drive] {label}: {speed:.1f} m/s, steering {a.steer:+.3f}', flush=True)
            t_end = time.monotonic() + dur
            t_slow = None
            while time.monotonic() < t_end:
                rclpy.spin_once(node, timeout_sec=0.0)
                if state['front'] < a.stop_dist:
                    print(f'[STOP] obstacle {state["front"]:.2f} m ahead during "{label}"')
                    aborted = True
                    break
                # stalled (VESC could not hold this speed) or stuck on something
                if speed >= 1.0 and state['odom'] is not None and state['odom'] < 0.3:
                    t_slow = t_slow or time.monotonic()
                    if time.monotonic() - t_slow > 1.5:
                        print(f'[STOP] car not moving at "{label}" (odom {state["odom"]:.2f} m/s)')
                        aborted = True
                        break
                else:
                    t_slow = None
                send(a.steer, speed)
                time.sleep(1.0 / a.rate)
            if aborted:
                break
        stop()
        print('[done] stopped.' + (' (aborted)' if aborted else ''))
    except KeyboardInterrupt:
        print('\n[Ctrl+C] stopping')
        stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
