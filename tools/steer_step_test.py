#!/usr/bin/env python3
'''
Open-loop steering step test: drives a fixed steering pattern (no policy, no keyboard) so the
real command -> yaw delay can be measured with tools/steer_delay.py.

Pattern (S-curves, the car ends up heading the same way it started):
    straight run-in  ->  [ +A, -A, -A, +A ] x cycles  ->  straight  ->  stop
each entry is held for --hold seconds; every change is a steering step.

Publishes on /drive (ackermann_mux "navigation", priority 10), so the keyboard teleop
(priority 100) can still override it.

Safety:
  - waits for /odom and /scan, then a countdown
  - stops (speed 0) if anything is closer than --stop_dist in front (+-25 deg)
  - Ctrl+C or the end of the pattern: publishes speed 0 for 1 s
  - --dry_run prints the planned path size (kinematic bicycle) without driving

usage (on the car, bringup running with use_agent:=false):
    python3 tools/steer_step_test.py --dry_run
    python3 tools/steer_step_test.py --speed 1.5 --amp 0.3 --hold 0.6 --cycles 2

--stand: wheels in the air, servo delay only (command -> wheels start to turn)
    speed 0 (motor never driven), steering 0 -> +A -> 0 -> -A ..., each held --hold s (default 1.0).
    At the moment each command is sent the whole terminal changes color
    (center = black, left = green, right = red). Film the screen and a front wheel together
    with phone slow motion (240 fps) and count frames from the color change to the first wheel
    movement. The screen shows the change ~1-2 display frames (~16-33 ms) after the command,
    so the true servo delay is that much longer than the counted frames.
    python3 tools/steer_step_test.py --stand --amp 0.497 --cycles 5
'''
import argparse
import math
import time

import numpy as np

WHEELBASE = 0.34
STEER_CMD_GAIN = 1.031      # command = wheel angle * gain (configs/task/f1tenth.yaml)


def build_plan(a):
    '''list of (duration, steering command, speed)'''
    plan = [(a.run_in, 0.0, a.speed)]
    for _ in range(a.cycles):
        for s in (+1, -1, -1, +1):
            plan.append((a.hold, s * a.amp, a.speed))
    plan.append((a.run_out, 0.0, a.speed))
    return plan


def build_stand_plan(a):
    '''wheels in the air: no throttle, steps from center to each side and back'''
    plan = [(1.0, 0.0, 0.0)]
    for _ in range(a.cycles):
        for s in (+1, 0, -1, 0):
            plan.append((a.hold, s * a.amp, 0.0))
    return plan


def flash(steer, k, n):
    '''fill the terminal with one color per command so the change is visible on video'''
    bg, name = ('\033[42m', 'LEFT') if steer > 1e-6 else (('\033[41m', 'RIGHT') if steer < -1e-6 else ('\033[40m', 'CENTER'))
    print(f'{bg}\033[97m\033[2J\033[H\n\n\n        {name}   {steer:+.3f} rad      step {k}/{n}\n', end='', flush=True)


def dry_run(plan, a):
    '''kinematic path with a 0.1 s steering delay and 3 m/s^2 acceleration'''
    dt, x, y, yaw, v = 0.01, 0.0, 0.0, 0.0, 0.0
    cmds = []
    for dur, st, sp in plan:
        cmds += [(st, sp)] * int(round(dur / dt))
    delay = int(0.1 / dt)
    xs, ys = [0.0], [0.0]
    for i in range(len(cmds)):
        st, sp = cmds[max(0, i - delay)]
        v = min(sp, v + 3.0 * dt)
        yaw += v * math.tan(st / STEER_CMD_GAIN) / WHEELBASE * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        xs.append(x); ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    total = sum(p[0] for p in plan)
    lat = v ** 2 * math.tan(a.amp / STEER_CMD_GAIN) / WHEELBASE
    print(f'[plan] {len(plan) - 2} steering holds of {a.hold:.2f} s at +-{a.amp:.2f} rad, {a.speed:.2f} m/s, '
          f'{total:.1f} s total')
    print(f'[plan] needs about {xs.max() - xs.min():.1f} m forward x {ys.max() - ys.min():.1f} m sideways '
          f'(+ braking ~{a.speed ** 2 / 6:.1f} m), final heading {math.degrees(yaw):+.0f} deg')
    print(f'[plan] lateral accel in the turns {lat:.1f} m/s^2 (keep below ~4 so the tires do not slide)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speed', type=float, default=1.5, help='[m/s]')
    p.add_argument('--amp', type=float, default=0.3, help='[rad] steering command amplitude (max 0.497)')
    p.add_argument('--hold', type=float, default=None,
                   help='[s] each steering hold (default 0.6 driving, >= 0.6 for the analysis; 1.0 with --stand)')
    p.add_argument('--cycles', type=int, default=None, help='pattern repeats (default 2 driving, 5 with --stand)')
    p.add_argument('--stand', action='store_true', help='wheels in the air: servo delay only, see top of file')
    p.add_argument('--run_in', type=float, default=1.5, help='[s] straight before the first step')
    p.add_argument('--run_out', type=float, default=0.5, help='[s] straight after the last step')
    p.add_argument('--stop_dist', type=float, default=0.8, help='[m] emergency stop distance ahead')
    p.add_argument('--rate', type=float, default=50.0, help='[Hz] publish rate')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    a.amp = min(abs(a.amp), 0.497)
    if a.hold is None:
        a.hold = 1.0 if a.stand else 0.6
    if a.cycles is None:
        a.cycles = 5 if a.stand else 2
    if a.stand:
        plan = build_stand_plan(a)
        print(f'[stand] wheels in the air, motor off: {len(plan) - 1} steering steps of +-{a.amp:.3f} rad, '
              f'{a.hold:.1f} s apart. Terminal color changes when each command is sent.')
    else:
        plan = build_plan(a)
        dry_run(plan, a)
    if a.dry_run:
        return

    import rclpy
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    node = Node('steer_step_test')
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
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - t0 > 10:
                print('[error] no /odom or /scan for 10 s. Is bringup running?')
                return
        if a.stand:
            input('Wheels off the ground? Start the slow-motion video (screen + front wheel in view), '
                  'then press Enter (Ctrl+C = stop) ')
        else:
            print(f'[ready] free space ahead {state["front"]:.1f} m. In the keyboard tab press r (release), not s.')
            input('Put the car at the start, then press Enter to start (Ctrl+C = stop) ')
        for i in (3, 2, 1):
            print(f'  {i}...')
            time.sleep(1.0)

        print('[drive] running the pattern')
        aborted = False
        step_log = []
        for k, (dur, steer, speed) in enumerate(plan):
            t_end = time.monotonic() + dur
            first = True
            while time.monotonic() < t_end:
                rclpy.spin_once(node, timeout_sec=0.0)
                if a.stand:
                    send(steer, 0.0)
                    if first:
                        flash(steer, k, len(plan) - 1)
                        step_log.append((time.time(), steer))
                        first = False
                    time.sleep(1.0 / a.rate)
                    continue
                if state['front'] < a.stop_dist:
                    print(f'[STOP] obstacle {state["front"]:.2f} m ahead during step {k}')
                    aborted = True
                    break
                send(steer, speed)
                time.sleep(1.0 / a.rate)
            if aborted:
                break
        stop()
        if a.stand:
            print('\033[0m\033[2J\033[H', end='')
            print(f'[done] {len(step_log) - 1} steps sent. In the video, for each color change count the frames '
                  'until the wheel starts to move: delay = frames / fps (+ ~16-33 ms display lag).')
        else:
            print('[done] stopped.' + (' (aborted: rerun with more space or fewer --cycles)' if aborted else ''))
    except KeyboardInterrupt:
        print('\033[0m\n[Ctrl+C] stopping')
        stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
