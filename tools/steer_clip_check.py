#!/usr/bin/env python3
'''
Checks on the car that agent_node's steering clip keeps the servo inside its range
(no "servo command value ... clipping" from vesc_driver any more).

Runs the real AgentNode (agent.yaml model, real /scan and /odom) but forces the policy output
to center / full left / full right / half left, and zeroes the speed before publishing
(the motor does not turn). For each phase it compares:
  - steering agent_node sent (/drive)
  - servo value vesc_driver actually sent (/sensors/servo_position_command, after its clip)
  - servo value the command maps to (gain*cmd + offset): if that is outside [0, 1] the VESC clipped it
and checks the action history holds the value that was sent.

usage (bringup running with use_agent:=false; tools/steer_clip_check.sh does it all):
    python3 tools/steer_clip_check.py
'''
import time

import numpy as np
import rclpy
import torch
from std_msgs.msg import Float64

from f1tenth_stack.agent_node import AgentNode, steering_command_limits

PHASES = [(0.0, '중앙'), (1.0, '왼쪽 끝'), (0.0, '중앙'), (-1.0, '오른쪽 끝'), (0.0, '중앙'), (0.5, '왼쪽 절반')]
PHASE_S = 1.5
VESC_CFG = 'src/f1tenth_ros/f1tenth_stack/config/vesc.yaml'


class ForcedPolicy:
    '''stands in for the actor: returns a fixed normalized action'''
    def __init__(self):
        self.steer = 0.0

    def __call__(self, x):
        class Dist:
            pass
        d = Dist()
        d.loc = torch.tensor([self.steer, 0.0])
        return d


def main():
    rclpy.init(args=['--ros-args', '--params-file', 'src/f1tenth_ros/f1tenth_stack/config/agent.yaml'])
    node = AgentNode()
    policy = ForcedPolicy()
    node.actor = policy

    # zero the speed of every command (steering untouched) so the motor never turns
    real_publish = node.drive_pub.publish
    sent = []

    def publish_no_speed(msg):
        msg.drive.speed = 0.0
        sent.append(msg.drive.steering_angle)
        real_publish(msg)
    node.drive_pub.publish = publish_no_speed

    servo = []
    node.create_subscription(Float64, 'sensors/servo_position_command', lambda m: servo.append(m.data), 10)

    from ruamel.yaml import YAML
    p = next(iter(YAML().load(open(VESC_CFG)).values()))['ros__parameters']
    gain, offset = float(p['steering_angle_to_servo_gain']), float(p['steering_angle_to_servo_offset'])
    scale = node.args.max_steer * float(node.args.get('steer_cmd_gain', 1.0))
    lo, hi = steering_command_limits(VESC_CFG)
    print(f'\n[설정] 직진 offset {offset}, 조향 한계: 오른쪽 {lo:+.3f} / 왼쪽 {hi:+.3f}, '
          f'정책 최대 조향 {scale:.3f} (모델 {node.get_parameter("name").value}/{node.get_parameter("algo_idx").value})')

    # vesc_to_odom publishes /odom only after a servo command: send "stopped" until it does
    from ackermann_msgs.msg import AckermannDriveStamped
    t0 = time.monotonic()
    while node.odom_time is None:
        m = AckermannDriveStamped()
        real_publish(m)
        rclpy.spin_once(node, timeout_sec=0.05)
        if time.monotonic() - t0 > 10:
            print('[오류] /odom이 안 들어와요. bringup이 실행 중인가요?')
            return

    rows = []
    for steer, label in PHASES:
        policy.steer = steer
        sent.clear(), servo.clear()
        t_end = time.monotonic() + PHASE_S
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not sent:
            print(f'[오류] {label}: agent_node가 명령을 안 냈어요 (/scan, /odom 확인)')
            return
        cmd = float(np.median(sent[len(sent) // 2:]))
        sv = float(np.median(servo[len(servo) // 2:])) if servo else float('nan')
        want = gain * cmd + offset
        hist = float(node.history.buf[0][0]) if node.history.length else float('nan')
        ok_servo = 0.0 <= want <= 1.0 and abs(sv - want) < 0.002
        ok_hist = (not node.history.length) or abs(hist - cmd / scale) < 0.002
        rows.append(ok_servo and ok_hist)
        print(f'  {label:6s} 정책 {steer:+.1f} → 원래 명령 {steer * scale:+.3f}, 보낸 명령 {cmd:+.3f} | '
              f'서보 필요값 {want:.3f}, VESC가 받은 값 {sv:.3f} | '
              + (f'이력 {hist:+.3f} | ' if node.history.length else '')
              + ('통과' if ok_servo and ok_hist else '실패: ' + ('서보 범위 밖/잘림 ' if not ok_servo else '') + ('이력 불일치' if not ok_hist else '')))

    # back to center, stopped
    policy.steer = 0.0
    for _ in range(20):
        real_publish(AckermannDriveStamped())
        time.sleep(0.05)
    print('\n[결과] ' + ('모든 단계 통과: 조향 명령이 서보 범위 안에 있고, 이력도 보낸 값과 같아요.'
                       if all(rows) else '실패한 단계가 있어요. 위 표를 확인하세요.'))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
