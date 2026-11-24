#!/usr/bin/env python3
'''
Steering servo range check, wheels in the air. Sends raw servo values (0..1, VESC hard limit)
straight to /commands/servo/position and asks what the wheel does at each one.
Answers are saved to ~/bags/servo_range_<time>.txt for the analysis.

Question: at servo 1.0 (left end, the current left command is clipped there) is the wheel at its
mechanical stop, or is there room left? Same for 0.0 on the right.

Needs bringup running (vesc_driver), and nothing else commanding the car (no agent, no keyboard).
    python3 tools/servo_range_check.py
'''
import os
import select
import sys
import time

CENTER = 0.567      # steering_angle_to_servo_offset in vesc.yaml (11/24; was 0.558)
# (servo value, label, questions); questions: 'more' = moved further than the previous value,
# 'gap' = gap left to the mechanical stop, 'buzz' = servo straining / buzzing, 'cm' = stick tip offset
SEQUENCE = [
    (CENTER, 'center', []),
    (0.80, 'left 0.80', ['cm']),
    (0.90, 'left 0.90', ['more', 'cm']),
    (0.95, 'left 0.95', ['more', 'cm']),
    (1.00, 'left 1.00 (END)', ['more', 'gap', 'buzz', 'cm']),
    (CENTER, 'center', []),
    (0.30, 'right 0.30', ['cm']),
    (0.15, 'right 0.15', ['more', 'cm']),
    (0.061, 'right 0.061 (current right max)', ['more', 'gap', 'buzz', 'cm']),
    (0.03, 'right 0.03', ['more', 'cm']),
    (0.00, 'right 0.00 (END)', ['more', 'gap', 'buzz', 'cm']),
    (CENTER, 'center', []),
]
# --quick: only the two open questions (left end gap, right end strain)
QUICK_SEQUENCE = [
    (CENTER, 'center', []),
    (1.00, 'left 1.00 (END)', ['gap', 'buzz']),
    (CENTER, 'center', []),
    (0.061, 'right 0.061 (current right max)', ['gap', 'buzz']),
    (0.00, 'right 0.00 (END)', ['buzz']),
    (CENTER, 'center', []),
]
QUESTIONS = {
    'more': '이전 값보다 바퀴가 더 꺾였나요? (y/n)',
    'gap':  '바퀴(너클)와 스토퍼 사이에 틈이 남아 있나요? (y/n)',
    'buzz': '서보가 "지잉" 소리 내며 힘주고 있나요? (y/n)',
    'cm':   '막대 끝이 중앙 표시에서 몇 cm 이동? (숫자, 모르면 Enter)',
}
END_HOLD = 8.0      # [s] at an end value, go back to center after this (servo strain)


def main():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64

    rclpy.init()
    node = Node('servo_range_check')
    pub = node.create_publisher(Float64, 'commands/servo/position', 10)
    seen = {'n': 0}
    node.create_subscription(Float64, 'sensors/servo_position_command', lambda m: seen.__setitem__('n', seen['n'] + 1), 10)

    def send(v):
        pub.publish(Float64(data=float(v)))

    def ask(prompt, hold_value, end):
        '''keep publishing hold_value while waiting for a line; at END values return to center after END_HOLD'''
        print(prompt + ' ', end='', flush=True)
        t0, value, warned = time.monotonic(), hold_value, False
        while not select.select([sys.stdin], [], [], 0.05)[0]:
            if end and not warned and time.monotonic() - t0 > END_HOLD:
                value, warned = CENTER, True
                print(f'\n  ({END_HOLD:.0f}s 지나서 서보 보호를 위해 중앙으로 복귀했어요. 본 대로 답해 주세요) ', end='', flush=True)
            send(value)
            rclpy.spin_once(node, timeout_sec=0.0)
        return sys.stdin.readline().strip(), value

    path = os.path.expanduser(f'~/bags/servo_range_{time.strftime("%m%d_%H%M")}.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = open(path, 'w')
    try:
        # check that vesc_driver is listening
        t0 = time.monotonic()
        while seen['n'] == 0 and time.monotonic() - t0 < 5:
            send(CENTER)
            rclpy.spin_once(node, timeout_sec=0.05)
        if seen['n'] == 0:
            print('[error] vesc_driver가 응답하지 않아요. bringup이 실행 중인가요?')
            return

        quick = '--quick' in sys.argv
        seq = QUICK_SEQUENCE if quick else SEQUENCE
        print('\n=== 조향 서보 범위 확인 (바퀴 띄운 상태) ===')
        if quick:
            print('두 가지만 봅니다: 왼쪽 끝에서 스토퍼와의 틈, 오른쪽 끝에서 서보 소리.')
            print('y/n 질문은 y 또는 n을 입력해야 다음으로 넘어가요.')
        else:
            print('앞바퀴 바깥면에 긴 막대(자)를 붙이고, 막대 끝 아래 종이에 위치를 표시하세요.')
            stick, _ = ask('막대 길이는 바퀴 중심에서 끝까지 몇 cm? (숫자, 모르면 Enter)', CENTER, False)
            out.write(f'stick_cm: {stick}\n')
            print(f'지금 중앙({CENTER:.3f})입니다. 막대 끝 위치를 종이에 표시하세요.')
        for i, (v, label, qs) in enumerate(seq):
            end = 'END' in label or '0.061' in label
            print(f'\n[{i + 1}/{len(seq)}] 서보 {v:.3f}  ({label})')
            ans = {}
            if not qs:
                ask('  Enter로 다음', v, False)
            hold = v
            for q in qs:
                while True:
                    a, hold = ask('  ' + QUESTIONS[q], hold, end)
                    if q == 'cm' or a.lower() in ('y', 'n'):
                        break
                    print('  y 또는 n을 입력해 주세요.')
                ans[q] = a.lower() if q != 'cm' else a
            out.write(f'servo {v:.3f} | {label} | ' + ' | '.join(f'{k}={x}' for k, x in ans.items()) + '\n')
            out.flush()
        print(f'\n[done] 결과 저장: {path}\n이제 Claude에게 "끝났어"라고 알려 주세요.')
    except KeyboardInterrupt:
        print('\n[Ctrl+C] 중앙으로 복귀')
    finally:
        for _ in range(10):
            send(CENTER)
            time.sleep(0.05)
        out.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
