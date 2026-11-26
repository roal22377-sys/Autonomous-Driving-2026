#!/usr/bin/env python3
'''
Graphical localization check for the plan-B (track preview) models on a PC WITHOUT ROS (what RViz would show):
the course map, the car's localized pose and the current LiDAR scan placed at it, live, plus RViz's
"2D Pose Estimate": click on the map where the car is and drag toward where it faces -> the agent searches its
start pose again near there (topic /initialpose, the same message RViz sends).

Over SSH it runs `tools/loc_view.py --stream` on the car (read-only: /scan + the agent's /loc_pose) and draws it here.
Needs the agent running (drive_session.sh with MODEL=preview / preview_robust / preview_il) and matplotlib with a
window backend (PyQt5 in ~/venv/torch210: pip install PyQt5==5.15.10).

    MODEL=preview tools/loc_gui.py                    # or: python3 tools/loc_gui.py --model preview_robust

Picture: grey = map walls, green = scan points on a map wall, red = scan points off the map, blue arrow = car.
Title: OK (>= 75 % of the scan on the walls, pose valid) / CHECK (45-75 %) / WRONG (< 45 %) / LOST / waiting.
Set the pose: press s in TELEOP (the car must stand still), then on the map press the mouse where the car is, drag
toward its heading and release: an orange arrow and the scan placed there (cyan) appear with their fit.
Enter = send it to the car, Esc = cancel. The agent log then shows "[preview] initial pose from map ... searching
near it" and "localization: start at ... (near the given pose)"; the picture turns green when it is right.
'''
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loc_view import read_params, read_walls, transform, verdict    # noqa: E402

CAR_HOST = os.environ.get('CAR_HOST', 'jowonjun@10.106.0.73')
SSH = ['ssh', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3', '-o', 'BatchMode=yes']
CAR_ENV = ('export PYTHONPATH=$HOME/venv/torch210/lib/python3.8/site-packages:$PYTHONPATH && cd ~/ros2_ws && '
           'source /opt/ros/foxy/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=6')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=os.environ.get('MODEL', 'preview'), help='agent config: agent_<model>.yaml')
    ap.add_argument('--rate', type=float, default=4.0, help='[Hz] updates from the car')
    a = ap.parse_args()
    yaml_rel = f'src/f1tenth_ros/f1tenth_stack/config/agent_{a.model}.yaml'
    prm = read_params(os.path.join(REPO, yaml_rel))
    map_dir = prm.get('map_dir')
    if not map_dir:
        raise SystemExit(f'{yaml_rel} has no map_dir (map the course first: tools/map_session.sh)')
    walls, near_wall = read_walls(os.path.join(REPO, map_dir))
    course = prm.get('course')
    center = None
    if course and os.path.exists(os.path.join(REPO, course)):
        c = np.asarray(np.load(os.path.join(REPO, course))['centerline'], float)
        t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
        center = np.column_stack([c, np.arctan2(t[:, 1], t[:, 0])])      # x, y, driving direction of the course

    def course_heading(x, y):
        '''driving direction of the course at the centerline point nearest (x, y) (a click without a drag)'''
        if center is None:
            return 0.0
        i = int(np.argmin(np.hypot(center[:, 0] - x, center[:, 1] - y)))
        return float(center[i, 2])

    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt

    state = dict(scan=None, pose=None, valid=False, res=None, t=0.0, ema=None, err='', proc=None)

    def reader():
        # the car side: loc_view.py --stream (one JSON line per refresh); restarted if the connection drops.
        # Never let an error end this thread (the window would then wait forever)
        while True:
            try:
                read_once()
            except Exception as e:          # noqa: BLE001
                state['err'] = f'stream error: {e!r}, retrying ...'
            time.sleep(2.0)

    def read_once():
        if True:
            cmd = SSH + [CAR_HOST, f'{CAR_ENV} && exec python3 tools/loc_view.py --yaml {yaml_rel} --stream --rate {a.rate}']
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            state['proc'] = p
            for line in p.stdout:
                if not line.startswith('{'):
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                state['scan'] = None if d['scan'] is None else np.asarray(d['scan'], float).reshape(-1, 2)
                state['pose'] = None if d['pose'] is None else np.asarray(d['pose'], float)
                state['valid'], state['res'], state['t'], state['err'] = d['valid'], d['res'], time.monotonic(), ''
            p.wait()
            state['err'] = f'connection to the car ended (exit {p.returncode}), retrying ...'

    threading.Thread(target=reader, daemon=True).start()

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.canvas.manager.set_window_title(f'loc_gui: {a.model} on {map_dir}')
    ax.plot(walls[:, 0], walls[:, 1], ',', color='0.55')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    on_pts, = ax.plot([], [], '.', ms=3, color='tab:green')
    off_pts, = ax.plot([], [], '.', ms=4, color='tab:red')
    hint_pts, = ax.plot([], [], '.', ms=3, color='tab:cyan')
    car = {'arrow': None, 'hint': None}
    drag = {'p0': None, 'pose': None}
    info = fig.text(0.01, 0.01, '', fontsize=9, family='monospace')
    fig.text(0.01, 0.97, 'Set the pose (car standing still, TELEOP s): CLICK where the car is (heading = course direction) '
             'or press-drag toward its heading -> Enter = send, Esc = cancel', fontsize=10, color='tab:purple')

    def arrow(p, color, old):
        if old is not None:
            old.remove()
        return ax.arrow(p[0], p[1], 0.5 * np.cos(p[2]), 0.5 * np.sin(p[2]), width=0.06, color=color, zorder=5)

    def show_hint():
        p = drag['pose']
        car['hint'] = arrow(p, 'tab:orange', car['hint'])
        if state['scan'] is not None:
            sw = transform(state['scan'], p)
            fit = near_wall(sw)
            hint_pts.set_data(sw[:, 0], sw[:, 1])
            ax.set_xlabel(f'given pose x {p[0]:.2f} y {p[1]:.2f} yaw {np.degrees(p[2]):.0f} deg: '
                          f'{fit.mean() * 100:.0f} % of the scan on the walls (cyan). Enter = send, Esc = cancel')
        fig.canvas.draw_idle()

    def clear_hint():
        drag['pose'] = None
        if car['hint'] is not None:
            car['hint'].remove()
            car['hint'] = None
        hint_pts.set_data([], [])
        ax.set_xlabel('')
        fig.canvas.draw_idle()

    def on_press(e):
        if e.inaxes is ax and e.button == 1 and not getattr(fig.canvas.toolbar, 'mode', ''):
            drag['p0'] = (e.xdata, e.ydata)

    def on_release(e):
        if drag['p0'] is None or e.inaxes is not ax:
            drag['p0'] = None
            return
        x0, y0 = drag['p0']
        dx, dy = e.xdata - x0, e.ydata - y0
        drag['p0'] = None
        if np.hypot(dx, dy) < 0.15:
            # a plain click: heading = the course's driving direction there (the direction the map was driven)
            drag['pose'] = np.array([x0, y0, course_heading(x0, y0)])
        else:
            drag['pose'] = np.array([x0, y0, np.arctan2(dy, dx)])
        show_hint()

    def send(p):
        qz, qw = np.sin(p[2] / 2), np.cos(p[2] / 2)
        msg = (f'{{header: {{frame_id: map}}, pose: {{pose: {{position: {{x: {p[0]:.3f}, y: {p[1]:.3f}, z: 0.0}}, '
               f'orientation: {{x: 0.0, y: 0.0, z: {qz:.5f}, w: {qw:.5f}}}}}}}}}')
        cmd = SSH + [CAR_HOST, f"{CAR_ENV} && timeout 15 ros2 topic pub --once /initialpose "
                               f"geometry_msgs/msg/PoseWithCovarianceStamped '{msg}'"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        ok = r.returncode == 0
        print(f'[loc_gui] initial pose x {p[0]:.2f} y {p[1]:.2f} yaw {np.degrees(p[2]):.0f} deg -> '
              f'{"sent" if ok else "FAILED: " + (r.stderr or r.stdout)[-200:]}', flush=True)
        return ok

    def on_key(e):
        if e.key == 'escape':
            clear_hint()
        elif e.key == 'enter' and drag['pose'] is not None:
            p = drag['pose'].copy()
            ax.set_xlabel('sending the pose to the car ...')
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            ok = send(p)
            clear_hint()
            ax.set_xlabel(f'pose x {p[0]:.2f} y {p[1]:.2f} yaw {np.degrees(p[2]):.0f} deg '
                          + ('sent: wait for green (agent log: "start at ... near the given pose")' if ok
                             else 'NOT sent (see the terminal); is the agent running?'))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(_=None):
        age = time.monotonic() - state['t'] if state['t'] else np.inf
        if state['err'] or age > 3.0:
            ax.set_title(state['err'] or 'connecting to the car ... (bringup + agent running?)', color='0.4')
            on_pts.set_data([], [])
            off_pts.set_data([], [])
        elif state['pose'] is None:
            ax.set_title('no /loc_pose: agent not running or still searching the start pose '
                         '(set it: press-drag-release on the map, Enter)', color='tab:orange')
            on_pts.set_data([], [])
            off_pts.set_data([], [])
        else:
            p = state['pose']
            sw = transform(state['scan'], p) if state['scan'] is not None else np.zeros((0, 2))
            fit = near_wall(sw) if len(sw) else np.zeros(0, bool)
            ex1 = float(fit.mean()) if len(fit) else 0.0
            state['ema'] = ex1 if state['ema'] is None else 0.7 * state['ema'] + 0.3 * ex1
            v = verdict(state['valid'], state['ema'])
            on_pts.set_data(sw[fit, 0], sw[fit, 1])
            off_pts.set_data(sw[~fit, 0], sw[~fit, 1])
            car['arrow'] = arrow(p, 'tab:blue', car['arrow'])
            color = {'OK': 'tab:green', 'CHECK': 'goldenrod'}.get(v, 'tab:red')
            res = f'{state["res"] * 100:.1f} cm' if state['res'] is not None and state['res'] >= 0 else '-'
            ax.set_title(f'{v}: {state["ema"] * 100:.0f} % of the scan on the map walls (now {ex1 * 100:.0f} %)   '
                         f'x {p[0]:.2f}  y {p[1]:.2f}  yaw {np.degrees(p[2]):.0f} deg   ICP residual {res}',
                         color=color, fontsize=12, fontweight='bold')
        info.set_text(f'{a.model}  map {map_dir}  car {CAR_HOST}  last update {age:.1f} s ago' if np.isfinite(age)
                      else f'{a.model}  map {map_dir}  car {CAR_HOST}')
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=int(1000 / a.rate))
    timer.add_callback(update)
    timer.start()
    try:
        plt.show()
    finally:
        if state['proc'] is not None:
            state['proc'].terminate()


if __name__ == '__main__':
    main()
