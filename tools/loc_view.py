#!/usr/bin/env python3
'''
Live localization check for the plan-B (track preview) models, in a terminal over SSH (no RViz, no display).
Read-only: subscribes /scan and the agent's /loc_pose, sends nothing to the car.

Draws the map around the car (walls '.'), the current scan placed at the localized pose ('o' = lands on a map wall,
'X' = does not) and the car ('@', heading '*'), and a verdict:
    OK       pose valid and >= 75 % of the scan lies on the map walls (1 s mean; correct poses: median 97 %)
    CHECK    valid, but the scan fits worse (45-75 %; people / objects near the car also do this): look at the picture
    WRONG    < 45 % fits (the localizer's own 'lost' level): the pose is wrong -> restart the agent at the start spot
    LOST     the agent reports localization lost (it brakes after 0.3 s)
If the pose is wrong at the start: press s in TELEOP, Ctrl-C the agent, put the car on the start spot (corner in
view, nobody standing in the corridor), `tools/drive_session.sh --agent` (it searches the start pose again).

    python3 tools/loc_view.py [--yaml src/f1tenth_ros/f1tenth_stack/config/agent_preview.yaml] [--png logs/loc_view.png]
'''
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src/f1tenth'))


def read_params(path):
    import yaml
    d = yaml.safe_load(open(path))
    for v in d.values():
        if isinstance(v, dict) and 'ros__parameters' in v:
            return v['ros__parameters']
    return {}


def transform(pts, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


def read_walls(map_dir, d=0.10):
    '''map.yaml + map.pgm (tools/map_offline.py) -> (wall cell centers (N, 2), f(points) -> within ~d of a wall)'''
    meta = {}
    for line in open(os.path.join(map_dir, 'map.yaml')):
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    res = float(meta['resolution'])
    org = np.array([float(x) for x in meta['origin'].strip('[]').split(',')][:2])
    raw = open(os.path.join(map_dir, meta.get('image', 'map.pgm')), 'rb').read()
    hdr = raw.split(b'\n', 3)
    W, H = map(int, hdr[1].split())
    occ = np.frombuffer(hdr[3][:W * H], np.uint8).reshape(H, W)[::-1] < 50        # row 0 = lowest y
    iy, ix = np.nonzero(occ)
    walls = np.column_stack([org[0] + (ix + 0.5) * res, org[1] + (iy + 0.5) * res])
    n = int(np.ceil(d / res))
    near = np.zeros_like(occ)
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            if dx * dx + dy * dy <= n * n:
                near |= np.roll(np.roll(occ, dy, 0), dx, 1)

    def near_wall(pts):
        j = np.floor((pts[:, 0] - org[0]) / res).astype(int)
        i = np.floor((pts[:, 1] - org[1]) / res).astype(int)
        ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
        out = np.zeros(len(pts), bool)
        out[ok] = near[i[ok], j[ok]]
        return out
    return walls, near_wall


def verdict(valid, ex):
    if not valid:
        return 'LOST'
    return 'OK' if ex >= 0.75 else ('CHECK' if ex >= 0.45 else 'WRONG')


def ascii_view(walls, scan_w, fit, pose, width=78, height=30, m_per_col=0.15):
    '''character picture centred on the car; a terminal cell is ~2x taller than wide -> rows are 2 x m_per_col'''
    m_per_row = 2 * m_per_col
    x0, y0 = pose[0] - width / 2 * m_per_col, pose[1] - height / 2 * m_per_row
    grid = [[' '] * width for _ in range(height)]

    def put(pts, ch):
        if not len(pts):
            return
        c = ((pts[:, 0] - x0) / m_per_col).astype(int)
        r = height - 1 - ((pts[:, 1] - y0) / m_per_row).astype(int)
        ok = (c >= 0) & (c < width) & (r >= 0) & (r < height)
        for rr, cc in zip(r[ok], c[ok]):
            grid[rr][cc] = ch
    put(walls, '.')
    put(scan_w[fit], 'o')
    put(scan_w[~fit], 'X')
    put(np.array([[pose[0] + k * 0.15 * np.cos(pose[2]), pose[1] + k * 0.15 * np.sin(pose[2])] for k in (1, 2, 3)]), '*')
    put(np.array([pose[:2]]), '@')
    return '\n'.join(''.join(row) for row in grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yaml', default='src/f1tenth_ros/f1tenth_stack/config/agent_preview.yaml',
                    help='agent params file: map_dir, laser_x, scan_topic (MODEL=preview_robust: agent_preview_robust.yaml)')
    ap.add_argument('--map', default=None, help='map dir (default: map_dir of --yaml)')
    ap.add_argument('--rate', type=float, default=2.0, help='[Hz] screen refresh')
    ap.add_argument('--scale', type=float, default=0.15, help='[m] per character column')
    ap.add_argument('--png', default=None, help='also save a picture of the whole map here every 2 s (needs matplotlib)')
    a = ap.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import PoseStamped
    # no scipy / utils.localizer import: runs with the plain ROS python too

    prm = read_params(a.yaml) if os.path.exists(a.yaml) else {}
    map_dir = a.map or prm.get('map_dir')
    if not map_dir:
        raise SystemExit(f'no map: give --map or a --yaml with map_dir ({a.yaml})')
    laser_x = float(prm.get('laser_x', 0.13))
    walls, near_wall = read_walls(map_dir)
    state = dict(scan=None, pose=None, valid=False, t_pose=0.0, res=np.nan, last_png=0.0, ema=None)

    class LocView(Node):
        def __init__(self):
            super().__init__('loc_view')
            qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(LaserScan, prm.get('scan_topic', 'scan'), self.on_scan, qos)
            self.create_subscription(PoseStamped, 'loc_pose', self.on_pose, 10)
            self.create_timer(1.0 / a.rate, self.draw)

        def on_scan(self, m):
            r = np.asarray(m.ranges, float)
            ang = m.angle_min + np.arange(len(r)) * m.angle_increment
            ok = np.isfinite(r) & (r > 0.1) & (r < 10.0)
            state['scan'] = np.column_stack([laser_x + r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])])[::2]

        def on_pose(self, m):
            q = m.pose.orientation
            state['pose'] = np.array([m.pose.position.x, m.pose.position.y, 2 * np.arctan2(q.z, q.w)])
            state['valid'] = m.header.frame_id == 'map'
            state['res'] = m.pose.position.z
            state['t_pose'] = time.monotonic()

        def draw(self):
            out = ['\033[H\033[2J']
            if state['scan'] is None:
                out.append('waiting for /scan (bringup running?)')
            elif state['pose'] is None or time.monotonic() - state['t_pose'] > 2.0:
                out.append('no /loc_pose: the agent is not running, or still searching the start pose '
                           '(agent tab: "[preview] localization: start at ...")')
            else:
                p = state['pose']
                sw = transform(state['scan'], p)
                fit = near_wall(sw)
                ex1 = float(fit.mean()) if len(fit) else 0.0
                k = min(1.0, 1.0 / a.rate)                         # ~1 s running mean
                state['ema'] = ex1 if state['ema'] is None else (1 - k) * state['ema'] + k * ex1
                ex = state['ema']
                v = verdict(state['valid'], ex)
                color = {'OK': '\033[92m', 'CHECK': '\033[93m'}.get(v, '\033[91m')
                out.append(f"{color}{v:6s}\033[0m  scan on map walls {ex * 100:5.1f} % (now {ex1 * 100:3.0f} %)   x {p[0]:6.2f}  y {p[1]:6.2f}  "
                           f"yaw {np.degrees(p[2]):6.1f} deg   ICP residual {state['res'] * 100:.1f} cm")
                out.append(ascii_view(walls, sw, fit, p, m_per_col=a.scale))
                out.append("'.' map wall  'o' scan on a wall  'X' scan off the map  '@' car  '*' heading"
                           + ('   -> WRONG start: s in TELEOP, put the car on the start spot, restart the agent' if v == 'WRONG' else ''))
                if a.png and time.monotonic() - state['last_png'] > 2.0:
                    state['last_png'] = time.monotonic()
                    save_png(a.png, walls, sw, fit, p, v, ex)
            sys.stdout.write('\n'.join(out) + '\n')
            sys.stdout.flush()

    rclpy.init()
    node = LocView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown() if hasattr(rclpy, 'try_shutdown') else rclpy.shutdown()


def save_png(path, walls, sw, fit, p, v, ex):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(walls[:, 0], walls[:, 1], ',', color='0.5')
    ax.plot(sw[fit, 0], sw[fit, 1], '.', ms=2, color='g')
    ax.plot(sw[~fit, 0], sw[~fit, 1], '.', ms=3, color='r')
    ax.arrow(p[0], p[1], 0.4 * np.cos(p[2]), 0.4 * np.sin(p[2]), width=0.05, color='b')
    ax.set_aspect('equal')
    ax.set_title(f'{v}: scan on map walls {ex * 100:.0f} %')
    tmp = path + '.tmp.png'
    fig.savefig(tmp, dpi=80)
    plt.close(fig)
    os.replace(tmp, path)


if __name__ == '__main__':
    main()
