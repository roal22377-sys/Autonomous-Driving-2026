#!/usr/bin/env python3
'''
Run the car's agent_node (src/f1tenth_ros/f1tenth_stack/f1tenth_stack/agent_node.py) on a recorded drive, on the
PC, without ROS: rclpy and the message types are replaced by small stand-ins, the node gets the parameters of an
agent yaml, and the log's /odom and /scan are fed to its callbacks in time order. Catches errors in the node
(parameters, model loading, observation size, preview / localization) before a car session, and shows what the
model would have commanded on those scans.

    python3 tools/agent_replay.py LOG [--config src/f1tenth_ros/f1tenth_stack/config/agent_preview.yaml]
                                      [--set name=imitation algo_idx=13 ...] [--t0 --t1]

LOG: logs/<run>_extracted (scan.npz + driving.csv from tools/bag_extract.py / eval_drive.py), or a
tools/sim_map_test.py log. Prints the node's log lines, the number of drive commands, step times, and
the steering / speed it commanded (the car did not run these commands: open loop).
'''
import argparse
import os
import sys
import time
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = os.path.join(REPO, 'src/f1tenth_ros/f1tenth_stack/f1tenth_stack/agent_node.py')


# ------------------------------------------------------------------ ROS stand-ins
class _Param:
    def __init__(self, v):
        self.value = v


class _Logger:
    def __init__(self):
        self.last, self.sev = {}, {}

    def _p(self, lvl, msg, throttle_duration_sec=None, **_):
        # like rclpy (Foxy): one call site must always log at the same severity, else ValueError
        import inspect
        f = inspect.currentframe().f_back.f_back
        site = (f.f_code.co_filename, f.f_lineno)
        if self.sev.setdefault(site, lvl) != lvl:
            raise ValueError(f'Logger severity cannot be changed between calls ({site[0]}:{site[1]})')
        if throttle_duration_sec:
            t = time.monotonic()
            if t - self.last.get(msg[:40], -1e9) < throttle_duration_sec:
                return
            self.last[msg[:40]] = t
        print(f'[node {lvl}] {msg}', flush=True)

    def info(self, msg, **k): self._p('INFO', msg, **k)
    def warn(self, msg, **k): self._p('WARN', msg, **k)
    warning = warn
    def error(self, msg, **k): self._p('ERROR', msg, **k)


class _Pub:
    def __init__(self, topic):
        self.topic, self.msgs = topic, []

    def publish(self, m):
        self.msgs.append(m)


class _Node:
    PARAMS = {}

    def __init__(self, name):
        self._params, self._log, self.pubs, self.subs = {}, _Logger(), {}, {}

    def declare_parameter(self, k, default):
        self._params[k] = self.PARAMS.get(k, default)

    def get_parameter(self, k):
        return _Param(self._params[k])

    def get_logger(self):
        return self._log

    def create_subscription(self, typ, topic, cb, qos):
        self.subs[topic] = cb

    def create_publisher(self, typ, topic, qos):
        self.pubs[topic] = _Pub(topic)
        return self.pubs[topic]


def _msg(**fields):
    class M(types.SimpleNamespace):
        def __init__(self, **kw):
            d = {k: (v() if callable(v) else v) for k, v in fields.items()}
            d.update(kw)
            super().__init__(**d)
    return M


def install_ros_stubs():
    NS = types.SimpleNamespace
    Header = _msg(stamp=0.0, frame_id='')
    Point = _msg(x=0.0, y=0.0, z=0.0)
    Quat = _msg(x=0.0, y=0.0, z=0.0, w=1.0)
    Pose = _msg(position=Point, orientation=Quat)
    mods = {
        'rclpy': NS(init=lambda **k: None, spin=lambda n: None),
        'rclpy.duration': NS(),
        'rclpy.time': NS(Time=object),
        'rclpy.node': NS(Node=_Node),
        'rclpy.qos': NS(QoSProfile=lambda **k: None, ReliabilityPolicy=NS(BEST_EFFORT=0), HistoryPolicy=NS(KEEP_LAST=0)),
        'sensor_msgs': NS(), 'sensor_msgs.msg': NS(LaserScan=object),
        'nav_msgs': NS(), 'nav_msgs.msg': NS(Odometry=object),
        'ackermann_msgs': NS(),
        'ackermann_msgs.msg': NS(AckermannDriveStamped=_msg(header=Header, drive=_msg(steering_angle=0.0, speed=0.0))),
        'geometry_msgs': NS(), 'geometry_msgs.msg': NS(PoseStamped=_msg(header=Header, pose=Pose),
                                                       PoseWithCovarianceStamped=_msg(header=Header, pose=_msg(pose=Pose))),
    }
    for k, v in mods.items():
        sys.modules[k] = types.ModuleType(k)
        sys.modules[k].__dict__.update(vars(v))
    sys.modules['rclpy'].duration = sys.modules['rclpy.duration']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log')
    ap.add_argument('--config', default=os.path.join(REPO, 'src/f1tenth_ros/f1tenth_stack/config/agent_preview.yaml'))
    ap.add_argument('--set', nargs='*', default=[], help='override parameters, e.g. algo_idx=13 speed_limit=3')
    ap.add_argument('--t0', type=float, default=-np.inf)
    ap.add_argument('--t1', type=float, default=np.inf)
    ap.add_argument('--blackout', default=None, help='t0,t1: feed empty scans in this interval (localization lost)')
    a = ap.parse_args()
    from ruamel.yaml import YAML
    params = dict(YAML().load(open(a.config))['agent_node']['ros__parameters'])
    for kv in a.set:
        k, v = kv.split('=', 1)
        params[k] = type(params[k])(v) if k in params and not isinstance(params[k], str) else v
    os.chdir(REPO)                        # the car runs the node from the checkout root (~/ros2_ws)
    install_ros_stubs()
    _Node.PARAMS = params
    import importlib.util
    spec = importlib.util.spec_from_file_location('agent_node', NODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.TELEOP_HOLD = -1.0

    class SimClock:
        '''the node's time.monotonic() follows the log (+ the node's own processing time), so its timeouts,
        throttles and the lost-stop timer behave as on the car; everything else is the real time module'''
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now + (time.perf_counter() - self.t_call)

        def __getattr__(self, k):
            return getattr(time, k)
    clock = SimClock()
    clock.t_call = time.perf_counter()
    mod.time = clock
    node = mod.AgentNode()
    # log
    import pandas as pd
    z = np.load(f'{a.log}/scan.npz')
    R, ts = z['ranges'].astype(np.float32), z['t']
    d = pd.read_csv(f'{a.log}/driving.csv')
    d = d[np.isfinite(d['x']) & np.isfinite(d['y']) & np.isfinite(d['yaw'])]
    sp = (d['odom_speed'].values if 'odom_speed' in d else
          np.r_[0, np.hypot(np.diff(d['x']), np.diff(d['y'])) / np.maximum(np.diff(d['t']), 1e-3)])     # sim logs: from x, y
    m = (ts >= max(a.t0, d['t'].min())) & (ts <= min(a.t1, d['t'].max()))
    R, ts = R[m], ts[m]
    Q = lambda t, c: np.interp(t, d['t'], c)
    yaw = np.unwrap(d['yaw'].values)
    Odom = _msg(pose=_msg(pose=_msg(position=_msg(x=0.0, y=0.0, z=0.0), orientation=_msg(x=0.0, y=0.0, z=0.0, w=1.0))),
                twist=_msg(twist=_msg(linear=_msg(x=0.0))))
    Scan = _msg(header=_msg(stamp=0.0), ranges=None, angle_min=float(z['angle_min']), angle_increment=float(z['angle_increment']))
    T = []
    odom_cb = node.subs.get('odom') or node.subs.get(params.get('odom_topic', 'odom'))
    scan_cb = node.subs[params.get('scan_topic', 'scan')]
    for k, t in enumerate(ts):
        clock.now, clock.t_call = float(t), time.perf_counter()
        if odom_cb is not None:
            o = Odom()
            o.pose.pose.position.x, o.pose.pose.position.y = float(Q(t, d['x'])), float(Q(t, d['y']))
            th = float(np.interp(t, d['t'], yaw))
            o.pose.pose.orientation.z, o.pose.pose.orientation.w = np.sin(th / 2), np.cos(th / 2)
            o.twist.twist.linear.x = float(np.interp(t, d['t'], sp))
            odom_cb(o)
        s = Scan()
        s.header.stamp = float(t)
        rr = R[k].astype(float)
        rr[~np.isfinite(rr)] = np.inf
        if a.blackout and float(a.blackout.split(',')[0]) <= t < float(a.blackout.split(',')[1]):
            rr[:] = np.inf
        s.ranges = list(rr)
        t0 = time.perf_counter()
        scan_cb(s)
        T.append(time.perf_counter() - t0)
    drv = node.pubs[params.get('drive_topic', 'drive')].msgs
    if os.environ.get('REPLAY_DUMP'):          # commanded (stamp, steering, speed) per scan, for checking the node's output
        np.savetxt(os.environ['REPLAY_DUMP'], np.array([[float(m.header.stamp) if not hasattr(m.header.stamp, 'sec') else 0.0,
                   m.drive.steering_angle, m.drive.speed] for m in drv]), delimiter=',', header='stamp,steer,speed', comments='')
    T = np.array(T)
    if a.blackout and drv:
        b0, b1 = (float(x) for x in a.blackout.split(','))
        tt = np.array([m.header.stamp for m in drv])
        vv = np.array([m.drive.speed for m in drv])
        for lo, hi, lab in ((b0 - 1, b0, 'before'), (b0, b0 + 0.3, 'first 0.3 s'), (b0 + 0.35, b1, 'rest of the blackout'),
                            (b1, b1 + 2, '2 s after')):
            m = (tt >= lo) & (tt < hi)
            if m.any():
                print(f'[replay] blackout {lab}: commanded speed {vv[m].min():.2f}..{vv[m].max():.2f} m/s')
    print(f'\n[replay] {len(ts)} scans -> {len(drv)} drive commands; scan callback median {np.median(T)*1e3:.1f} ms, '
          f'p99 {np.percentile(T, 99)*1e3:.1f} ms, max {T.max()*1e3:.0f} ms (this PC; first scan includes the start search)')
    if drv:
        st = np.array([m.drive.steering_angle for m in drv])
        v = np.array([m.drive.speed for m in drv])
        print(f'[replay] commanded steering {st.min():+.3f}..{st.max():+.3f} rad (mean |.| {np.abs(st).mean():.3f}), '
              f'speed {v.min():.2f}..{v.max():.2f} m/s (mean {v.mean():.2f})')
    if 'loc_pose' in node.pubs:
        lp = node.pubs['loc_pose'].msgs
        ok = np.mean([m.header.frame_id == 'map' for m in lp]) if lp else 0.0
        print(f'[replay] loc_pose: {len(lp)} poses, valid {ok*100:.1f}%')


if __name__ == '__main__':
    main()
