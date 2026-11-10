#!/usr/bin/env python3
'''
ROS 2 bag -> CSV / NPZ  (for ROS 2 Foxy, which has no rosbag2_py)

Reads the bag's sqlite3 .db3 file(s) directly and deserializes with rclpy, so it
also works on a bag whose recording was cut off (no metadata.yaml).

usage:
    source /opt/ros/foxy/setup.bash && source ~/ros2_ws/install/setup.bash
    python3 tools/bag_extract.py ~/bags/run1                 # -> ~/bags/run1_extracted/
    python3 tools/bag_extract.py ~/bags/run1 -o out --topics /scan /odom

outputs:
    <topic>.csv   every message, one row each; nested scalar fields are flattened
                  (drive.speed, pose.pose.position.x, state.current_motor, ...).
                  Array fields are skipped. t = seconds since the start of the bag.
    <topic>.npz   LaserScan topics: ranges (N, beams), t, stamp, angle_min, angle_increment
    driving.csv   one row per /scan: command sent to the car (/ackermann_cmd), agent (/drive)
                  and keyboard (/teleop) commands, odom pose/speed, VESC erpm/current, and
                  which source the mux was passing (manual / auto / idle)
'''

import argparse
import glob
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

MUX_TIMEOUT = 0.2       # [s] ackermann_mux timeout, src/f1tenth_ros/f1tenth_stack/config/mux.yaml
LASER_TYPE = 'sensor_msgs/msg/LaserScan'


def find_db3(path):
    if os.path.isfile(path):
        return [path]
    def split_idx(f):                       # run1_0.db3, run1_1.db3, ... (split bags)
        m = re.search(r'_(\d+)\.db3$', f)
        return int(m.group(1)) if m else 0
    files = sorted(glob.glob(os.path.join(path, '*.db3')), key=split_idx)
    if not files:
        sys.exit(f'no .db3 file in {path}')
    return files


def flatten(msg, prefix='', out=None):
    out = {} if out is None else out
    for name in msg.get_fields_and_field_types():
        value = getattr(msg, name)
        key = prefix + name
        if hasattr(value, 'sec') and hasattr(value, 'nanosec'):     # builtin Time / Duration
            out[key] = value.sec + value.nanosec * 1e-9
        elif hasattr(value, 'get_fields_and_field_types'):         # nested message
            flatten(value, key + '.', out)
        elif isinstance(value, (bool, int, float, str)):
            out[key] = value
        # arrays (scan ranges, covariances, tf lists) are skipped here
    return out


def read_bag(db_files, topics):
    t0 = min(sqlite3.connect(db).execute('SELECT MIN(timestamp) FROM messages').fetchone()[0] or 0
             for db in db_files)
    rows, scans, classes = {}, {}, {}
    for db in db_files:
        conn = sqlite3.connect(db)
        info = {tid: (name, typ) for tid, name, typ in conn.execute('SELECT id, name, type FROM topics')}
        for tid, stamp_ns, data in conn.execute('SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp'):
            name, typ = info[tid]
            if topics and name not in topics:
                continue
            if name not in classes:
                try:
                    classes[name] = get_message(typ)
                except Exception as e:
                    print(f'  skip {name} ({typ}): {e}. Did you source install/setup.bash?')
                    classes[name] = None
            if classes[name] is None:
                continue

            msg = deserialize_message(data, classes[name])
            t = (stamp_ns - t0) * 1e-9
            row = {'t': t}
            row.update(flatten(msg))
            rows.setdefault(name, []).append(row)
            if typ == LASER_TYPE:
                s = scans.setdefault(name, {'t': [], 'stamp': [], 'ranges': [],
                                            'angle_min': msg.angle_min, 'angle_increment': msg.angle_increment})
                s['t'].append(t)
                s['stamp'].append(row.get('header.stamp', np.nan))
                s['ranges'].append(np.asarray(msg.ranges, dtype=np.float32))
    return {name: pd.DataFrame(r) for name, r in rows.items()}, scans


def attach(base, df, colmap, tolerance=None):
    '''Add, at each base time, the latest value of df's columns (renamed by colmap).'''
    cols = [c for c in colmap if df is not None and c in df.columns]
    if cols:
        part = df[['t'] + cols].rename(columns=colmap)
        base = pd.merge_asof(base, part, on='t', direction='backward', tolerance=tolerance)
    for new in colmap.values():
        if new not in base:
            base[new] = np.nan
    return base


def driving_table(dfs):
    base_topic = '/scan' if '/scan' in dfs else '/odom' if '/odom' in dfs else None
    if base_topic is None:
        return None
    base = dfs[base_topic][['t']].copy()

    odom = dfs.get('/odom')
    if odom is not None and 'pose.pose.orientation.w' in odom:
        qx, qy, qz, qw = (odom[f'pose.pose.orientation.{k}'] for k in 'xyzw')
        odom = odom.assign(yaw=np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))

    drive_cols = lambda p: {'drive.speed': f'{p}_speed', 'drive.steering_angle': f'{p}_steer'}
    base = attach(base, dfs.get('/ackermann_cmd'), drive_cols('cmd'))                  # sent to the car
    base = attach(base, dfs.get('/drive'), drive_cols('agent'), tolerance=MUX_TIMEOUT)  # live only
    base = attach(base, dfs.get('/teleop'), drive_cols('key'), tolerance=MUX_TIMEOUT)
    base = attach(base, odom, {'twist.twist.linear.x': 'odom_speed', 'pose.pose.position.x': 'x',
                               'pose.pose.position.y': 'y', 'yaw': 'yaw'})
    base = attach(base, dfs.get('/sensors/core'), {'state.speed': 'erpm', 'state.current_motor': 'motor_current',
                                                   'state.voltage_input': 'battery_v'})

    # mux: keyboard wins while it is publishing, otherwise the agent
    base.insert(1, 'mode', np.where(base['key_speed'].notna(), 'manual',
                                    np.where(base['agent_speed'].notna(), 'auto', 'idle')))
    return base


def main():
    ap = argparse.ArgumentParser(description='ROS 2 Foxy bag -> CSV / NPZ')
    ap.add_argument('bag', help='bag directory (from ros2 bag record -o) or a .db3 file')
    ap.add_argument('-o', '--out', help='output directory (default: <bag>_extracted)')
    ap.add_argument('--topics', nargs='+', help='only these topics, e.g. /scan /odom /drive')
    a = ap.parse_args()

    bag = os.path.abspath(os.path.expanduser(a.bag)).rstrip('/')
    out = a.out or (os.path.splitext(bag)[0] if os.path.isfile(bag) else bag) + '_extracted'
    os.makedirs(out, exist_ok=True)

    db_files = find_db3(bag)
    print(f'reading {", ".join(os.path.basename(f) for f in db_files)}')
    dfs, scans = read_bag(db_files, set(a.topics) if a.topics else None)

    fname = lambda topic: topic.strip('/').replace('/', '_') or 'root'
    for topic, df in sorted(dfs.items()):
        dur = df['t'].iloc[-1] - df['t'].iloc[0]
        rate = f'{(len(df) - 1) / dur:6.1f} Hz' if dur > 0 else '      - '
        if df.shape[1] > 1:
            df.to_csv(os.path.join(out, fname(topic) + '.csv'), index=False)
            note = f'{fname(topic)}.csv'
        else:
            note = '(array-only message, no csv)'
        print(f'  {topic:<22} {len(df):6d} msgs {rate}  {note}')

    for topic, s in scans.items():
        np.savez_compressed(os.path.join(out, fname(topic) + '.npz'),
                            t=np.array(s['t']), stamp=np.array(s['stamp']), ranges=np.stack(s['ranges']),
                            angle_min=s['angle_min'], angle_increment=s['angle_increment'])
        print(f'  {topic:<22} ranges {np.stack(s["ranges"]).shape} -> {fname(topic)}.npz')

    drive = driving_table(dfs)
    if drive is not None:
        drive.to_csv(os.path.join(out, 'driving.csv'), index=False)
        print(f'  driving.csv: {len(drive)} rows, mode counts {drive["mode"].value_counts().to_dict()}')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
