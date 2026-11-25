#!/usr/bin/env python3
'''
Real-car driving evaluation from extracted logs (tools/bag_extract.py output).

For every autonomous segment (mux passing /drive = mode 'auto'):
  duration, distance (odom), mean / max speed, commanded vs actual speed
  wall clearance from lidar: min distance in front (+-30 deg) and to the sides (30-100 deg)
  steering: weave = dominant frequency and amplitude of the steering command (0.2-5 Hz),
            mean |steering rate|, share of time at full lock
  how it ended: 'stopped by operator' (car still moving when control went back to the keyboard),
                'possible collision' (sudden speed drop while the policy still commanded speed,
                or lidar < 0.15 m), or 'log ended'

usage:
    python3 tools/eval_drive.py ~/bags/run_1117_1421_extracted ~/bags/run_1117_1440_extracted
'''
import argparse
import os

import numpy as np
import pandas as pd

MAX_STEER = 0.497       # command at full lock
CONTACT_DIST = 0.12     # [m] lidar returns this close ...
CONTACT_FRAC = 0.20     # ... on more than this share of the beams -> the car is touching a wall


def segments(mask):
    out, start = [], None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def weave(t, steer):
    '''dominant frequency [Hz] and RMS amplitude [rad] of the steering command in 0.2-5 Hz'''
    if len(t) < 64:
        return np.nan, np.nan
    dt = 0.025
    g = np.arange(t[0], t[-1], dt)
    s = np.interp(g, t, steer)
    s = s - np.convolve(s, np.ones(40) / 40, mode='same')      # remove the slow (cornering) part, 1 s window
    s = s[20:-20]
    if len(s) < 64:
        return np.nan, np.nan
    f = np.fft.rfftfreq(len(s), dt)
    p = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    band = (f >= 0.2) & (f <= 5.0)
    return float(f[band][np.argmax(p[band])]), float(np.sqrt(np.mean(s ** 2)))


def evaluate(log_dir):
    d = pd.read_csv(os.path.join(log_dir, 'driving.csv'))
    z = np.load(os.path.join(log_dir, 'scan.npz'))
    R, ts = z['ranges'], z['t']
    ang = float(z['angle_min']) + np.arange(R.shape[1]) * float(z['angle_increment'])
    Rv = np.where(np.isfinite(R) & (R > 0.05), R, np.inf).astype(np.float32)
    # isolated spikes (single-beam short returns) -> 5-beam median across the scan
    Rv = np.median(np.stack([np.roll(Rv, k, axis=1) for k in range(-2, 3)]), axis=0)
    # beams blocked by the car itself (short in 90% of scans) are not walls
    blocked = np.percentile(np.where(np.isfinite(Rv), Rv, 30.0), 90, axis=0) < 0.35
    front = (np.abs(ang) < np.radians(30)) & ~blocked
    side = (np.abs(ang) >= np.radians(30)) & (np.abs(ang) <= np.radians(100)) & ~blocked
    d_front = np.interp(d.t, ts, Rv[:, front].min(1))
    d_side = np.interp(d.t, ts, Rv[:, side].min(1))
    # wall contact: the car pressed against a wall puts a large part of the scan within a few cm
    # (normal driving: 0-3 % of the beams, against a wall: 30-60 %)
    close = (Rv < CONTACT_DIST) & ~blocked
    contact_scan = close.sum(1) > CONTACT_FRAC * (~blocked).sum()
    contact = np.interp(d.t, ts, contact_scan.astype(float)) > 0.5

    rows = []
    for s, e in segments((d['mode'] == 'auto').values):
        t_all = d.t.values[s:e]
        if t_all[-1] - t_all[0] < 1.0:
            continue
        c = np.flatnonzero(contact[s:e])
        t_contact = t_all[c[0]] - t_all[0] if len(c) else np.nan
        contact_pct = float(np.mean(contact[s:e]) * 100)
        if len(c):                       # evaluate the driving only up to the first wall contact
            e_all, e = e, s + max(c[0], 2)
        else:
            e_all = e
        a = d.iloc[s:e]
        t = a.t.values
        v = a.odom_speed.values
        moving = v > 0.3
        dist = float(np.trapz(np.abs(v), t))
        steer = a.cmd_steer.values
        f, amp = weave(t, steer)
        srate = np.abs(np.diff(steer) / np.maximum(np.diff(t), 1e-3))
        # how it ended
        if np.isfinite(t_contact):
            end = f'wall contact at {t_contact:.1f}s ({contact_pct:.0f}% of the segment against the wall)'
        elif e_all >= len(d):
            end = 'log ended'
        elif d.odom_speed.values[e_all - 1] > 0.5:
            end = 'stopped by operator (moving)'
        else:
            end = 'stopped by operator'
        rows.append({
            'contact': bool(np.isfinite(t_contact)), 'seg_s': t_all[-1] - t_all[0],
            'start_s': t[0], 'dur_s': t[-1] - t[0], 'dist_m': dist,
            'v_mean': float(v[moving].mean()) if moving.any() else 0.0, 'v_max': float(v.max()),
            'v_cmd_mean': float(a.agent_speed[moving].mean()) if moving.any() else np.nan,
            'front_min': float(d_front[s:e][moving].min()) if moving.any() else np.nan,
            'side_min': float(d_side[s:e][moving].min()) if moving.any() else np.nan,
            'side_p10': float(np.percentile(d_side[s:e][moving], 10)) if moving.any() else np.nan,
            'weave_hz': f, 'weave_rms': amp,
            'steer_rate': float(np.mean(srate[moving[1:]])) if moving[1:].any() else np.nan,
            'full_lock_pct': float(np.mean(np.abs(steer[moving]) > 0.95 * MAX_STEER) * 100) if moving.any() else np.nan,
            'end': end,
        })
    return rows


def loc_summary(log_dir):
    '''plan B runs (agent_preview.yaml): how often the car's localization was valid while driving itself'''
    f = os.path.join(log_dir, 'driving.csv')
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    if 'loc_valid' not in d or not d['loc_valid'].notna().any():
        return
    d = d[(d['mode'] == 'auto') & d['loc_valid'].notna()]
    if d.empty:
        return
    v = d['loc_valid'].values > 0.5
    edges = np.diff(np.r_[0, (~v).astype(int), 0])
    runs = (np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]) if (~v).any() else np.array([0])
    dt = float(np.median(np.diff(d['t']))) if len(d) > 1 else 0.025
    print(f'   localization while driving: valid {v.mean()*100:.1f}% of {len(v)} scans, {int((edges == 1).sum())} lost '
          f'stretches (longest {runs.max()*dt:.2f} s), ICP residual median {np.nanmedian(d["loc_res"])*100:.1f} cm')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_dirs', nargs='+')
    a = p.parse_args()
    allrows = []
    for ld in a.log_dirs:
        rows = evaluate(ld)
        name = os.path.basename(os.path.normpath(ld)).replace('_extracted', '')
        print(f'\n== {name}: {len(rows)} autonomous segments (stats up to the first wall contact)')
        print('  start  dur   dist  v_mean v_max v_cmd | front_min side_min side_p10 | weave Hz  rms   rate  lock% | end')
        for r in rows:
            print(f"  {r['start_s']:5.0f} {r['dur_s']:5.1f}s {r['dist_m']:5.1f}m  {r['v_mean']:4.2f}  {r['v_max']:4.2f}  {r['v_cmd_mean']:4.2f} |"
                  f"   {r['front_min']:5.2f}    {r['side_min']:5.2f}    {r['side_p10']:5.2f}  |"
                  f"  {r['weave_hz']:4.2f}  {r['weave_rms']:.3f} {r['steer_rate']:5.2f} {r['full_lock_pct']:5.1f} | {r['end']}")
        allrows += rows
        loc_summary(ld)
    if allrows:
        df = pd.DataFrame(allrows)
        w = df.dur_s / df.dur_s.sum()
        n_c = int(df.contact.sum())
        print(f'\n== total: {len(df)} segments, {df.dur_s.sum():.0f} s / {df.dist_m.sum():.0f} m driven before any wall contact')
        print(f'   wall contacts: {n_c}/{len(df)} segments, {df.dist_m.sum() / max(n_c, 1):.1f} m per contact, '
              f'{df.dur_s.sum() / max(n_c, 1):.1f} s per contact')
        print(f'   mean speed {np.sum(w * df.v_mean):.2f} m/s (commanded {np.nansum(w * df.v_cmd_mean):.2f}), '
              f'closest side wall {df.side_min.min():.2f} m, closest front {df.front_min.min():.2f} m')
        print(f'   weave {np.nanmedian(df.weave_hz):.2f} Hz (median), steering RMS {np.nanmedian(df.weave_rms):.3f} rad, '
              f'mean |steering rate| {np.nanmean(df.steer_rate):.2f} rad/s, full lock {np.nanmean(df.full_lock_pct):.1f}% of the time')
        print('   endings: ' + ', '.join(f'{k} x{v}' for k, v in df.end.value_counts().items()))


if __name__ == '__main__':
    main()
