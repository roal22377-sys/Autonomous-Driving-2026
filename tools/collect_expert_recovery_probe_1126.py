#!/usr/bin/env python3
"""Training-only student-prefix/expert continuation feasibility probe.

Writes separate raw diagnostic shards, not a replay cache or training dataset
adoption. Full-episode student controls and exact-prefix checks accompany every
handoff. Failed continuations are explicitly ineligible for future review.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC/'installation')]
DEST = Path('results/expert_recovery_probe_1126')
CONFIG = 'results/ppo_s2r/29/backup/f1tenth.yaml'
STUDENT = 'results/ppo_s2r/29/checkpoint/model_880000.pt'
SELECTION_SEED = 112629


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(destination, failed_only=False, student_checkpoint=None):
    global IL, torch, np, ACTOR, MEAN, STD, COLS, EXPERT, DEST, FAILED_ONLY, STUDENT
    DEST = Path(destination)
    FAILED_ONLY = failed_only
    if student_checkpoint is not None:
        STUDENT = str(student_checkpoint)
    import numpy as np
    import torch
    import train_il_s2r as IL
    from ruamel.yaml import YAML
    from utils.vectorize import RunningMeanStd
    from utils.policy_transfer import columns
    torch.set_num_threads(1)
    args = IL.get_args(['--task_config', CONFIG])
    ck = torch.load(STUDENT, map_location='cpu', weights_only=False)
    if not all(torch.isfinite(v).all().item() for v in ck['actor'].values()):
        raise ValueError('Nonfinite student actor')
    ACTOR = IL.actor_from_state(ck['actor']); ACTOR.eval()
    with torch.no_grad():
        probe = ACTOR(torch.zeros(args.obs_dim)).loc
    if probe.shape != (2,) or not torch.isfinite(probe).all():
        raise ValueError('Student observation/action shape or output is invalid')
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/imitation/1607/sim2real', required=True)
    if 'obs_mean' in ck or 'obs_var' in ck:
        np.testing.assert_array_equal(ck['obs_mean'], rms.mean)
        np.testing.assert_array_equal(ck['obs_var'], rms.var)
    MEAN, STD = rms.mean, np.sqrt(rms.var+1e-8)
    COLS = columns(args)
    EXPERT = dict(YAML().load(Path('configs/task/f1tenth_preview_1126.yaml'))['expert_plan'])


def environment(track, seed):
    from utils.env import F1Wrapper
    args = IL.get_args(['--task_config', CONFIG]); args.seed = seed
    np.random.seed(seed); random.seed(seed)
    env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='eval', tracks=[track])
    env.expert_cfg = dict(EXPERT); env.fixed_speed_cap = 8.; env.start_mode = 'center'
    obs, _ = env.reset(options={'track':track})
    return args, env, obs


def trigger(obs, elapsed):
    if elapsed < 2.5:
        return None
    col = lambda name: float(obs[COLS.index(name)])
    if col('preview/valid') > .5:
        if max(abs(col('preview/curv/'+d)) for d in ('0.5','1','2')) >= .3:
            return 'near_curvature_ge03'
        if abs(col('preview/e_psi')) >= .15:
            return 'reference_heading_ge015'
        if abs(col('preview/e_y')) >= .15:
            return 'lateral_offset_ge015'
    return '30s_prefix_limit' if elapsed >= 30. else None


def drive(job):
    from utils.env import normalize_speed
    from utils.rule_based import RacelinePlanner
    family, track, seed = job
    args, env, obs = environment(track, seed)
    history = []; switch_step = None; switch_reason = None
    try:
        while True:
            reason = trigger(obs, float(env.elapsed))
            if switch_step is None and reason is not None:
                switch_step, switch_reason = len(history), reason
            with torch.no_grad():
                action = ACTOR(torch.as_tensor((obs-MEAN)/STD, dtype=torch.float32)).loc.numpy()
            action = np.clip(action, -1, 1)
            history.append((obs.copy(), action.copy(), env.vehicle_state().copy(), float(env.elapsed)))
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        baseline = dict(success=bool(env.success), collision=bool(env.collision),
            seconds=float(env.elapsed), steps=len(history), lap_times=[float(t) for t in env.lap_times])
        lead_steps = int(round(2./env.timestep))
    finally:
        env.close()
    row = dict(family=family, track=track, track_sha256=sha(track), seed=seed,
        baseline=baseline, switch_step=switch_step, switch_reason=switch_reason,
        admissible_for_future_review=False)
    if FAILED_ONLY:
        if baseline['success']:
            row.update(status='baseline_complete_no_handoff', switch_step=None, switch_reason=None)
            return row
        switch_step = max(0,len(history)-lead_steps)
        switch_reason = '2s_before_training_failure'
        row.update(switch_step=switch_step,switch_reason=switch_reason,
                   handoff_at_reset=(switch_step==0))
    if switch_step is None:
        row['status'] = 'no_handoff_before_termination'
        return row
    args, env, obs = environment(track, seed)
    samples = []; verified = 0; step = 0
    try:
        planner = RacelinePlanner(env._env.unwrapped.track, args.length_f+args.length_r,
            v_min=env.min_speed, v_max=env.speed_cap, max_steer=env.max_steer,
            max_steer_left=env.steer_left_max, max_steer_right=env.steer_right_max,
            **IL.expert_kwargs(env))
        switch_time = history[switch_step][3]
        while True:
            if step <= switch_step:
                np.testing.assert_array_equal(obs, history[step][0])
                np.testing.assert_array_equal(env.vehicle_state(), history[step][2])
                assert float(env.elapsed) == history[step][3]
                verified += 1
            sq, vq = env.pending_commands()
            speed, wheel = planner.plan(env.vehicle_state(), sq, vq, env.timestep)
            if step < switch_step:
                action = history[step][1]
            else:
                action = np.clip(np.asarray([env.steer_command(wheel),
                    normalize_speed(speed,env.min_speed,env.max_speed)], dtype=np.float32), -1, 1)
                samples.append((obs.copy(), action.copy(), env.vehicle_state().copy(),
                    np.asarray(sq), np.asarray(vq), float(speed), float(wheel), float(env.elapsed)))
            obs, _, term, trunc, _ = env.step(action)
            step += 1
            if term or trunc:
                break
        assert samples and verified == switch_step+1
        O = np.asarray([s[0] for s in samples], np.float32)
        Y = np.asarray([s[1] for s in samples], np.float32)
        assert O.shape == (len(samples),132) and Y.shape == (len(samples),2)
        assert np.isfinite(O).all() and np.isfinite(Y).all() and np.abs(Y).max() <= 1
        assert np.all(O[:,COLS.index('preview/valid')] > .5)
        assert np.all(O[:,COLS.index('speed_cap')] == 8)
        shard = DEST/'shards'/('%s_%s_seed%d.npz' % (family,Path(track).stem,seed))
        np.savez_compressed(shard,O=O,Y=Y,columns=np.asarray(COLS),
            state=np.asarray([s[2] for s in samples]),
            pending_steer=np.asarray([s[3] for s in samples]),
            pending_speed=np.asarray([s[4] for s in samples]),
            teacher_speed=np.asarray([s[5] for s in samples]),
            teacher_wheel=np.asarray([s[6] for s in samples]),
            command_time=np.asarray([s[7] for s in samples]),
            seconds_since_handoff=np.asarray([s[7]-switch_time for s in samples]))
        continuation = dict(success=bool(env.success), collision=bool(env.collision),
            seconds=float(env.elapsed), steps=step, lap_times=[float(t) for t in env.lap_times])
        row.update(status='complete_case', handoff=continuation, switch_time=switch_time,
            prefix_observations_and_states_exact=verified, rows=len(samples),
            first_two_second_rows=sum(s[7]-switch_time <= 2 for s in samples),
            shard=str(shard), shard_sha256=sha(shard),
            admissible_for_future_review=bool(env.success),
            outcome=('both_complete' if baseline['success'] else 'rescued') if env.success
                    else ('handoff_regressed' if baseline['success'] else 'both_failed'))
        return row
    finally:
        env.close()


def main():
    global DEST, STUDENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed_open',action='store_true',
        help='Audit all184 open training tracks; branch2s before student failure only')
    parser.add_argument('--student_checkpoint', default=STUDENT,
        help='Frozen actor checkpoint; embedded normalization must match frozen IL1607')
    parser.add_argument('--destination',
        help='Separate collection directory; must not already exist')
    cli = parser.parse_args()
    if cli.failed_open:
        DEST = Path('results/expert_recovery_failures_1126')
    if cli.destination:
        DEST = Path(cli.destination)
    os.chdir(SRC)
    STUDENT = str(Path(cli.student_checkpoint).resolve())
    if not Path(STUDENT).is_file():
        raise FileNotFoundError(STUDENT)
    import numpy as np
    import train_il_s2r as IL
    from utils.training_mix import groups_for
    from eval_open_space import require_fresh_open_train
    if DEST.exists():
        raise FileExistsError('Preserve existing collection; inspect its process before resuming')
    args = IL.get_args(['--task_config',CONFIG])
    regular = [str(Path(p).resolve()) for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p]
    groups = groups_for(args,regular)
    original_names = {m+d for m in IL.MAPS for d in ('','_rev')}
    originals = sorted(p for p in regular if Path(p).stem in original_names)
    opens = sorted(str(Path(p).resolve()) for p in groups['open'])
    assert len(originals)==46 and len(opens)==184
    rng = np.random.default_rng(SELECTION_SEED)
    jobs = []
    if cli.failed_open:
        jobs = [('open',p,64000+i) for i,p in enumerate(opens)]
    else:
        for family,pool in [('regular',originals),('open',opens)]:
            chosen = sorted(rng.choice(len(pool),12,replace=False).tolist())
            for i in chosen:
                jobs.append((family,pool[i],63000+len(jobs)))
    regular_jobs = [p for family,p,_ in jobs if family=='regular']
    if regular_jobs:
        IL.require_fresh_variants(regular_jobs)
    require_fresh_open_train([p for family,p,_ in jobs if family=='open'])
    allowed = {str(Path(p).resolve()) for g in ('regular','open') for p in groups[g]}
    assert len(jobs)==(184 if cli.failed_open else 24)
    assert all(p in allowed and IL.HOLDOUT not in p for _,p,_ in jobs)
    # Validate before starting the pool: a failing initializer otherwise causes
    # multiprocessing to repeatedly spawn replacement workers.
    initialize(str(DEST), cli.failed_open, STUDENT)
    (DEST/'shards').mkdir(parents=True)
    report = dict(status='running', purpose='Training-only recovery feasibility; not an adopted cache',
        student_checkpoint=STUDENT,student_sha256=sha(STUDENT),config_sha256=sha(CONFIG),
        normalizer_sha256=sha('results/imitation/1607/sim2real/agent_obs_scale.pkl'),
        dynamics_sha256=sha('configs/task/dynamic.yaml'),expert_config_sha256=sha('configs/task/f1tenth_preview_1126.yaml'),
        planner_code_sha256=sha('utils/rule_based.py'),selection_seed=None if cli.failed_open else SELECTION_SEED,
        protocol='all_training_open_failures' if cli.failed_open else 'balanced_early_continuations',
        jobs=[dict(family=f,track=p,seed=s,track_sha256=sha(p)) for f,p,s in jobs],
        trigger=('For student failures only:2s before termination, or reset if the episode is shorter.' if cli.failed_open else
                 'After2.5s: preview valid and nearcurvature>=.3, abs referenceheading>=.15 or abs lateraloffset>=.15; otherwise at30s.'),
        runs=[],qualification=(('All184 training opening variants, fixed seeds64000+sortedindex. '
            'Only failed student trajectories receive an expert continuation; successful controls produce no shards. '
            'This is explicit training-failure mining, not a held-out performance comparison. ' if cli.failed_open else
            'Predeclared random12 original training directions plus12 training opening variants. ')+
            'No held-out opening or w105 observations. Expert sees privileged pose/actuator queues; '
            'completed continuations prove simulator feasibility, not actor observability or a deployable '
            'policy. Shards contain only executed whole expert actions after an exact student prefix. '
            'Only full2lap successes are admissible for later review; nothing is automatically fed to '
            'IL/PPO. Caps8 and valid preview only; this does not cover outages or all rare recoveries.'))
    def save():
        temp=DEST/'collection.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(DEST/'collection.json')
    save()
    with mp.get_context('spawn').Pool(2,initializer=initialize,initargs=(str(DEST),cli.failed_open,STUDENT)) as pool:
        for row in pool.imap_unordered(drive,jobs):
            report['runs'].append(row);save()
            print('CASE',len(report['runs']),'/',len(jobs),Path(row['track']).name,row.get('outcome',row['status']),flush=True)
    report['status']='complete_not_applied';save()
    print('Complete; inspect outcomes and provenance before any training use',flush=True)


if __name__=='__main__':
    main()
