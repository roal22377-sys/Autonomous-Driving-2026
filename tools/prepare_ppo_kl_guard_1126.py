#!/usr/bin/env python3
"""Stage an opt-in transactional actor KL guard without changing the learner."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
BASE_SHA = '8efbabe8723b8307f29ed397a387ebea9a65ab816cfe967b3838f203963580d9'


def main():
    source = SRC/'train_ppo_s2r.py'
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Review staged patch against the changed learner')
    code = raw.decode()

    def replace(old, new):
        nonlocal code
        if code.count(old) != 1:
            raise ValueError('Ambiguous or missing injection: '+old[:100])
        code = code.replace(old, new)

    replace('import argparse\n', 'import argparse\nimport copy\n')
    line = "    p.add_argument('--max_kl', type=float, default=0.015)\n"
    replace(line, line+"    p.add_argument('--kl_guard', choices=['epoch', 'minibatch_rollback'], default='epoch',\n"
            "                   help='epoch: legacy end-of-epoch stop; minibatch_rollback: reject the actor step '\n"
            "                        'that exceeds rollout-mean KL, restore actor/optimizer/cache RNG, and stop; '\n"
            "                        'the rejected minibatch critic update is retained')\n")
    line = "        state = torch.load(resume_path, map_location='cpu', weights_only=False)\n"
    replace(line, line+"        if state['args'].get('kl_guard', 'epoch') != args.kl_guard:\n"
            "            raise ValueError('KL guard mode changed during resume; start a declared new trial')\n")
    replace('        retention_sum, retention_steps = 0., 0\n',
            '        retention_sum, retention_steps = 0., 0\n'
            '        guard_rejected, guard_attempted_kl = False, 0.0\n')
    replace('                d = actor(S[idx])\n',
            "                if args.kl_guard == 'minibatch_rollback':\n"
            '                    guard_before = copy.deepcopy((actor.state_dict(), opt_actor.state_dict(),\n'
            '                        retention.state_dict() if retention is not None else None))\n'
            '                d = actor(S[idx])\n')
    replace('                opt_actor.step()\n', '''                opt_actor.step()
                if args.kl_guard == 'minibatch_rollback':
                    with torch.no_grad():
                        guard_attempted_kl = torch.distributions.kl_divergence(old_dist, actor(S)).sum(-1).mean().item()
                    if not np.isfinite(guard_attempted_kl) or guard_attempted_kl > args.max_kl:
                        actor.load_state_dict(guard_before[0])
                        opt_actor.load_state_dict(guard_before[1])
                        opt_actor.zero_grad()
                        if retention is not None:
                            retention.load_state_dict(guard_before[2])
                            retention_sum -= float(retention_loss.detach())
                            retention_steps -= 1
                        guard_rejected = True
                        break
''')
    replace('                if kl > args.max_kl:\n',
            '                if guard_rejected or kl > args.max_kl:\n')
    line = "        if retention is not None:\n            row.update(retention_loss=retention_sum / max(1, retention_steps),"
    replace(line, "        row.update(kl_guard=args.kl_guard, kl_guard_rejected_minibatches=int(guard_rejected),\n"
            "                   actor_attempted_minibatches=n_mb, actor_accepted_minibatches=n_mb-int(guard_rejected),\n"
            "                   kl_guard_attempted_kl=guard_attempted_kl if np.isfinite(guard_attempted_kl) else None)\n"+line)
    ast.parse(code)
    stage = SRC/'results/preflight_1126/kl_guard_staged'
    stage.mkdir(parents=True, exist_ok=True)
    target = stage/'train_ppo_s2r.py'
    if target.exists():
        raise FileExistsError('Preserve previous staged implementation')
    target.write_text(code)
    (stage/'learner.patch').write_text(''.join(difflib.unified_diff(raw.decode().splitlines(True),
        code.splitlines(True), fromfile='production/train_ppo_s2r.py', tofile='staged/train_ppo_s2r.py')))
    report = dict(status='staged_not_applied', production_sha256=BASE_SHA,
        staged_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        staged_file=str(target.relative_to(ROOT)),
        qualification='Syntax checked only. Must pass unchanged-default and active rollback optimizer probes, parser/resume checks before adoption. Mean KL is measured before exploration-noise scheduling, as in the control.')
    (stage/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    assert source.read_bytes() == raw
    print(json.dumps(report))


if __name__ == '__main__':
    main()
