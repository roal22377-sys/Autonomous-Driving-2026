#!/usr/bin/env python3
"""Build and validate the stopped PPO30 resume argv without launching learning.

This is a fallback launch path using the stored reference maps. It preserves
PPO30's hyperparameters and checkpoint; improved reference promotion and
traffic robustness remain separate gates documented in the training brief.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import redirect_stdout
import io

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'


class ParserCaptured(Exception):
    def __init__(self, parser):
        self.parser = parser


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', required=True)
    options = cli.parse_args()
    output = Path(options.output).resolve()
    os.chdir(SRC)
    sys.path.insert(0, str(SRC))
    saved_path = SRC / 'results/ppo_s2r/30/backup/ppo_args.json'
    saved = json.loads(saved_path.read_text())
    import train_ppo_s2r as train

    original = argparse.ArgumentParser.parse_args
    def capture(parser, args=None, namespace=None):
        raise ParserCaptured(parser)
    argparse.ArgumentParser.parse_args = capture
    try:
        train.get_args()
    except ParserCaptured as caught:
        parser = caught.parser
    finally:
        argparse.ArgumentParser.parse_args = original

    argv = []
    for action in parser._actions:
        key = action.dest
        if key not in saved or key in ('help', 'resume') or saved[key] == action.default:
            continue
        option = next((x for x in action.option_strings if x.startswith('--')), None)
        if not option:
            raise ValueError('No long option for ' + key)
        value = saved[key]
        if isinstance(action, argparse._StoreTrueAction):
            if value is not True:
                raise ValueError('Cannot reconstruct flag ' + key)
            argv.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if value is not False:
                raise ValueError('Cannot reconstruct flag ' + key)
            argv.append(option)
        elif isinstance(value, list):
            argv.extend([option] + [str(x) for x in value])
        else:
            argv.extend([option, str(int(value)) if isinstance(value, bool) else str(value)])
    argv.append('--resume')
    parsed = vars(parser.parse_args(argv))
    for action in parser._actions:
        key = action.dest
        if key in saved and key != 'resume' and key in parsed:
            if parsed[key] != saved[key]:
                raise ValueError('Stored argument mismatch: ' + key + ' ' + repr((saved[key], parsed[key])))
    if saved['algo_idx'] != 30 or saved['name'] != 'ppo_s2r':
        raise ValueError('Unexpected saved experiment')
    with tempfile.TemporaryDirectory(prefix='ppo30-merge-audit-') as temporary:
        with redirect_stdout(io.StringIO()):
            merged = train.merge_sim_override(
                'results/' + saved['il'] + '/backup/f1tenth.yaml',
                saved['sim_override'], temporary)
        if sha(merged) != sha(SRC / 'results/ppo_s2r/30/backup/f1tenth.yaml'):
            raise ValueError('Current IL config plus sim override no longer reproduces PPO30 config')
    if sha(SRC / 'configs/task/dynamic.yaml') != sha(SRC / 'results/ppo_s2r/30/backup/dynamic.yaml'):
        raise ValueError('Current dynamics config differs from PPO30 backup')
    required = {
        'checkpoint': SRC / 'results/ppo_s2r/30/checkpoint/resume.pt',
        'saved_args': saved_path,
        'il_config': SRC / ('results/' + saved['il'] + '/backup/f1tenth.yaml'),
        'sim_override': SRC / saved['sim_override'],
        'retention_file': SRC / saved['retention_file'],
        'retention_manifest': SRC / saved['retention_manifest'],
        'merged_config': SRC / 'results/ppo_s2r/30/backup/f1tenth.yaml',
        'dynamic_config': SRC / 'results/ppo_s2r/30/backup/dynamic.yaml',
        'actor_init': SRC / ('results/' + saved['init']),
    }
    artifacts = {key: dict(path=str(path.relative_to(SRC)), sha256=sha(path))
                 for key, path in required.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='prepared_not_run', training_started=False,
                  purpose='Exact PPO30 checkpoint resume with its saved arguments; baseline fallback only.',
                  cwd=str(SRC), command=['/home/daehyeok/venv/f1tenth/bin/python',
                                         'train_ppo_s2r.py'] + argv,
                  artifacts=artifacts,
                  caution='Do not execute before user explicitly asks to start training. New line and dynamic traffic are not promoted by this resume command.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], options=len(argv),
                          artifacts=list(artifacts), output=str(output))))


if __name__ == '__main__':
    main()
