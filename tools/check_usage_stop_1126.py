#!/usr/bin/env python3
"""Read only current-thread quota metadata for the user's20percent stop rule.

Exit20 means save/checkpoint, stop work and pause the goal. Exit3 means unknown
or stale; it never means quota is available. This tool does not pause any goal
or signal any process itself. The supervising assistant handles those actions.
"""
import argparse
import datetime
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max_age_seconds',type=float,default=300.)
    cli=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    state=json.loads((root/'training_state.json').read_text())
    rule=state['user_usage_stop_rule'];assert rule['active'] and rule['window_minutes']==300
    path=Path(rule['session_log'])
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-8*1024*1024))
        if f.tell():f.readline()
        lines=f.readlines()
    found=None
    for line in lines:
        try:entry=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        payload=entry.get('payload',{});quota=payload.get('rate_limits') or {};primary=quota.get('primary') or {}
        if (entry.get('type')=='event_msg' and payload.get('type')=='token_count'
                and quota.get('limit_id')=='codex' and primary.get('window_minutes')==300):
            found=dict(observed_utc=entry['timestamp'],used_percent=primary['used_percent'],resets_at=primary['resets_at'])
    if found is None:
        print(json.dumps(dict(status='unknown',reason='No primary300minute quota snapshot in latest8MiB')));return 3
    stamp=datetime.datetime.fromisoformat(found['observed_utc'].replace('Z','+00:00'))
    age=(datetime.datetime.now(datetime.timezone.utc)-stamp).total_seconds()
    remaining=100-float(found['used_percent'])
    if not 0<=remaining<=100 or not -30<=age<=cli.max_age_seconds:
        print(json.dumps(dict(status='unknown_or_stale',age_seconds=age,last_snapshot=found)));return 3
    stop=remaining<float(rule['stop_when_remaining_percent_less_than'])
    result=dict(found,status='stop_required' if stop else 'continue',remaining_percent=remaining,
                age_seconds=age,stop_below_percent=rule['stop_when_remaining_percent_less_than'])
    dest=root/'src/f1tenth/results/preflight_1126/latest_usage_stop_check.json'
    temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(dest)
    print(json.dumps(result));return 20 if stop else 0


if __name__=='__main__':sys.exit(main())
