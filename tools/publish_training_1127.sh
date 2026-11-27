#!/usr/bin/env bash
# Publish the reviewed training branch using the caller's normal Git credentials.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${script_dir}/.."
branch="$(git branch --show-current)"
if [[ "$branch" != "codex/1126-hairpin-position-ppo" ]]; then
    printf 'Expected training branch; current branch: %s\n' "$branch" >&2
    exit 2
fi
git push https://github.com/roal22377-sys/Autonomous-Driving-2026.git \
    HEAD:refs/heads/codex/1127-raceline-robustness
