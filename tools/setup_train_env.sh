#!/bin/bash
# Create the training / evaluation Python environment on a new PC (no GPU needed: everything runs on CPU).
#   tools/setup_train_env.sh                 # -> ~/venv/f1tenth
#   tools/setup_train_env.sh /path/to/venv   # somewhere else
# Then, once per PC (generated tracks are not in git):
#   cd src/f1tenth
#   OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -m utils.track_gen 6 <workers>
#   OMP_NUM_THREADS=1 ~/venv/f1tenth/bin/python -u eval_open_space.py --make --workers <workers>
set -e
VENV="${1:-$HOME/venv/f1tenth}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 8), "Python 3.8 expected (verified with 3.8.10), got %s" % sys.version.split()[0]'
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q -r "$REPO/src/f1tenth/requirements_train.txt"
"$VENV/bin/python" - <<'PYEOF'
import torch, numpy, gymnasium, numba, yamldataclassconfig, matplotlib
print(f'[ok] torch {torch.__version__}, numpy {numpy.__version__}, gymnasium {gymnasium.__version__}, matplotlib {matplotlib.__version__}')
PYEOF
echo "[done] $VENV  (next: track generation, see the top of this script or src/f1tenth/TOOLS.md)"
