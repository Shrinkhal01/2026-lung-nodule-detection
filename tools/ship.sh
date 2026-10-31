#!/usr/bin/env bash
# MAC: test -> commit (Claude as co-author) -> push, in one command. Stops if any test fails.
#   bash tools/ship.sh "feat: what changed"
set -euo pipefail
cd "$(dirname "$0")/.."
msg="${1:?usage: bash tools/ship.sh \"type: message\"}"
# use the first Python that has the test kit (repo .venv, then ~/.venv, then whatever is on PATH)
PY=""
for c in .venv/bin/python "$HOME/.venv/bin/python" python3 python; do
  if "$c" -c "import pytest, numpy, scipy, yaml" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "No Python with the test kit found. Install it with:"
  echo "  .venv/bin/python -m pip install pytest numpy scipy pandas pyyaml scikit-image"
  exit 1
fi
echo "tests run with: $PY"
"$PY" -m pytest tests/ -q
git add -A
git status --short
git commit -m "$msg" -m "Co-authored-by: Claude <noreply@anthropic.com>"
git push
