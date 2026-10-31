#!/usr/bin/env bash
# MAC: copy the server's small results into this repo (no checkpoints). Claude reads them from here.
#   one-time:  echo 'export LUNG_SERVER=user@server-address' >> ~/.zshrc && source ~/.zshrc
#   then:      bash tools/pull_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LUNG_SERVER:?set LUNG_SERVER first (see the top of this file)}"
rsync -a --exclude '*.pth' "$LUNG_SERVER:/scratch/mtech_shrinkhal/lungdet_results/" results/
echo "synced results/ from $LUNG_SERVER"
