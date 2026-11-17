#!/usr/bin/env bash
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu
source "$suite/environment.sh"
cd "$suite/Cactus"
python - <<'PY'
from pathlib import Path
import json,socket
root=Path.cwd().parent
assert socket.gethostname().startswith('wm2-login')
assert json.loads((root/'FETCH_RESULT.json').read_text())['returncode']==0
assert (root/'DEPENDENCY_ARCHIVES.json').is_file()
PY
./simfactory/bin/sim build wm2cuda --machine=generic \
  --optionlist="$suite/wm2-cuda.cfg" --thornlist="$suite/et-gpu.th" --jobs=2
