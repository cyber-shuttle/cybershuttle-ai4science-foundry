#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -e .venv ]]; then
  echo 'Existing .venv found. Use it or choose a fresh checkout; this script does not overwrite it.' >&2
  exit 1
fi
python3 - <<'PY'
import sys
if sys.version_info[:2] != (3,10):
    raise SystemExit('Load Python 3.10 (baseline: 3.10.13) before setup.')
PY
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.2
mkdir -p artifacts
commit=$(python -c 'import json; print(json.load(open("config/workflow.json"))["source_commit"])')
git clone https://github.com/CederGroupHub/chgnet.git artifacts/upstream
git -C artifacts/upstream checkout --detach "$commit"
test "$(git -C artifacts/upstream rev-parse HEAD)" = "$commit"
# CPU for portability; CUDA is needed for the Makerspace GPU run.
backend=${TORCH_BACKEND:-cu121}
case "$backend" in cpu|cu121) ;; *) echo 'TORCH_BACKEND must be cpu or cu121' >&2; exit 1;; esac
python -m pip install torch==2.5.1 --index-url "https://download.pytorch.org/whl/$backend"
if [[ -n "${DEPENDENCY_LOCK:-}" ]]; then
  # Lockfile must omit torch (installed above) and chgnet (installed below).
  python -m pip install -r "$DEPENDENCY_LOCK"
else
  python -m pip install -r requirements-direct.txt
fi
python -m pip install ./artifacts/upstream
python -m pip check
python -m pip freeze > artifacts/environment-lock.txt
python - <<'PY'
import hashlib,json,pathlib
from importlib.metadata import version
root=pathlib.Path('artifacts/upstream')
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file() and (str(p.relative_to(root)) in
        ['examples/mp-18767-LiMnO2.cif','tests/test_model.py','LICENSE'] or
        ('pretrained/0.3.0/' in str(p) and ('.pth' in p.name or p.suffix == '.pt')))}
if not any('pretrained/0.3.0/' in k for k in hashes):
    raise SystemExit('No checkpoint file found: inspect the pinned source before inference.')
pathlib.Path('artifacts/manifest.json').write_text(json.dumps({'sha256':hashes,
  'packages':{k:version(k) for k in ['chgnet','torch','numpy','pymatgen']}},indent=2)+'\n')
print('Saved source/input/checkpoint hashes and environment lock.')
PY
