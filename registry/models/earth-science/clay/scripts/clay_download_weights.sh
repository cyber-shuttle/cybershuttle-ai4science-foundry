#!/bin/bash
# Downloads the Clay v1.5 checkpoint and pre-fetches the DINOv2 teacher that
# Clay builds at model-load time. Safe to rerun: files that are already
# present with the right checksum are skipped.
#
# Normally run by: sbatch scripts/clay_download.sbatch
# Can also be run from the Clay model folder after the env is built:
#   bash scripts/clay_download_weights.sh
#
# Writes:
#   $CLAY_WORKDIR/weights/clay-v1.5.ckpt     (5,158,764,629 bytes)
#   $HF_HOME/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/  (~1.2 GB)
#   benchmarks/outputs/checksums.sha256  (adds both checksums)

set -euo pipefail
# shellcheck source=scripts/clay_common.sh
source scripts/clay_common.sh
clay_require_repo_root
clay_make_dirs

echo "== Clay v1.5 checkpoint (made-with-clay/Clay @ $CLAY_HF_REVISION)"
clay_fetch "$CLAY_CKPT_URL" "$CLAY_WEIGHTS/clay-v1.5.ckpt" "$CLAY_CKPT_SHA256" "$CLAY_CKPT_BYTES"

echo "== DINOv2 teacher ($CLAY_TEACHER)"
clay_activate
export CLAY_WORKDIR CLAY_TEACHER CLAY_TEACHER_HF_REPO CLAY_TEACHER_HF_REVISION CLAY_TEACHER_SHA256
python - <<'PY'
import hashlib
import os
from pathlib import Path

import timm
from huggingface_hub import try_to_load_from_cache

name = os.environ["CLAY_TEACHER"]
repo = os.environ["CLAY_TEACHER_HF_REPO"]
# Same call Clay makes inside ClayMAE.__init__; downloads into $HF_HOME once.
model = timm.create_model(name, pretrained=True, num_classes=0)
del model  # free ~1.2 GB before hashing (job 6188620 peaked at 7.9 of 8 GB)

path = try_to_load_from_cache(repo, "model.safetensors")
if not isinstance(path, str):
    raise SystemExit(f"ERROR: teacher weights not found in the cache for {repo}")
revision = Path(path).parent.name  # .../snapshots/<revision>/model.safetensors
h = hashlib.sha256()
with open(path, "rb") as f:  # hash in 1 MB blocks instead of reading 1.2 GB at once
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
digest = h.hexdigest()
print(f"   cached at {path}")
print(f"   hf revision {revision}")
if digest != os.environ["CLAY_TEACHER_SHA256"]:
    raise SystemExit(f"ERROR: teacher sha256 {digest} != expected {os.environ['CLAY_TEACHER_SHA256']}")
if revision != os.environ["CLAY_TEACHER_HF_REVISION"]:
    # Same file content under a newer repo revision: note it, do not fail.
    print(f"   NOTE: repo revision changed from {os.environ['CLAY_TEACHER_HF_REVISION']}; file checksum still matches")
# Path relative to $CLAY_WORKDIR, so `sha256sum -c` works on the whole record.
rel = os.path.relpath(os.path.realpath(path), os.path.realpath(os.environ["CLAY_WORKDIR"]))
Path(os.environ["TMPDIR"], "teacher.sha256").write_text(f"{digest}  {rel}\n")
PY
clay_record_checksum "$(cat "$TMPDIR/teacher.sha256")"
echo "   sha256 $(cat "$TMPDIR/teacher.sha256")"

echo "== Offline check: GPU jobs read the teacher from the cache with HF_HUB_OFFLINE=1"
HF_HUB_OFFLINE=1 python -c "import timm; timm.create_model('$CLAY_TEACHER', pretrained=True, num_classes=0); print('   offline load OK')"

echo "== WEIGHTS READY"
