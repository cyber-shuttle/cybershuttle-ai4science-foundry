#!/usr/bin/env bash
# Fix two upstream compatibility issues that show up under the pinned stack.
#
# 1. timm 0.3.2 imports torch._six.container_abcs (removed).
# 2. SatMAE util/pos_embed.py uses np.float (removed in NumPy 1.24+).
#    The first rewrite was a naive substring replace and turned
#    np.float32 into np.float6432 and left torch.arange on a numpy
#    dtype (job 6196065). Only rewrite the bare alias; torch.arange
#    must use torch.float64.
# 3. CustomDatasetFromImagesTemporal uses transforms.Scale, which this
#    torchvision 0.13.1 build does not ship (job 6195919).
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the satmae-temporal conda env before running this script." >&2
  exit 1
fi

# Locate timm on disk. Do not `import timm` here: that import is what fails
# before the rewrite (job 6190908).
python - <<'PY'
import pathlib
import sys
import sysconfig

root = pathlib.Path(sysconfig.get_path("purelib")) / "timm"
if not root.is_dir():
    raise SystemExit(f"timm not found under {root}")
rewritten = 0
for path in root.rglob("*.py"):
    text = path.read_text()
    if "torch._six" not in text:
        continue
    updated = (
        text.replace("from torch._six import container_abcs", "import collections.abc as container_abcs")
            .replace("from torch._six import inf", "from torch import inf")
    )
    if updated != text:
        path.write_text(updated)
        rewritten += 1
        print(f"patched {path}")
print(f"timm files rewritten: {rewritten}")
print("python", sys.version.split()[0], "site", root)
PY

CODE_DIR="${SATMAE_CODE_DIR:-${SATMAE_TEMPORAL_WORKDIR:-$HOME/scratch/satmae-temporal}/SatMAE}"
if [[ -f "$CODE_DIR/util/pos_embed.py" ]]; then
  python - "$CODE_DIR/util/pos_embed.py" <<'PY'
import pathlib
import re
import sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
updated = (
    text.replace("dtype=np.float6432", "dtype=np.float32")
        .replace("dtype=np.float6464", "dtype=np.float64")
        .replace(
            "torch.arange(embed_dim // 2, dtype=np.float64, device=pos.device)",
            "torch.arange(embed_dim // 2, dtype=torch.float64, device=pos.device)",
        )
        .replace(
            "torch.arange(embed_dim // 2, dtype=np.float, device=pos.device)",
            "torch.arange(embed_dim // 2, dtype=torch.float64, device=pos.device)",
        )
)
updated = re.sub(r"dtype=np\.float(?!\d)", "dtype=np.float64", updated)
if updated != text:
    path.write_text(updated)
    print(f"patched {path}")
else:
    print(f"no np.float rewrite needed in {path}")
PY
fi

if [[ -f "$CODE_DIR/util/datasets.py" ]]; then
  python - "$CODE_DIR/util/datasets.py" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
updated = text.replace("transforms.Scale(", "transforms.Resize(")
if updated != text:
    path.write_text(updated)
    print(f"patched {path} Scale -> Resize")
else:
    print(f"no Scale rewrite needed in {path}")
PY
fi
