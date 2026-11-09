#!/bin/bash
# Downloads the input NAIP tile and Clay's published embeddings for it.
# Safe to rerun: files already present with the right checksum are skipped.
#
# Normally run by: sbatch scripts/clay_download.sbatch
# Can also be run from the Clay model folder after the env is built:
#   bash scripts/clay_download_data.sh
#
# Inputs (pinned in scripts/clay_common.sh):
#   NAIP tile  m_3211002_se_12_1_20150530.tif  from Microsoft Planetary Computer
#              (STAC item az_m_3211002_se_12_1_20150530_20151022; free, no account)
#   Reference  m_3211002_se_12_1_20150530.parquet  from source.coop
#              clay/clay-v1-5-naip-2 (Clay v1.5 NAIP embeddings, rev2, CC-BY-4.0)
# Writes:
#   $CLAY_WORKDIR/data/<both files>
#   benchmarks/outputs/naip_item.json    (STAC item, unsigned)
#   benchmarks/outputs/checksums.sha256  (adds the checksums)

set -euo pipefail
# shellcheck source=scripts/clay_common.sh
source scripts/clay_common.sh
clay_require_repo_root
clay_make_dirs
clay_activate

echo "== NAIP tile: STAC item $NAIP_ITEM_ID"
SIGNED_URL_FILE="$TMPDIR/naip_signed_url.$$"
export NAIP_STAC_API NAIP_ITEM_ID NAIP_TILE_NAME SIGNED_URL_FILE CLAY_EVIDENCE
python - <<'PY'
import json
import os
from pathlib import Path

import planetary_computer
import pystac_client

catalog = pystac_client.Client.open(os.environ["NAIP_STAC_API"])
items = list(catalog.search(collections=["naip"], ids=[os.environ["NAIP_ITEM_ID"]]).items())
if len(items) != 1:
    raise SystemExit(f"ERROR: expected 1 STAC item {os.environ['NAIP_ITEM_ID']}, found {len(items)}")
item = items[0]
href = item.assets["image"].href
props = item.properties
# The STAC projection extension v1 calls it proj:epsg; newer pystac
# versions migrate items to v2, which uses proj:code ("EPSG:26912").
epsg = props.get("proj:epsg") or int(str(props.get("proj:code", "")).replace("EPSG:", "") or 0)
found = {"gsd": props.get("gsd"), "epsg": epsg, "shape": props.get("proj:shape")}
print(f"   href      {href}")
print(f"   datetime  {props['datetime']}  {found}")
expected = {"gsd": 1.0, "epsg": 26912, "shape": [7570, 6490]}
if found != expected:
    raise SystemExit(f"ERROR: item properties {found}, expected {expected}")
if not href.endswith("/" + os.environ["NAIP_TILE_NAME"]):
    raise SystemExit(f"ERROR: unexpected asset file {href}")
# Keep the unsigned item as provenance; the signed URL holds a short-lived token.
out = Path(os.environ["CLAY_EVIDENCE"], "outputs", "naip_item.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(item.to_dict(), indent=2) + "\n")
signed = planetary_computer.sign(href)
fd = os.open(os.environ["SIGNED_URL_FILE"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, signed.encode())
os.close(fd)
PY
SIGNED_URL=$(cat "$SIGNED_URL_FILE")
rm -f "$SIGNED_URL_FILE"
clay_fetch "$SIGNED_URL" "$CLAY_DATA/$NAIP_TILE_NAME" "$NAIP_TILE_SHA256" "$NAIP_TILE_BYTES"

echo "== Reference embeddings (source.coop clay/clay-v1-5-naip-2)"
clay_fetch "$REF_URL" "$CLAY_DATA/$REF_NAME" "$REF_SHA256"

echo "== DATA READY"
