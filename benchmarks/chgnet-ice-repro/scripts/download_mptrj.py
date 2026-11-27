#!/usr/bin/env python3
"""Retrieve the archival MPtrj Figshare article (version 2), with checksums.

This may download ~11.35 GB. Do this once in scratch/shared storage
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ARTICLE_ID = 23713842
ARTICLE_VERSION = 2
METADATA_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{ARTICLE_VERSION}"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retrieve_metadata() -> dict:
    req = Request(METADATA_URL, headers={"User-Agent": "chgnet-ice-repro/1.0"})
    with urlopen(req, timeout=60) as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("raw/mptrj_source"))
    parser.add_argument("--force", action="store_true", help="Download again, replacing an existing file")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = retrieve_metadata()
    files = meta.get("files", [])
    if not files:
        raise RuntimeError(f"No downloadable files in {METADATA_URL}")

    candidates = [f for f in files if f.get("name", "").lower().endswith((".json", ".json.gz", ".zip"))]
    if not candidates:
        raise RuntimeError(f"Unexpected Figshare files: {[f.get('name') for f in files]}")

    selected = max(candidates, key=lambda f: f.get("size", 0))
    if args.output.exists() and not args.force:
        if args.output.stat().st_size != selected["size"]:
            raise RuntimeError("Existing file has wrong size; remove it or pass --force")
        print(f"Reusing existing file: {args.output}")
    else:
        temp_path = args.output.with_name(args.output.name + ".partial")
        req = Request(selected["download_url"], headers={"User-Agent": "chgnet-ice-repro/1.0"})
        try:
            with urlopen(req, timeout=120) as source, temp_path.open("wb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
            if temp_path.stat().st_size != selected["size"]:
                raise RuntimeError("Incomplete download: byte count differs from Figshare metadata")
            temp_path.replace(args.output)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    payload = {
        "article_url": f"https://figshare.com/articles/dataset/23713842/{ARTICLE_VERSION}",
        "api_metadata_url": METADATA_URL,
        "article_id": ARTICLE_ID,
        "article_version": ARTICLE_VERSION,
        "figshare_file_id": selected.get("id"),
        "original_filename": selected["name"],
        "original_download_url": selected["download_url"],
        "expected_bytes": selected["size"],
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "local_file": str(args.output),
        "local_sha256": checksum(args.output),
    }
    manifest = args.output.with_name(args.output.name + ".manifest.json")
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Manifest written to {manifest}")


if __name__ == "__main__":
    main()
