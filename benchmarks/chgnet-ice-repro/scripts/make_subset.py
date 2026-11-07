#!/usr/bin/env python3
"""
Extract a small, order-independent deterministic MPtrj sample.

One eligible frame per material. Ranking is SHA256(seed, material_id).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
import zipfile
from contextlib import contextmanager
from pathlib import Path


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


@contextmanager
def open_dataset(path: Path):
    with path.open("rb") as handle:
        header = handle.read(4)
    if header.startswith(b"\x1f\x8b"):
        with gzip.open(path, "rb") as handle:
            yield handle
    elif header.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(path) as archive:
            candidates = sorted(f for f in archive.namelist() if f.endswith(".json"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one JSON inside ZIP, found {candidates}")
            with archive.open(candidates[0]) as handle:
                yield handle
    else:
        with path.open("rb") as handle:
            yield handle


def rank(seed: int, material_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{material_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def choose_frame(frames: dict, min_atoms: int, max_atoms: int) -> tuple[str, dict] | None:
    for frame_id in sorted(frames):
        item = frames[frame_id]
        if not isinstance(item, dict) or not isinstance(item.get("structure"), dict):
            continue
        n = len(item["structure"].get("sites", []))
        if not min_atoms <= n <= max_atoms:
            continue
        try:
            e = float(item["energy_per_atom"])
            if not math.isfinite(e):
                continue
            if len(item["force"]) != n or any(len(row) != 3 for row in item["force"]):
                continue
            if len(item["stress"]) != 3 or any(len(row) != 3 for row in item["stress"]):
                continue
        except (TypeError, KeyError, ValueError):
            continue
        return frame_id, item
    return None


def select_sample(dataset_path: Path, count: int, seed: int, min_atoms: int, max_atoms: int) -> tuple[list[dict], int, int]:
    import ijson  # optional until a real dataset is processed

    # Max-heap in effect: negate rank so the worst chosen item is at heap[0].
    selected: list[tuple[int, str, dict]] = []
    materials_seen = 0
    materials_eligible = 0
    with open_dataset(dataset_path) as source:
        for material_id, frames in ijson.kvitems(source, "", use_float=True):
            materials_seen += 1
            if not isinstance(frames, dict):
                continue
            choice = choose_frame(frames, min_atoms, max_atoms)
            if choice is None:
                continue
            materials_eligible += 1
            frame_id, item = choice
            score = rank(seed, material_id)
            candidate = (-score, str(material_id), {
                "material_id": str(material_id),
                "frame_id": str(frame_id),
                "structure": item["structure"],
                "reference": {
                    "energy_per_atom": item["energy_per_atom"],
                    "force": item["force"],
                    "stress": item["stress"],
                    "magmom": item.get("magmom"),
                },
            })
            if len(selected) < count:
                heapq.heappush(selected, candidate)
            elif candidate > selected[0]:
                heapq.heapreplace(selected, candidate)
    records = [entry[2] for entry in sorted(selected, key=lambda x: (-x[0], x[1]))]
    return records, materials_seen, materials_eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("inputs/mptrj_subset.json"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-atoms", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=64)
    args = parser.parse_args()
    if args.count < 1 or args.min_atoms < 1 or args.max_atoms < args.min_atoms:
        parser.error("Invalid sampling limits")
    records, seen, eligible = select_sample(args.input, args.count, args.seed, args.min_atoms, args.max_atoms)
    if len(records) != args.count:
        raise RuntimeError(f"Only {len(records)} eligible materials found; requested {args.count}")
    manifest_path = args.input.with_name(args.input.name + ".manifest.json")
    upstream = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    result = {
        "provenance": {
            "source": "Materials Project Trajectory (MPtrj), Figshare article 23713842 version 2",
            "source_url": "https://doi.org/10.6084/m9.figshare.23713842.v2",
            "source_file_sha256": sha256_file(args.input),
            "source_download_manifest": upstream,
            "selection": "one eligible frame per material, lowest SHA256(seed:material_id) ranks",
            "seed": args.seed,
            "requested_structures": args.count,
            "min_atoms": args.min_atoms,
            "max_atoms": args.max_atoms,
            "material_ids_seen": seen,
            "material_ids_eligible": eligible,
            "warning": "Deterministic full-MPtrj subset, NOT verified original held-out test partition",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {len(records)} structures from {seen} materials ({eligible} eligible) -> {args.output}")
    print(f"Subset SHA256: {sha256_file(args.output)}")
    print("WARNING: This is not the published held-out test split.")


if __name__ == "__main__":
    main()
