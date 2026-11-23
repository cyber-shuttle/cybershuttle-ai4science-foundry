#!/usr/bin/env python3
"""Fetch the inputs for the proof-of-concept targets.

For each target in poc/targets.tsv this writes:
  poc/inputs/fasta/<id>.fasta          query sequence (header is the target id)
  poc/inputs/reference/<pdb>.pdb       experimental structure from RCSB
  poc/inputs/msa/<id>/*.a3m.gz         MSA from the public ColabFold MMseqs2 server
  poc/inputs/MANIFEST.tsv              lengths, MSA depth, sha-256 of every file

The cluster has no sequence databases, so MSAs come from the ColabFold
server (https://api.colabfold.com). Run this once on a login node, which has
internet access, then commit poc/inputs so later runs do not depend on the
server. Uses only the Python standard library.

usage: python poc/fetch_inputs.py [--no-msa] [--only ID ...]
"""

import argparse
import datetime
import gzip
import hashlib
import io
import json
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent
RECIPE_DIR = POC_DIR.parent
INPUTS = POC_DIR / "inputs"
COLABFOLD_URL = "https://api.colabfold.com"
USER_AGENT = "openfold-poc-recipe/1.0"
# Keep at most this many sequences per MSA file. AlphaFold's own pipeline caps
# uniref90 hits at 10000; OpenFold only uses 512 + 5120 rows at inference.
MAX_MSA_SEQS = 10000

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def http(url, data=None, tries=4):
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == tries - 1:
                raise
            print(f"  retry after error: {err}")
            time.sleep(5 * (attempt + 1))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_targets():
    targets = []
    for line in (POC_DIR / "targets.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tid, pdb, chain, extra, note = (line.split("\t") + ["", "", "", "-", ""])[:5]
        refs = [(pdb, chain)]
        if extra.strip() not in ("", "-"):
            refs += [tuple(x.split(":")) for x in extra.split(",")]
        targets.append({"id": tid, "pdb": pdb, "chain": chain, "refs": refs, "note": note})
    return targets


def reference_pdb(pdb):
    """Return the PDB text, using the committed copy for 6KWC."""
    out = INPUTS / "reference" / f"{pdb.lower()}.pdb"
    if out.is_file():
        return out.read_bytes()
    local = RECIPE_DIR / "inputs" / "reference" / f"{pdb.lower()}.pdb"
    data = local.read_bytes() if local.is_file() else http(f"https://files.rcsb.org/download/{pdb.upper()}.pdb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return data


def seqres_sequence(pdb_text, chain):
    """Full deposited sequence of one chain from the SEQRES records."""
    residues = []
    for line in pdb_text.decode(errors="replace").splitlines():
        if line.startswith("SEQRES") and line[11] == chain:
            residues += line[19:].split()
    if not residues:
        raise ValueError(f"no SEQRES records for chain {chain}")
    unknown = sorted({r for r in residues if r not in THREE_TO_ONE})
    if unknown:
        print(f"  note: non-standard residues {unknown} written as X")
    return "".join(THREE_TO_ONE.get(r, "X") for r in residues)


def query_sequence(target, pdb_text):
    if target["pdb"].upper() == "6KWC":
        # Use the exact assignment 2 input so the OpenFold example MSA lines up.
        lines = (RECIPE_DIR / "inputs" / "6KWC_1.fasta").read_text().split()
        return "".join(lines[1:])
    return seqres_sequence(pdb_text, target["chain"])


def colabfold_msa(seq):
    """Submit one sequence to the ColabFold MMseqs2 server, return {name: a3m text}."""
    ticket = json.loads(http(f"{COLABFOLD_URL}/ticket/msa", {"q": f">101\n{seq}\n", "mode": "env"}))
    waited = 0
    while ticket.get("status") in ("PENDING", "RUNNING", "UNKNOWN", "RATELIMIT"):
        if waited > 1800:
            raise TimeoutError("ColabFold server did not finish in 30 minutes")
        time.sleep(10)
        waited += 10
        if ticket.get("status") in ("UNKNOWN", "RATELIMIT"):
            ticket = json.loads(http(f"{COLABFOLD_URL}/ticket/msa", {"q": f">101\n{seq}\n", "mode": "env"}))
        else:
            ticket = json.loads(http(f"{COLABFOLD_URL}/ticket/{ticket['id']}"))
        print(f"  colabfold status: {ticket.get('status')} ({waited} s)")
    if ticket.get("status") != "COMPLETE":
        raise RuntimeError(f"ColabFold server returned status {ticket.get('status')}")
    archive = http(f"{COLABFOLD_URL}/result/download/{ticket['id']}")
    msas = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".a3m"):
                text = tar.extractfile(member).read().decode().replace("\x00", "")
                msas[Path(member.name).name] = text
    if not msas:
        raise RuntimeError("ColabFold archive had no .a3m files")
    return msas


def clean_a3m(text, seq, tag):
    """Make the query the first record, named after the target, and cap the depth."""
    records, name, chunks = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name, chunks = line[1:].strip(), []
        elif line.strip() and name is not None:
            chunks.append(line.strip())
    if name is not None:
        records.append((name, "".join(chunks)))
    if not records:
        return "", 0, 0
    aligned = "".join(c for c in records[0][1] if not c.islower())
    if aligned != seq:
        raise ValueError("first MSA record is not the query sequence")
    raw = len(records)
    kept = [(tag, seq)] + records[1:MAX_MSA_SEQS]
    return "".join(f">{n}\n{s}\n" for n, s in kept), raw, len(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-msa", action="store_true", help="skip the ColabFold server (single-sequence runs only)")
    parser.add_argument("--only", nargs="*", help="target ids to fetch (default: all)")
    args = parser.parse_args()

    manifest_rows = []
    failed = []
    for target in read_targets():
        tid = target["id"]
        if args.only and tid not in args.only:
            continue
        print(f"== {tid} ({target['note']})")
        try:
            for pdb, _ in target["refs"]:
                data = reference_pdb(pdb)
                print(f"  reference {pdb}: {len(data)} bytes")
            seq = query_sequence(target, reference_pdb(target["pdb"]))
        except Exception as err:  # keep going with the other targets
            print(f"  error: {err}")
            failed.append(tid)
            continue
        fasta = INPUTS / "fasta" / f"{tid}.fasta"
        fasta.parent.mkdir(parents=True, exist_ok=True)
        fasta.write_text(f">{tid}\n{seq}\n")
        print(f"  sequence: {len(seq)} residues")

        msa_note = "skipped (--no-msa)"
        msa_dir = INPUTS / "msa" / tid
        if not args.no_msa:
            try:
                msas = colabfold_msa(seq)
                msa_dir.mkdir(parents=True, exist_ok=True)
                parts = []
                for name, text in sorted(msas.items()):
                    try:
                        cleaned, raw, kept = clean_a3m(text, seq, tid)
                    except ValueError as err:
                        print(f"  skipping {name}: {err}")
                        continue
                    if kept < 2:
                        continue
                    with gzip.open(msa_dir / f"{name}.gz", "wt") as handle:
                        handle.write(cleaned)
                    parts.append(f"{name}:{raw}->{kept}")
                    print(f"  msa {name}: {raw} sequences, kept {kept}")
                msa_note = ";".join(parts) or "no hits"
            except Exception as err:
                msa_note = f"failed: {err}"
                print(f"  msa error: {err} (this target will run single-sequence only)")
        manifest_rows.append([
            tid, target["pdb"], target["chain"], str(len(seq)), sha256(seq.encode()),
            ",".join(f"{p}:{c}" for p, c in target["refs"]), msa_note,
        ])

    if manifest_rows:
        manifest = INPUTS / "MANIFEST.tsv"
        old = {}
        if manifest.is_file():
            for line in manifest.read_text().splitlines()[2:]:
                old[line.split("\t")[0]] = line
        for row in manifest_rows:
            old[row[0]] = "\t".join(row)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header = "id\tpdb\tchain\tlength\tsequence_sha256\treferences\tmsa (file:raw->kept)"
        manifest.write_text(f"# fetched {stamp} from files.rcsb.org and {COLABFOLD_URL} (mode env)\n{header}\n"
                            + "\n".join(old.values()) + "\n")
        lines = [f"{sha256(p.read_bytes())}  {p.relative_to(POC_DIR)}"
                 for p in sorted(INPUTS.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
        (INPUTS / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"fetch finished: {len(manifest_rows)} targets ok, failed: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
