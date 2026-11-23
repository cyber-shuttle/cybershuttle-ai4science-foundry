"""Load a pretrained HyenaDNA checkpoint and embed a DNA sequence.

This intentionally does not import HazyResearch/hyena-dna's own huggingface.py:
that module calls inference_single() at import time with no `if __name__ ==
"__main__"` guard, so `from huggingface import HyenaDNAPreTrainedModel` runs a
full hardcoded inference example as a side effect before your own code ever
gets a chance to run. The checkpoint-loading logic it contains is small, so
it's reimplemented here against standalone_hyenadna.py directly instead.

Usage:
    python hyenadna_inference.py --model hyenadna-tiny-1k-seqlen --sequence ACTG...
    python hyenadna_inference.py --model hyenadna-tiny-1k-seqlen --fasta input.fasta
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import torch

# standalone_hyenadna.py lives in the hyena-dna checkout, not this repo.
sys.path.insert(0, os.environ.get("HYENADNA_REPO", os.path.expanduser("~/scratch/hyena-dna")))
from standalone_hyenadna import CharacterTokenizer, HyenaDNAModel  # noqa: E402

MAX_LENGTHS = {
    "hyenadna-tiny-1k-seqlen": 1024,
    "hyenadna-tiny-1k-seqlen-d256": 1024,
    "hyenadna-small-32k-seqlen": 32768,
    "hyenadna-medium-160k-seqlen": 160000,
    "hyenadna-medium-450k-seqlen": 450000,
    "hyenadna-large-1m-seqlen": 1_000_000,
}


def inject_substring(key):
    # Checkpoints trained with gradient checkpointing nest mixer/mlp modules
    # one level deeper; the state dict keys differ by this exact substring.
    key = re.sub(r"\.mixer", ".mixer.layer", key)
    key = re.sub(r"\.mlp", ".mlp.layer", key)
    return key


def load_pretrained(checkpoint_dir, model_name, device):
    model_dir = os.path.join(checkpoint_dir, model_name)
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "weights.ckpt")
    # A directory can exist but be incomplete (e.g. an interrupted clone), so
    # check for the files we actually need rather than just the directory.
    if not (os.path.isfile(config_path) and os.path.isfile(weights_path)):
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)  # git clone refuses a non-empty target dir
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://huggingface.co/LongSafari/{model_name}"],
            cwd=checkpoint_dir, check=True,
        )

    config = json.load(open(config_path))
    model = HyenaDNAModel(**config, use_head=False, n_classes=2)

    # weights.ckpt is a full PyTorch Lightning checkpoint, not a bare state
    # dict — the actual weights are nested under ckpt["state_dict"], and its
    # keys carry a "model." prefix that the standalone model doesn't use.
    ckpt = torch.load(weights_path, map_location=device)
    checkpointing = config.get("checkpoint_mixer", False) is True

    state_dict = model.state_dict()
    for key in state_dict:
        if "backbone" in key:
            loaded_key = "model." + key
            if checkpointing:
                loaded_key = inject_substring(loaded_key)
            state_dict[key] = ckpt["state_dict"][loaded_key]
    model.load_state_dict(state_dict)
    return model, config


def read_first_fasta_record(path):
    lines = []
    in_record = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if in_record:
                    break  # second header reached; stop before the next record
                in_record = True
                continue
            if in_record:
                lines.append(line)
    return "".join(lines)


def embed(model, tokenizer, sequence, device):
    tok_seq = tokenizer(sequence)["input_ids"]
    tok_seq = torch.LongTensor(tok_seq).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model(tok_seq), tok_seq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hyenadna-tiny-1k-seqlen", choices=MAX_LENGTHS.keys())
    parser.add_argument("--checkpoint-dir", default=os.path.expanduser("~/scratch/hyena-dna/checkpoints"))
    parser.add_argument("--sequence", default=None, help="DNA sequence (A/C/G/T/N) to embed")
    parser.add_argument("--fasta", default=None, help="FASTA file; embeds the first record")
    parser.add_argument("--output", default=None, help="Path to save embeddings as a .pt tensor")
    args = parser.parse_args()

    if args.sequence is None and args.fasta is None:
        parser.error("pass --sequence or --fasta")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model, config = load_pretrained(args.checkpoint_dir, args.model, device)
    model.to(device)
    model.eval()

    max_length = MAX_LENGTHS[args.model]
    tokenizer = CharacterTokenizer(characters=["A", "C", "G", "T", "N"], model_max_length=max_length)

    if args.sequence:
        sequence = args.sequence
    else:
        sequence = read_first_fasta_record(args.fasta)

    if len(sequence) > max_length:
        print(f"warning: sequence length {len(sequence)} exceeds {args.model}'s "
              f"trained context {max_length}; truncating", file=sys.stderr)
        sequence = sequence[:max_length]

    embeddings, tok_seq = embed(model, tokenizer, sequence, device)

    print(f"model: {args.model} (d_model={config['d_model']}, n_layer={config['n_layer']})")
    print(f"device: {device}")
    print(f"input length: {len(sequence)} nt")
    print(f"token shape: {tuple(tok_seq.shape)}")
    print(f"embedding shape: {tuple(embeddings.shape)}")
    print(f"embedding mean/std: {embeddings.float().mean().item():.4f} / {embeddings.float().std().item():.4f}")

    if args.output:
        torch.save(embeddings.cpu(), args.output)
        print(f"saved embeddings to {args.output}")


if __name__ == "__main__":
    main()
