"""Cross-model representation hand-off: HyenaDNA consuming DNABERT-2 embeddings.

Research proof of concept for Assignment 5. Mirrors the BENDR-trained-on-
CEBRA's-output pattern: instead of HyenaDNA building its own per-nucleotide
embedding from a learned character lookup table, this feeds it per-position
embeddings produced by a *different* pretrained DNA foundation model
(DNABERT-2, 117M params, BPE tokenization + ALiBi, from PR #43 of this
course's Foundry) through a small trainable projection.

Why this pair, not HyenaDNA + Evo 2 (the original Future Direction plan):
Evo 2's smallest released checkpoint (1B params, StripedHyena2) needs
transformer-engine and an Ampere-or-newer GPU with specific FP8/BF16 kernel
support that was not reliably available in the time this assignment allows
to debug; DNABERT-2 is already a verified, reproducible Foundry recipe
(PR #43) that runs on a single A100 in under a minute, which made it
possible to actually run this live instead of only describing it. The
extraction/projection/alignment pattern implemented here (see
`resample_to_length` and `CrossModelBridge`) is identical to what an
Evo 2 version would need -- only the source model's checkpoint id and
hidden size would change.

The hard part is NOT loading either model -- it's that HyenaDNA operates
per-nucleotide (vocab size 12, one embedding per A/C/G/T/N) while DNABERT-2
tokenizes with byte-pair encoding (vocab size ~4096, each token spans a
variable number of nucleotides). The two models' "time axes" don't line
up. This script resamples DNABERT-2's token-level embedding sequence to
HyenaDNA's nucleotide-level sequence length with linear interpolation --
a coarse alignment (it assumes BPE tokens are evenly spaced across the
sequence, which they are not) that is flagged here and in the report as
the most likely source of any representational mismatch.

Usage:
    python dnabert2_hyenadna_bridge.py --fasta data/example.fasta
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.environ.get("HYENADNA_REPO", os.path.expanduser("~/scratch/hyena-dna")))
from standalone_hyenadna import CharacterTokenizer, HyenaDNAModel  # noqa: E402

DNABERT2_MODEL_ID = "zhihan1996/DNABERT-2-117M"
# Same pinned revision PR #43 (athellis/add-dnabert2-inference) verified on PACE ICE.
DNABERT2_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
DNABERT2_HIDDEN = 768

HYENADNA_MODEL = "hyenadna-tiny-1k-seqlen"
MAX_LENGTH = 1024


def inject_substring(key):
    key = re.sub(r"\.mixer", ".mixer.layer", key)
    key = re.sub(r"\.mlp", ".mlp.layer", key)
    return key


def load_hyenadna(checkpoint_dir, model_name, device):
    model_dir = os.path.join(checkpoint_dir, model_name)
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "weights.ckpt")
    if not (os.path.isfile(config_path) and os.path.isfile(weights_path)):
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://huggingface.co/LongSafari/{model_name}"],
            cwd=checkpoint_dir, check=True,
        )
    config = json.load(open(config_path))
    model = HyenaDNAModel(**config, use_head=False, n_classes=2)
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


def load_dnabert2(device):
    from transformers import AutoModel, AutoTokenizer
    from transformers.models.bert.configuration_bert import BertConfig

    tokenizer = AutoTokenizer.from_pretrained(
        DNABERT2_MODEL_ID, revision=DNABERT2_REVISION, trust_remote_code=True
    )
    config = BertConfig.from_pretrained(DNABERT2_MODEL_ID, revision=DNABERT2_REVISION)
    model = AutoModel.from_pretrained(
        DNABERT2_MODEL_ID, revision=DNABERT2_REVISION, trust_remote_code=True, config=config
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def read_first_fasta_record(path):
    lines, in_record = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if in_record:
                    break
                in_record = True
                continue
            if in_record:
                lines.append(line)
    return "".join(lines)


def resample_to_length(seq_embeddings, target_len):
    """[1, L_src, H] -> [1, target_len, H] by linear interpolation along the
    sequence axis. DNABERT-2's BPE tokens don't map 1:1 to nucleotides (each
    token spans a handful of bases, and spans vary), so this treats the
    token embedding sequence as a continuous signal and resamples it --
    exact per-nucleotide alignment would need the tokenizer's char-to-token
    offsets, which this tokenizer (loaded via trust_remote_code) does not
    expose through the fast-tokenizer offset_mapping API.
    """
    x = seq_embeddings.permute(0, 2, 1)  # [1, H, L_src]
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)  # [1, target_len, H]


class CrossModelBridge(nn.Module):
    """Projects DNABERT-2 hidden states into HyenaDNA's embedding space and
    runs them through HyenaDNA's own Hyena-operator layers -- bypassing
    HyenaDNA's native nn.Embedding(vocab_size=12, ...) lookup entirely.
    No position embedding needs adding back: HyenaDNA is built with
    max_position_embeddings=0 (position information comes implicitly from
    the Hyena convolution filters), so GPT2Embeddings never instantiates one.
    """

    def __init__(self, hyenadna_backbone, d_model):
        super().__init__()
        self.backbone = hyenadna_backbone
        self.project = nn.Linear(DNABERT2_HIDDEN, d_model)

    def forward(self, dnabert2_hidden_states, target_len):
        resampled = resample_to_length(dnabert2_hidden_states, target_len)
        hidden_states = self.project(resampled)
        # No explicit position embedding to add back: HyenaDNA's own
        # GPT2Embeddings is constructed with max_position_embeddings=0 (Hyena
        # gets its position information implicitly from the convolution
        # filters, not a learned position table), so it never even
        # instantiates a `position_embeddings` submodule -- confirmed live,
        # this was the first thing tried and it doesn't exist on this model.

        residual = None
        for layer in self.backbone.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = self.backbone.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        return self.backbone.ln_f(residual.to(dtype=self.backbone.ln_f.weight.dtype))


def native_hyenadna_embedding(model, tokenizer, sequence, device):
    tok_seq = torch.LongTensor(tokenizer(sequence)["input_ids"]).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model(tok_seq), tok_seq.shape[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--checkpoint-dir", default=os.path.expanduser("~/scratch/hyena-dna/checkpoints"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sequence = read_first_fasta_record(args.fasta)
    print(f"input length: {len(sequence)} nt", flush=True)

    print("loading HyenaDNA...", flush=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    hyenadna_model, hyenadna_config = load_hyenadna(args.checkpoint_dir, HYENADNA_MODEL, device)
    hyenadna_model.to(device)
    hyenadna_model.eval()
    d_model = hyenadna_config["d_model"]
    hyenadna_tokenizer = CharacterTokenizer(characters=["A", "C", "G", "T", "N"], model_max_length=MAX_LENGTH)

    print("loading DNABERT-2...", flush=True)
    dnabert2_model, dnabert2_tokenizer = load_dnabert2(device)

    bridge = CrossModelBridge(hyenadna_model.backbone, d_model).to(device)

    def run(seq):
        dnabert2_ids = dnabert2_tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            dnabert2_hidden = dnabert2_model(dnabert2_ids)[0]  # [1, L_bpe, 768]
        target_len = len(hyenadna_tokenizer(seq)["input_ids"])  # match native HyenaDNA token count
        with torch.no_grad():
            bridged = bridge(dnabert2_hidden, target_len)
        return bridged, dnabert2_hidden.shape[1], target_len

    bridged_out, l_bpe, l_nt = run(sequence)
    print(f"[bridge] DNABERT-2 tokens: {l_bpe}  ->  resampled to HyenaDNA length: {l_nt}", flush=True)
    print(f"[bridge] output shape: {tuple(bridged_out.shape)}  "
          f"nan={torch.isnan(bridged_out).any().item()}  inf={torch.isinf(bridged_out).any().item()}", flush=True)

    bridged_out_2, _, _ = run(sequence)
    determinism = (bridged_out - bridged_out_2).norm().item()
    print(f"[determinism] L2 distance across repeated runs: {determinism:.2e}", flush=True)

    shuffled = "".join([sequence[i] for i in torch.randperm(len(sequence)).tolist()])
    bridged_shuffled, _, _ = run(shuffled)
    sensitivity = (bridged_out - bridged_shuffled).norm().item()
    print(f"[sensitivity] L2 distance, real vs shuffled sequence: {sensitivity:.2f}", flush=True)

    native_out, native_len = native_hyenadna_embedding(hyenadna_model, hyenadna_tokenizer, sequence, device)
    comparison = (bridged_out - native_out).norm().item()
    print(f"[comparison] L2 distance, DNABERT-2-bridged vs HyenaDNA-native embedding "
          f"(same sequence): {comparison:.2f}  (native norm scale: {native_out.norm().item():.2f}, "
          f"bridged norm scale: {bridged_out.norm().item():.2f})", flush=True)

    ok = (not torch.isnan(bridged_out).any() and not torch.isinf(bridged_out).any()
          and determinism < 1e-3 and sensitivity > 1e-3)
    print("BRIDGE VALIDATION PASSED" if ok else "BRIDGE VALIDATION FAILED", flush=True)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save({
            "bridged_embedding": bridged_out.cpu(),
            "native_embedding": native_out.cpu(),
            "dnabert2_token_count": l_bpe,
            "hyenadna_token_count": l_nt,
        }, args.output)
        print(f"saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
