"""Smoke test + validation check for the HyenaDNA inference recipe.

Smoke test: inference runs, produces the expected tensor shape, and contains
no NaN/Inf values.

Validation check: embeddings should actually reflect the input sequence, not
just be "some numbers that didn't crash". Running the same sequence twice
must give bit-identical output (eval mode, no dropout); two different random
sequences must give measurably different per-position embeddings. A broken
checkpoint load (wrong keys, wrong prefix, right shape but wrong values) or a
model that silently ignores its input would still pass the smoke test but
fail this check.

Note: cosine similarity on mean-pooled embeddings was tried first and
rejected — for this checkpoint it stayed above 0.99 even between two
unrelated random sequences, so it couldn't actually discriminate anything.
Raw per-position L2 distance on the un-pooled embeddings turned out to be a
far more reliable signal (measured ~56 for two unrelated 200nt sequences,
against a ~138 embedding norm, vs. exactly 0 for the same sequence twice).

Pretraining-objective check: HyenaDNA is trained to predict the next
nucleotide, so a real sequence (a human mRNA, HBB) should get a higher mean
next-nucleotide log-likelihood than a random shuffle of the same nucleotides.
Same composition, different order, so any gap reflects learned sequence
structure rather than base frequencies. The LM head is the tied embedding
matrix (logits = hidden_states @ word_embeddings.weight.T), as in upstream
training. A missing gap points at a loading or tokenization bug.

Usage:
    python hyenadna_validate.py --model hyenadna-tiny-1k-seqlen [--fasta path]
"""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

DEFAULT_FASTA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "registry", "models", "genomics", "hyenadna", "data", "example.fasta",
)

sys.path.insert(0, os.path.dirname(__file__))
from hyenadna_inference import MAX_LENGTHS, embed, load_pretrained  # noqa: E402

sys.path.insert(0, os.environ.get("HYENADNA_REPO", os.path.expanduser("~/scratch/hyena-dna")))
from standalone_hyenadna import CharacterTokenizer  # noqa: E402


def random_sequence(length, seed):
    return "".join(random.Random(seed).choices("ACGT", k=length))


def read_fasta(path):
    with open(path) as f:
        return "".join(l.strip() for l in f if not l.startswith(">")).upper()


def mean_next_nucleotide_loglik(model, sequence, max_length, device):
    # Special tokens off, so every position is a real nucleotide with a real next token.
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"], model_max_length=max_length + 2,
        add_special_tokens=False, padding_side="left",
    )
    ids = torch.LongTensor(tokenizer(sequence)["input_ids"]).unsqueeze(0).to(device)
    with torch.inference_mode():
        hidden = model(ids)
        logits = hidden @ model.backbone.embeddings.word_embeddings.weight.T
        log_probs = F.log_softmax(logits.float(), dim=-1)[:, :-1]
        return log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hyenadna-tiny-1k-seqlen", choices=MAX_LENGTHS.keys())
    parser.add_argument("--fasta", default=DEFAULT_FASTA, help="real sequence for the pretraining-objective check")
    parser.add_argument("--checkpoint-dir", default=os.path.expanduser("~/scratch/hyena-dna/checkpoints"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model, config = load_pretrained(args.checkpoint_dir, args.model, device)
    model.to(device)
    model.eval()
    tokenizer = CharacterTokenizer(characters=["A", "C", "G", "T", "N"], model_max_length=MAX_LENGTHS[args.model])

    sequence = random_sequence(200, seed=1)
    other_sequence = random_sequence(200, seed=2)

    failures = []

    # --- smoke test -------------------------------------------------------
    emb_a, _ = embed(model, tokenizer, sequence, device)
    expected_shape = (1, len(sequence) + 2, config["d_model"])  # +2 for special tokens
    if tuple(emb_a.shape) != expected_shape:
        failures.append(f"shape mismatch: got {tuple(emb_a.shape)}, expected {expected_shape}")
    if torch.isnan(emb_a).any() or torch.isinf(emb_a).any():
        failures.append("embeddings contain NaN or Inf")
    print(f"[smoke test] shape={tuple(emb_a.shape)} nan={torch.isnan(emb_a).any().item()} "
          f"inf={torch.isinf(emb_a).any().item()}")

    # --- determinism --------------------------------------------------------
    emb_a_repeat, _ = embed(model, tokenizer, sequence, device)
    same_seq_dist = (emb_a.float() - emb_a_repeat.float()).norm().item()
    if same_seq_dist > 1e-5:
        failures.append(f"non-deterministic output across repeated runs: L2 distance {same_seq_dist}")
    print(f"[determinism] L2 distance across repeated runs of the same sequence: {same_seq_dist:.2e}")

    # --- sensitivity to input content ---------------------------------------
    emb_b, _ = embed(model, tokenizer, other_sequence, device)
    diff_seq_dist = (emb_a.float() - emb_b.float()).norm().item()
    embedding_scale = emb_a.float().norm().item()
    print(f"[validation] L2 distance between two unrelated sequences: {diff_seq_dist:.2f} "
          f"(embedding norm scale: {embedding_scale:.2f})")

    if diff_seq_dist < 0.05 * embedding_scale:
        failures.append(
            f"embeddings barely changed between two unrelated sequences "
            f"(distance {diff_seq_dist:.2f}, <5% of embedding scale {embedding_scale:.2f}) "
            f"-- model may be ignoring its input"
        )

    # --- pretraining objective: real vs. composition-matched shuffle ----------
    real = read_fasta(args.fasta)[:MAX_LENGTHS[args.model] - 2]
    shuffled = "".join(random.Random(0).sample(real, len(real)))
    max_length = MAX_LENGTHS[args.model]
    real_ll = mean_next_nucleotide_loglik(model, real, max_length, device)
    shuffled_ll = mean_next_nucleotide_loglik(model, shuffled, max_length, device)
    print(f"[objective] mean next-nucleotide log-lik: real={real_ll:.4f} shuffled={shuffled_ll:.4f} "
          f"(gap {real_ll - shuffled_ll:+.4f} nats)")
    if real_ll <= shuffled_ll:
        failures.append(
            f"real sequence is not more predictable than its shuffle "
            f"(log-lik {real_ll:.4f} vs {shuffled_ll:.4f}) -- possible loading or tokenization bug"
        )

    if failures:
        print("\nVALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
