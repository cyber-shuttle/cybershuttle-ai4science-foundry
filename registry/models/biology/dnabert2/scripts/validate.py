import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

# Official pretrained DNABERT-2 model and pinned revision
MODEL_ID = "zhihan1996/DNABERT-2-117M"
MODEL_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Validation device:", device, flush=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True
)

# Load pretrained model
config = BertConfig.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    config=config
).to(device)

model.eval()

# Reference sequence from the official DNABERT-2 README
reference = (
    "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
)

# Identical sequence for reproducibility testing
duplicate = reference

# One-nucleotide substitution
modified = reference[:-1] + (
    "A" if reference[-1] != "A" else "T"
)

def embed(dna):
    inputs = tokenizer(
        dna,
        return_tensors="pt"
    )["input_ids"].to(device)

    with torch.inference_mode():
        hidden = model(inputs)[0]
        vector = hidden[0].mean(dim=0)

    return vector.detach().cpu()

# Generate embeddings
ref_embedding = embed(reference)
dup_embedding = embed(duplicate)
mod_embedding = embed(modified)

# Check expected embedding dimensions
assert ref_embedding.shape == (768,)
assert dup_embedding.shape == (768,)
assert mod_embedding.shape == (768,)

# Check for invalid numerical values
all_finite = all(
    torch.isfinite(x).all().item()
    for x in [ref_embedding, dup_embedding, mod_embedding]
)

assert all_finite, "Embeddings contain NaN or Inf"

# Compare identical DNA sequences
duplicate_max_diff = torch.max(
    torch.abs(ref_embedding - dup_embedding)
).item()

# Compare reference and modified sequences
modified_l2_diff = torch.linalg.vector_norm(
    ref_embedding - mod_embedding
).item()

modified_cosine = F.cosine_similarity(
    ref_embedding.unsqueeze(0),
    mod_embedding.unsqueeze(0)
).item()

duplicate_pass = duplicate_max_diff < 1e-5
assert duplicate_pass, "Identical input validation failed"

# Save validation results
results = {
    "model": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "device": str(device),
    "embedding_dimension": 768,
    "all_finite": all_finite,
    "duplicate_max_absolute_difference": duplicate_max_diff,
    "duplicate_validation_passed": duplicate_pass,
    "modified_l2_difference": modified_l2_diff,
    "modified_cosine_similarity": modified_cosine,
    "reference_sequence": reference,
    "modified_sequence": modified
}

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "validation.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2), flush=True)
print("VALIDATION SUCCESSFUL", flush=True)
