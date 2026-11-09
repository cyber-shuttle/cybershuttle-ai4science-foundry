import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

# Official pretrained DNABERT-2 model and pinned revision
MODEL_ID = "zhihan1996/DNABERT-2-117M"
MODEL_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True
)

print("Loading DNABERT-2 model...", flush=True)
config = BertConfig.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    config=config
)

# Select GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device, flush=True)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

model = model.to(device)
model.eval()

# DNA sequence from official DNABERT-2 README
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"

# Tokenize input
inputs = tokenizer(
    dna,
    return_tensors="pt"
)["input_ids"].to(device)

# Run inference using mean pooling
with torch.no_grad():
    hidden_states = model(inputs)[0]
    embedding = torch.mean(hidden_states[0], dim=0)

# Display results
print("Model revision:", MODEL_REVISION, flush=True)
print("DNA sequence:", dna, flush=True)
print("Hidden states shape:", hidden_states.shape, flush=True)
print("Embedding shape:", embedding.shape, flush=True)
print("First 10 embedding values:", embedding[:10], flush=True)

# Validate output
assert embedding.shape == (768,), "Incorrect embedding dimension"
assert torch.isfinite(embedding).all(), "Embedding contains NaN or Inf"

# Create output directory
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Save complete embedding
torch.save(
    embedding.detach().cpu(),
    output_dir / "embedding.pt"
)

# Save human-readable inference results
results = {
    "model": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dna_sequence": dna,
    "device": str(device),
    "hidden_states_shape": list(hidden_states.shape),
    "embedding_shape": list(embedding.shape),
    "embedding_first_10": embedding[:10].detach().cpu().tolist(),
    "all_finite": bool(torch.isfinite(embedding).all().item())
}

with open(output_dir / "inference_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Outputs saved successfully!", flush=True)
print("INFERENCE SUCCESSFUL", flush=True)
