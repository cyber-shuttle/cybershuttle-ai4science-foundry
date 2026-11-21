import json
import time
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL, OUT_DIR, REVISION

SEQ = "ACTGACTGACTGACTG"

# use pinned model commit
revision = REVISION

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Model:", MODEL)
print("Model revision (commit SHA):", revision)

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=revision, trust_remote_code=True
).to("cuda").eval()

inputs = {k: v.to("cuda") for k, v in tokenizer(SEQ, return_tensors="pt").items()}

start = time.time()
with torch.no_grad():
    out = model(**inputs)
torch.cuda.synchronize()
elapsed = time.time() - start

print("Input sequence:", SEQ)
print("Token IDs:", inputs["input_ids"].tolist())
print("Logits shape:", tuple(out.logits.shape))
print("Inference time (s):", round(elapsed, 4))

# write output to file
os.makedirs(OUT_DIR, exist_ok=True)
torch.save(
    {"input_ids": inputs["input_ids"].cpu(), "logits": out.logits.cpu()},
    os.path.join(OUT_DIR, "smoke_test_logits.pt"),
)
with open(os.path.join(OUT_DIR, "smoke_test_meta.json"), "w") as f:
    json.dump(
        {
            "model": MODEL,
            "revision": revision,
            "sequence": SEQ,
            "logits_shape": list(out.logits.shape),
            "inference_seconds": elapsed,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        f,
        indent=2,
    )
print(f"Saved {OUT_DIR}/smoke_test_logits.pt and {OUT_DIR}/smoke_test_meta.json")