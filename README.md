# Cybershuttle AI Model Foundry

**A community registry for curating, benchmarking, and running AI4Science models on Nexus and other advanced cyberinfrastructure. The goal is to turn AI4Science models into portable, optimized, reproducible, and executable scientific workflows.**

[Goals](#goals) · [Model lifecycle](#how-a-model-moves-through-the-foundry) · [Get involved](#get-involved) · [Repository structure](#repository-structure) · [Key concepts](#key-concepts) · [Example model entry](#example-model-entry) · [Issue template](#issue-template-add-ai4science-model) · [Optimization tracks](#optimization-tracks) · [Roadmap](#roadmap) · [Contributing](#contributing)

---

AI models are becoming reusable scientific instruments. Foundation models for proteins, molecules, genomics, weather, materials, and more are increasingly central to discovery. But using them at scale is still hard. Researchers run into the same questions over and over:

- Which models are relevant to my problem?
- What hardware do they need?
- Is there a validated container or environment?
- How do I reproduce someone else's workflow?
- Can I run this on Nexus?

The Cybershuttle AI Model Foundry goal is to answer those questions in a structured, community-maintained, execution-ready way. It adds the missing cyberinfrastructure layer on top of public model hubs like Hugging Face, NVIDIA NGC, and BioNeMo:

> Public catalogs answer, *"What models exist?"*
> The Foundry answers, *"Which models can I use for my science, how do I run them efficiently, and how do I reproduce the workflow?"*

Nexus is the flagship deployment and validation environment. The broader framework is resource-agnostic with a goal to support GPU, TPU, CPU, HPC, cloud, and campus computing through portable metadata, execution recipes, resource profiles, and Cybershuttle workflows.

---

## Goals

1. **Registry**: catalog AI4Science models across biology, chemistry, genomics, climate, materials, physics, medicine, and more
2. **Requirements**: record GPU memory, CPU/RAM, storage, runtimes, software dependencies, and distributed training needs
3. **Recipes**: community-developed execution recipes for training, fine-tuning, inference, benchmarking, and deployment
4. **Workflows**: machine-readable metadata that generates executable Cybershuttle and Slurm workflows
5. **Optimization**: vendor-neutral profiles for NVIDIA GPU, Google TPU, PyTorch, JAX/XLA, and portable pathways
6. **Reproducibility**: versioned, benchmarked, validated models that are easier to share and reuse

---

## How a model moves through the Foundry

```
candidate → registered → containerized → nexus-tested → benchmarked → optimized → cybershuttle-ready
```

| Status | Meaning |
|---|---|
| `candidate` | Model has been suggested or requested |
| `registered` | Metadata is complete and schema-valid |
| `containerized` | Runnable container or software environment exists |
| `nexus-tested` | Model has successfully run on Nexus |
| `benchmarked` | Runtime, memory, and science metrics are recorded |
| `optimized` | Hardware-specific optimization profile exists |
| `cybershuttle-ready` | Workflow can be generated and launched |
| `production` | Maintained, documented, benchmarked, and user-ready |

---

## Get involved

### Step 1: Tell us which models you want to run

Open an issue using the **Add AI4Science Model** template. You don't need to know every detail, a partial request is still useful.

Include as much as you can:

- model name and scientific domain
- link to the paper, GitHub repo, Hugging Face page, or NGC entry
- intended use: training, fine-tuning, inference, evaluation, or deployment
- expected input datasets
- current hardware and any known bottlenecks
- whether you want to collaborate on an optimized workflow

Example model domains we're prioritizing:

- protein and molecular foundation models
- genomics and pangenome models
- weather, climate, and energy models
- neural operators and surrogate models
- vision transformers for scientific imaging
- speech and multimodal foundation models
- interpretability and attribution workflows

### Step 2: Collaborate on optimized Nexus workflows

Once models are identified, we work with contributors, domain scientists, Nexus staff, and optimization partners to develop validated containers, sample datasets, training and inference recipes, Slurm and Cybershuttle workflows, GPU and TPU optimization profiles, benchmark records, and documentation.

---

## Repository structure

```
cybershuttle-ai-model-foundry/
├── schemas/                  # Generated JSON Schema (do not edit by hand)
├── registry/
│   ├── models/               # Per-model metadata (by domain)
│   ├── datasets/
│   └── collections/
├── recipes/                  # Training, fine-tuning, inference, evaluation, deployment
├── resource-profiles/        # Nexus, Google, NVIDIA, ACCESS, cloud, generic
├── optimization-profiles/    # NVIDIA, Google, portable, experimental
├── containers/               # Base, domain, and build containers
├── workflows/                # Cybershuttle, Slurm, notebooks, services
├── benchmarks/               # Definitions, results, reports
├── tools/                    # CLI, schema generation, workflow utilities
├── docs/
└── .github/
    ├── workflows/
    └── ISSUE_TEMPLATE/
```

---

## Key concepts

### Schema design

All Foundry objects (models, datasets, recipes, resource profiles, optimization profiles, benchmarks, workflows) are defined as **Pydantic models** in `tools/schemas/`. These are the authoritative source of truth. JSON Schema 2020-12 files in `schemas/` are generated from them — do not edit those files by hand.

```bash
# Regenerate schemas/ after changing a Pydantic model
python tools/generate_schemas.py
```

YAML files in the registry include a `$schema` header so editors provide inline validation and autocomplete automatically:

```yaml
# yaml-language-server: $schema=../../../schemas/model.schema.json
id: molformer
name: MoLFormer
...
```

This works out of the box in VS Code with the [YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml).

### Model registry

Each model is described in a `model.yaml` file covering domain, model family, upstream source, lifecycle stage, datasets, containers, execution recipes, resource requirements, optimization profiles, readiness status, and maintainers.

```
registry/models/chemistry/molformer/model.yaml
```

### Execution recipes

Recipes describe how to run a model for a specific task.

```
registry/models/chemistry/molformer/recipes/finetune-lora.yaml
```

### Resource profiles

Resource profiles describe where a model can run — Nexus partitions, Google TPU environments, NVIDIA reference systems, ACCESS resources, campus clusters, and cloud.

```
resource-profiles/nexus/nexus-b200.yaml
resource-profiles/google/tpu-v5e.yaml
```

### Optimization profiles

Optimization profiles describe how to improve execution on a given hardware or software stack.

```
optimization-profiles/nvidia/b200-fsdp-bf16.yaml
optimization-profiles/google/jax-xla-sharding.yaml
optimization-profiles/portable/pytorch-fsdp.yaml
```

### Cybershuttle workflows

Model metadata, execution recipes, resource profiles, and optimization profiles combine into a generated, executable workflow:

```
model.yaml + recipe.yaml + resource_profile.yaml + optimization_profile.yaml
  → generated Cybershuttle workflow
  → Nexus / TPU / HPC / cloud execution
```

---

## Example model entry

```yaml
# yaml-language-server: $schema=../../../schemas/model.schema.json
id: molformer
name: MoLFormer
domain:
  - chemistry
  - molecular-discovery
model_family:
  - transformer
  - foundation-model
  - molecular-language-model
tasks:
  - inference
  - fine-tuning
  - property-prediction
upstream:
  huggingface: ibm/MoLFormer-XL-both-10pct
  github: https://github.com/IBM/molformer
  paper: https://arxiv.org/abs/2106.09553
lifecycle:
  supported_stages:
    - inference
    - fine-tuning
    - evaluation
datasets:
  recommended:
    - zinc
    - chembl
resource_profiles:
  recommended:
    - resource-profiles/nexus/nexus-b200.yaml
    - resource-profiles/nexus/nexus-rtx6000.yaml
optimization_profiles:
  recommended:
    - optimization-profiles/nvidia/b200-fsdp-bf16.yaml
    - optimization-profiles/portable/mixed-precision.yaml
cybershuttle:
  enabled: true
  workflow_templates:
    - workflows/cybershuttle/templates/finetune-workflow.yaml.j2
readiness:
  registry_status: candidate
  nexus_validation: untested
  benchmark_status: not-started
  optimization_status: not-started
  workflow_status: candidate
```

---

## CLI (planned)

```bash
foundry validate

foundry list models --domain chemistry

foundry generate \
  --model molformer \
  --recipe finetune-lora \
  --resource nexus-b200 \
  --optimization nvidia-b200-fsdp-bf16 \
  --target cybershuttle
# → workflows/cybershuttle/generated/molformer-finetune-lora-nexus-b200.yaml
```

---

## Issue template: Add AI4Science Model

```markdown
## Model name

## Scientific domain

## Model link or reference
- Paper:
- GitHub:
- Hugging Face:
- NVIDIA NGC / NIM / BioNeMo / NeMo:
- Other:

## Intended use
- [ ] Training
- [ ] Fine-tuning
- [ ] Inference
- [ ] Evaluation
- [ ] Deployment
- [ ] Benchmarking

## Why is this model useful for your science?

## Datasets or inputs needed

## Current hardware or software environment, if known

## Known bottlenecks
- [ ] GPU memory
- [ ] CPU/RAM
- [ ] Storage / I/O
- [ ] Distributed training
- [ ] Inference latency
- [ ] Containerization
- [ ] Workflow integration
- [ ] Other:

## Desired Nexus execution mode
- [ ] Batch job
- [ ] Interactive notebook
- [ ] Inference service
- [ ] Distributed training workflow
- [ ] Cybershuttle application

## Are you interested in collaborating on an optimized workflow?
- [ ] Yes
- [ ] No
- [ ] Maybe
```

---

## Optimization tracks

The Foundry supports multiple hardware tracks without being vendor-specific.

**NVIDIA GPU** — B200 and RTX profiles, NGC containers, NeMo and BioNeMo workflows, TensorRT-LLM inference, NIM deployment, NCCL tuning, FSDP and tensor/pipeline parallelism, mixed precision and FP8, multi-node scaling benchmarks

**Google TPU** — JAX/XLA execution, TPU sharding strategies, PJRT runtime, MaxText-style training recipes, TPU inference pathways, GPU-to-TPU portability studies

**Portable** — PyTorch DDP/FSDP, DeepSpeed ZeRO, ONNX Runtime, vLLM, mixed precision, quantization, checkpointing, containerized execution

Vendor contributions are welcome provided they use the common schemas and remain interoperable with the rest of the framework.

---

## Roadmap

**Phase 1 — Community intake**
Collect model requests, create initial entries, identify high-priority Nexus use cases, define minimal schemas.

**Phase 2 — Nexus validation**
Containerize selected models, run initial workflows, capture hardware requirements, publish sample Slurm and Cybershuttle workflows.

**Phase 3 — Optimization**
Develop NVIDIA GPU and Google TPU optimization profiles, add portable PyTorch and JAX pathways, publish benchmark records.

**Phase 4 — Cybershuttle integration**
Generate workflows programmatically from model metadata, provide sample applications and notebooks, support model selection and resource matching through Cybershuttle.

**Phase 5 — Community expansion**
Add more domains, models, datasets, and resource profiles. Support broader national cyberinfrastructure. Establish maintainers and working groups.

---

## Contributing

Contributions welcome via GitHub issues and pull requests.

- **Model request** — open an issue with the Add AI4Science Model template
- **Execution recipe** — submit a PR linking a recipe to a model entry with expected inputs, outputs, resources, and dependencies
- **Resource profile** — submit a profile for a Nexus partition, TPU resource, ACCESS system, cloud instance, or campus cluster
- **Optimization profile** — submit vendor-specific or portable optimization profiles (quantization, FSDP, TensorRT-LLM, NIM, ONNX, etc.)
- **Schema changes** — edit the Pydantic models in `tools/schemas/`, then run `python tools/generate_schemas.py` to regenerate `schemas/`. Never edit `schemas/*.json` directly
- **Benchmark results** — submit results using the benchmark schema: runtime, peak memory, throughput, scaling efficiency, cost, energy, science-specific metrics

See [CONTRIBUTING.md](CONTRIBUTING.md) and [GOVERNANCE.md](GOVERNANCE.md) for full details.