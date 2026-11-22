# Achilles behavioral decoding with POYO

## Objective

Apply POYO to the same Achilles hippocampal neural recording used by CEBRA.
Compare CEBRA representation learning with POYO behavioral decoding. This
directory is independent of the CEBRA reproducibility workflow.

## Dataset

Dataset: `rat-hippocampus-single-achilles`. Expected neural shape: `(10178, 120)`;
aligned behavior shape: `(10178, 3)`. Rows are temporal samples, and the 120
columns retain their original neural feature/unit identity. The behavior
columns are position, right direction, and left direction, according to the
[CEBRA loader](https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/cebra/datasets/hippocampus.py).
That loader documents spike counts in 25 ms bins and reads `spikes` and
`position` from the cached recording. Its matrix interface does not itself
establish original bin coordinates, continuity after preprocessing, or exact
within-bin spike times. Local values and installed loader metadata still need
inspection. No recording duration is inferred from row count alone.

`inspect_achilles.py` uses the existing CEBRA loader with `download=False`,
reports array statistics and example rows, and examines numerical timing/unit
metadata in both the loader and existing cache. It does not download data.

## CEBRA Baseline

CEBRA trains with neural activity and behavior to learn a three-dimensional
latent representation. Inference takes neural activity and produces an
embedding of shape `(10178, 3)`. The existing workflow does not directly decode
behavior. Matching the dimensionality of the embedding and the labels does not
make their coordinates equivalent.

## POYO Approach

Achilles counts → verified temporal/event conversion → compatible spike data
→ official POYO training → behavior predictions.

POYO uses time and unit identities to decode neural populations. All three
Achilles label columns are used as a custom three-output regression target.
Direction outputs are continuous scores; this adapter does not claim they are
calibrated probabilities. MSE/R² are reported separately for each column to
avoid hiding different scales or semantics. No target normalization is applied;
position scale can dominate the training loss. This is an initial decoding
experiment, not a matched benchmark against a CEBRA behavior decoder.

### Timing evidence required before conversion

Supply a JSON document containing:

| Field | Required evidence |
| --- | --- |
| `representation` | Exactly `spike_counts`, confirmed for this cache |
| `counts_source` | Public HTTPS reference confirming count semantics |
| `timing_source` | Public HTTPS reference supporting bin coordinates and continuity/gaps |
| `unit_mapping_source` | Public HTTPS reference supporting the unit/column mapping |
| `behavior_alignment_source` | Public HTTPS reference supporting label timing and row alignment |
| `bin_start_seconds` | 10178 verified bin starts, in original row order |
| `bin_end_seconds` | 10178 verified bin ends, each 0.025 seconds after its start |
| `behavior_timestamps_seconds` | 10178 verified target coordinates, each inside its matching bin |
| `recording_end_seconds` | Coordinate of the final bin end in the same time reference |

The arrays must come from source metadata/preprocessing evidence. A reference
URL is an audit trail, not an automatic proof; review its applicability to the
local cache. Do not synthesize consecutive bin starts solely from row count or
the documented bin width. Explicit gaps are retained. The converter subtracts
one common origin and validates ordering, domain, counts, unit indices and
alignment. Insufficient evidence blocks conversion. Rates, smoothed values,
negative values and fractional counts are rejected.

With `--accept-bin-center-approximation`, a count of *k* becomes *k* event tokens
for that column, all at the documented bin's center. This deterministic
quantization preserves multiplicity without random jitter. These coordinates
are an approximation for model input, not recovered biological spike times.
Timing precision is limited by the original bins. Unit IDs are stable names
for the 120 source columns; they do not invent anatomical metadata.

The preparation writes `temporaldata.Data` containing `units.id`,
`spikes.timestamps`, `spikes.unit_index`, `session.id`, aligned
`behavior.timestamps`/`values`, original row indices and train/valid/test
domains. It uses a chronological 70/15/15 row split; every gap also breaks a
sampling interval. Fixed context windows stay within a split and a continuous
segment. Segments shorter than one second are excluded from sampling, and the
export reports actual test coverage. A trial-aware split can be added once
trial metadata is verified; this split is not claimed to match the CEBRA PR.

## POYO Implementation

Official implementation: [nerdslab/poyo](https://github.com/nerdslab/poyo), for
*A Unified, Scalable Framework for Neural Population Decoding*. This adapter
targets revision `6b052f1`; use a clean checkout of that revision. It reuses
[official train.py](https://github.com/nerdslab/poyo/blob/6b052f1/train.py),
composes `configs/train_poyo_1.yaml` with Hydra, replaces its dataset target
with the local `AchillesDataset`, then passes the configuration to the actual
decorated `main(cfg)` entrypoint. This is Hydra configuration passthrough, not
a new POYO training API. The official CLI examples are:

```bash
python train.py --config-name train_poyo_mp.yaml
python train.py --config-name train_poyo_1.yaml
```

The local dataset subclasses the verified
[`torch_brain.dataset.Dataset`](https://github.com/neuro-galaxy/torch_brain/blob/ca3cfb/torch_brain/dataset/dataset.py)
and supplies recording configuration, unit IDs and split domains. The
[modality registry](https://github.com/neuro-galaxy/torch_brain/blob/ca3cfb/torch_brain/registry.py)
registers `achilles_behavior_3d` with `DataType.CONTINUOUS` and the real
`torch_brain.nn.loss.MSELoss`. The official data module initializes vocabulary
and attaches the model tokenizer. Export restores the official `TrainWrapper`
checkpoint and uses the verified model forward call with its tokenized
`model_inputs`, `torch_brain.data.collate`, and `SequentialFixedWindowSampler`.
Overlapping predictions are averaged by original target row; only test rows
contribute to exported metrics.

Two explicit callback compatibility bridges apply only to the imported module
in memory: the upstream evaluator's `self.log_dict` is forwarded to the actual
Lightning module, and dense gradient statistics are disabled for sparse
embedding gradients. The
[upstream callback code](https://github.com/nerdslab/poyo/blob/6b052f1/src/callbacks.py)
otherwise remains in use. No upstream source files, model architecture,
optimizer or training loop are rewritten.

The [official requirements](https://github.com/nerdslab/poyo/blob/6b052f1/requirements.txt)
pin `torch_brain` at `ca3cfb`, `brainsets` at `4ccee58`, and `temporaldata` at
`0.1.4`, alongside torch, NumPy, Lightning, wandb and torchmetrics. The
historical `pytorch_brain` name is not an additional dependency. The
[current torch_brain repository](https://github.com/neuro-galaxy/torch_brain)
has since merged its data infrastructure; do not mix current-main import
paths with POYO's pinned revision. The pinned stack uses
`from temporaldata import Data, IrregularTimeSeries, Interval, ArrayDict`.

### Future environment setup

The official procedure uses Python 3.11 and uv. From a separately obtained
official checkout, after checking out the reviewed revision:

```bash
git checkout 6b052f1
uv venv venv -p 3.11
source venv/bin/activate
uv pip install -r requirements.txt
```

Then install this directory's `requirements.txt` into that environment. It
references the same official requirements and adds real inspection/adapter
dependencies. There is no invented `poyo` PyPI package. Environment resolution
and GPU compatibility have not been tested. Keep the checkout outside the
integration directory so its source and environment remain separate artifacts.

### Future invocation

Set `POYO_WORKFLOW` to this directory, `POYO_REPO` to the clean official
checkout, `ACHILLES_DATA_ROOT` to the existing cache root, `ACHILLES_TIMING` to
the verified JSON, and `POYO_OUTPUT` to a fresh destination. Then:

```bash
python "$POYO_WORKFLOW/inspect_achilles.py" --data-root "$ACHILLES_DATA_ROOT"
python "$POYO_WORKFLOW/prepare_poyo_input.py" \
  --data-root "$ACHILLES_DATA_ROOT" --timing "$ACHILLES_TIMING" \
  --output "$POYO_OUTPUT/prepared" --accept-bin-center-approximation
python "$POYO_WORKFLOW/run_poyo.py" --poyo-repo "$POYO_REPO" \
  --prepared "$POYO_OUTPUT/prepared" --output "$POYO_OUTPUT/decoding" --epochs 10
```

For PACE, also set `POYO_ENV` and `POYO_ACCEPT_BIN_CENTERS=yes`, then submit
`jobs/poyo.slurm`. It requests `coc-gpu`, `coc-ice`, one GPU, four CPUs, 16 GB,
and 30 minutes. It activates `$POYO_ENV/bin/activate` and runs inspection,
preparation and decoding. Ten epochs is an initial configurable setting, not
a convergence claim; runtime within the allocation is unverified. W&B is
disabled. No cluster modules, accounts or personal paths are assumed.

## Outputs

Only future successful execution creates:

- `prepared/achilles.h5`: event/unit/time input plus all `(10178, 3)` targets.
- `prepared/manifest.json`: evidence, counts, shape and approximation provenance.
- `decoding/predictions.npz`: predictions and targets `(N_test_covered, 3)`,
  timestamps, original row indices and overlap counts.
- `decoding/metrics.json`: per-column MSE/R² and coverage; constant-target R²
  is `null`. No score is claimed here.
- Official training checkpoints/logs and `decoding/upstream_revision.txt`.

Checkpoints and resolved runtime configuration may contain environment paths;
keep execution artifacts outside the source PR. Nothing in this workflow has
been executed or installed, no dataset downloaded, and no logs or results
generated during implementation. Static review cannot establish runtime or
scientific validity; the timing evidence and runtime still need verification.
