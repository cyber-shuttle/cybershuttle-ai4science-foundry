# ClimaX Analog Forecast on PACE ICE

This proof-of-concept compares:

- persistence;
- training climatology;
- raw ERA5 `geopotential_500` cosine analogs;
- frozen pretrained pooled-ClimaX cosine analogs.

It uses a chronological train/test split and a 72-hour target. It does not train or
fine-tune ClimaX.

## 1. Start the PACE GPU allocation

For embedding extraction, request a V100 explicitly:

```bash
srun --account="<YOUR_PACE_ACCOUNT>" \
  --partition=ice-gpu \
  --gres=gpu:v100:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --pty bash
```

Inside the allocation, initialize the environment:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export WORKFLOW_DIR="$HOME/scratch/climax-inference/registry/models/climax/purvikathalkar-global-forecast/workflows/climax-global-forecast"
export FORECAST_DATA="$PROJECT_DIR/data/geopotential_forecast_npz"
export EMBEDDING_DATA="$PROJECT_DIR/data/geopotential_npz_2016_test"
export CLIMAX_SOURCE="$PROJECT_DIR/ClimaX"
export CHECKPOINT="$PROJECT_DIR/checkpoints/5.625deg.ckpt"
export ANALOG_DIR="$PROJECT_DIR/runs/climax-analog-forecast"

cd "$WORKFLOW_DIR"
conda activate "$HOME/scratch/.conda/envs/climax-inference"
```

Stop here unless the following paths exist:

```bash
test -d "$FORECAST_DATA" || { echo "Missing prepared forecast data: $FORECAST_DATA"; exit 1; }
test -d "$EMBEDDING_DATA" || { echo "Missing prepared embedding data: $EMBEDDING_DATA"; exit 1; }
test -f "$CHECKPOINT" || { echo "Missing checkpoint: $CHECKPOINT"; exit 1; }
test -d "$CLIMAX_SOURCE/src" || { echo "Missing ClimaX source: $CLIMAX_SOURCE"; exit 1; }
mkdir -p "$ANALOG_DIR"
```

If either prepared directory is missing, return to the [main PACE runbook](../../../README.md),
steps 4 and 5, and run the documented downloads and preprocessing first.

## 2. Generate persistence and climatology baselines

This CPU-only command does not require a GPU but must use the active Conda environment:

```bash
python "$WORKFLOW_DIR/scripts/baseline_forecast.py" \
  --data-dir "$FORECAST_DATA" \
  --output-dir "$ANALOG_DIR/baseline" \
  --variable geopotential_500 \
  --split test \
  --lead-hours 72 \
  --num-samples 32 \
  --sample-stride 24
```

Inspect and verify the baseline outputs:

```bash
test -f "$ANALOG_DIR/baseline/baseline-forecast.npz"
test -f "$ANALOG_DIR/baseline/metrics.json"
test -f "$ANALOG_DIR/baseline/baseline-forecast.svg"
cat "$ANALOG_DIR/baseline/metrics.json"
ls -lh "$ANALOG_DIR/baseline"
```

## 3. Extract reference embeddings

From the login shell, submit a train-split embedding job:

```bash
exit

cd "$HOME/scratch/climax-inference/registry/models/climax/purvikathalkar-global-forecast/workflows/climax-global-forecast"
export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export EMBEDDING_DATA="$PROJECT_DIR/data/geopotential_npz_2016_test"
export EMBED_SPLIT=train
export EMBED_TIME_INDEX=0
export EMBED_NUM_SAMPLES=32
export EMBED_SAMPLE_STRIDE=4
export OUTPUT_DIR="$PROJECT_DIR/runs/climax-analog-reference-embeddings"
mkdir -p logs "$OUTPUT_DIR"

sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/embed.sbatch
squeue --me
```

Wait until the job is complete. Verify it:

```bash
test -f "$OUTPUT_DIR/climax-embeddings.npz"
test -f "$OUTPUT_DIR/metrics.json"
cat "$OUTPUT_DIR/metrics.json"
```

## 4. Extract evaluation embeddings

Submit another GPU job for the test split:

```bash
export EMBED_SPLIT=test
export EMBED_TIME_INDEX=0
export EMBED_NUM_SAMPLES=32
export EMBED_SAMPLE_STRIDE=4
export OUTPUT_DIR="$PROJECT_DIR/runs/climax-analog-evaluation-embeddings"
mkdir -p logs "$OUTPUT_DIR"

sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/embed.sbatch
squeue --me
```

Wait until complete, then verify:

```bash
test -f "$OUTPUT_DIR/climax-embeddings.npz"
test -f "$OUTPUT_DIR/metrics.json"
cat "$OUTPUT_DIR/metrics.json"
```

The evaluation embedding indices must match the analog experiment's
`--time-index` and `--sample-stride-hours` values. The commands below use
index 0 and 24-hour spacing.

These paths use 6-hour prepared data arrays. Therefore:

- `EMBED_SAMPLE_STRIDE=4` produces embedding indices `[0, 4, 8, ...]`,
  corresponding to 24-hour spacing;
- `EMBED_SAMPLE_STRIDE=24` produces `[0, 24, 48, ...]`, corresponding to
  144-hour spacing.

Do not reuse the existing embedding archive from a job that used
`EMBED_SAMPLE_STRIDE=24` as the evaluation archive for the command in step 5.

## 5. Run the analog experiment

Enter a CPU allocation:

```bash
srun --account="<YOUR_PACE_ACCOUNT>" \
  --cpus-per-task=4 \
  --mem=16G \
  --time=01:00:00 \
  --pty bash
```

Initialize variables and run:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/scratch/.conda/envs/climax-inference"

export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export WORKFLOW_DIR="$HOME/scratch/climax-inference/registry/models/climax/purvikathalkar-global-forecast/workflows/climax-global-forecast"
export FORECAST_DATA="$PROJECT_DIR/data/geopotential_forecast_npz"
export REFERENCE_EMBEDDINGS="$PROJECT_DIR/runs/climax-analog-reference-embeddings/climax-embeddings.npz"
export EVALUATION_EMBEDDINGS="$PROJECT_DIR/runs/climax-analog-evaluation-embeddings/climax-embeddings.npz"
export ANALOG_DIR="$PROJECT_DIR/runs/climax-analog-forecast"

cd "$WORKFLOW_DIR"
mkdir -p "$ANALOG_DIR/results"

test -f "$REFERENCE_EMBEDDINGS" || { echo "Missing reference embeddings"; exit 1; }
test -f "$EVALUATION_EMBEDDINGS" || { echo "Missing evaluation embeddings"; exit 1; }

python experiments/climax-analog-forecast/scripts/run_experiment.py \
  --data-dir "$FORECAST_DATA" \
  --output-dir "$ANALOG_DIR/results" \
  --reference-embeddings "$REFERENCE_EMBEDDINGS" \
  --evaluation-embeddings "$EVALUATION_EMBEDDINGS" \
  --reference-split train \
  --evaluation-split test \
  --lead-hours 72 \
  --time-step-hours 6 \
  --time-index 0 \
  --num-samples 32 \
  --sample-stride-hours 24 \
  --k-values 1 3 5

python experiments/climax-analog-forecast/scripts/visualize_results.py \
  --metrics "$ANALOG_DIR/results/metrics.json" \
  --output "$ANALOG_DIR/results/metrics-comparison.svg"
```

## 6. Verify and inspect results

```bash
test -f "$ANALOG_DIR/results/analog-forecast.npz"
test -f "$ANALOG_DIR/results/climax-analog-forecast.npz"
test -f "$ANALOG_DIR/results/metrics.json"
test -f "$ANALOG_DIR/results/forecast-comparison.svg"
test -f "$ANALOG_DIR/results/analog-retrieval.svg"
test -f "$ANALOG_DIR/results/metrics-comparison.svg"

cat "$ANALOG_DIR/results/metrics.json"
ls -lh "$ANALOG_DIR/results"
```

Copy the SVG files to a browser or download them locally.

Embedding archives must contain `pooled_embeddings`, `sample_indices`, and
`variable`. Reference and evaluation embeddings must use disjoint chronological
splits.

## 7. Run local tests

```bash
cd "$WORKFLOW_DIR"
python -m unittest discover -s experiments/climax-analog-forecast/tests -v
python -m unittest discover -s tests -v
```
