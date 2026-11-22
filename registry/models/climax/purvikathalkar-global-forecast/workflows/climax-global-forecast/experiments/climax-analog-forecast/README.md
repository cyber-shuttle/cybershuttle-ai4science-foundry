# ClimaX Analog Forecast Proof of Concept

This isolated experiment compares persistence and training climatology with raw ERA5
analog forecasts and pooled pretrained-ClimaX embedding analog forecasts.

The experiment uses cosine similarity, chronological train/test splits, and a
configurable 72-hour aligned target. It does not fine-tune or retrain ClimaX.

## Run

From the workflow directory:

```bash
python scripts/baseline_forecast.py \
  --data-dir /absolute/path/to/geopotential_forecast_npz \
  --output-dir results/climax-analog-forecast/baseline

python experiments/climax-analog-forecast/scripts/run_experiment.py \
  --data-dir /absolute/path/to/geopotential_forecast_npz \
  --output-dir experiments/climax-analog-forecast/results \
  --reference-embeddings /absolute/path/to/train/climax-embeddings.npz \
  --evaluation-embeddings /absolute/path/to/test/climax-embeddings.npz \
  --lead-hours 72 \
  --time-step-hours 6 \
  --k-values 1 3 5

python experiments/climax-analog-forecast/scripts/visualize_results.py \
  --metrics experiments/climax-analog-forecast/results/metrics.json \
  --output experiments/climax-analog-forecast/results/metrics-comparison.svg
```

Embedding archives must contain `pooled_embeddings`, `sample_indices`, and
`variable`. The evaluation archive indices must exactly match the requested
evaluation initial indices. Reference and evaluation embeddings must not come
from overlapping chronological periods.

## Tests

```bash
python -m unittest discover -s experiments/climax-analog-forecast/tests -v
python -m unittest discover -s tests -v
```
