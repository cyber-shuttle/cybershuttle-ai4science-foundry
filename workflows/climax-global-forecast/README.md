# ClimaX PACE workflow reference

Follow the repository's [top-level README](../../README.md) for the complete, current, step-by-step workflow. The directions for setup, data download and preprocessing, embedding extraction and validation, forecast fine-tuning and inference, wall-time preflight, and troubleshooting are all described there. This page is a landing page for technical evidence and implementation files.

## Technical and assignment records

- [Assignment evidence and rubric coverage](A2DETAILS.md): records the successful PACE embedding run, hashes, validation, limitations, and pending forecast validation.
- [Sanitized embedding logs](logs/): includes successful and failed SLURM attempts for diagnosing setup and inference.
- [Environment specification](environment.yml) and [environment setup helper](scripts/setup_env.sh).
- [Geopotential preprocessor](scripts/prepare_geopotential.py), [baseline forecast and plot](scripts/baseline_forecast.py), [embedding inference](scripts/embed.py), [forecast inference](scripts/infer.py), [forecast SVG helper](scripts/forecast_visualization.py), and [embedding validator](scripts/validate_embeddings.py).
- [SLURM scripts](slurm/): embedding, fine-tuning, and forecast inference jobs.

<!-- The recorded PACE validation covers frozen pretrained embeddings on the 2016 test shard. It does not validate the separate geopotential forecast fine-tuning and inference path; run that path and retain its metrics before claiming forecast results. -->
