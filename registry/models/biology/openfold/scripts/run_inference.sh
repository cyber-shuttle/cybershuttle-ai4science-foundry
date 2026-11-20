#!/bin/bash
#SBATCH -J openfold-infer
#SBATCH -p ice-gpu
#SBATCH -N 1 -n 1 -c 8
#SBATCH --gres=gpu:1
#SBATCH -C HX00
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-3
#SBATCH -o results/logs/%x-%A_%a.out
# GPU array job: one task per weight set, identical inputs.
# - -C HX00: H100 or H200 (sm_90)
# - Output: $OF_RUNS/<set>/<weight set>/{predictions/,timings.json,gpu_usage.csv}
# - Usage (from registry/models/biology/openfold): sbatch scripts/run_inference.sh [smoke|cameo]
set -eo pipefail
source scripts/config.sh
activate_env

SET=${1:-cameo}
CONFIGS=(
    "openfold_templates     model_1_ptm --openfold_checkpoint_path $OF_PARAMS/openfold_params/finetuning_ptm_2.pt"
    "alphafold2_templates   model_1_ptm --jax_param_path $OF_PARAMS/params/params_model_1_ptm.npz"
    "openfold_no_templates  model_3_ptm --openfold_checkpoint_path $OF_PARAMS/openfold_params/finetuning_no_templ_ptm_1.pt"
    "alphafold2_no_templates model_3_ptm --jax_param_path $OF_PARAMS/params/params_model_3_ptm.npz"
)
read -r NAME PRESET WEIGHTS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
OUT=$OF_RUNS/$SET/$NAME
mkdir -p "$OUT"
echo "set=$SET config=$NAME preset=$PRESET weights=$WEIGHTS host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > "$OUT/gpu_usage.csv" &

python "$OF_REPO/run_pretrained_openfold.py" "$OF_DATA/fasta/$SET" "$OF_DATA/mmcif" \
    --use_precomputed_alignments "$OF_DATA/alignments" \
    --output_dir "$OUT" \
    --config_preset "$PRESET" $WEIGHTS \
    --model_device cuda:0 \
    --data_random_seed 42 \
    --skip_relaxation \
    --max_template_date "$MAX_TEMPLATE_DATE" \
    --release_dates_path "$OF_DATA/mmcif_cache.json" \
    --obsolete_pdbs_path "$OF_DATA/obsolete.dat"
kill %1
