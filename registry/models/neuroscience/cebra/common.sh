# Shared setup sourced by the .sbatch files. Not meant to be run directly.
ENV_NAME=${ENV_NAME:-cebra}

module load anaconda3 2>/dev/null || module load miniforge3 2>/dev/null || module load miniconda3 2>/dev/null || true
set +u  # conda activate scripts reference unset variables
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

export CEBRA_DATADIR="${SLURM_SUBMIT_DIR}/data"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p logs outputs

JOB_T0=$(date +%s)
echo "===== JOB INFO ====="
echo "job_id=${SLURM_JOB_ID} name=${SLURM_JOB_NAME} partition=${SLURM_JOB_PARTITION:-?}"
echo "node=$(hostname) start=$(date -Is)"
echo "cpus=${SLURM_CPUS_PER_TASK:-?} mem_per_node=${SLURM_MEM_PER_NODE:-?}MB gpus=${CUDA_VISIBLE_DEVICES:-none}"
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')"
echo "python=$(which python)"
nvidia-smi || echo "WARN: nvidia-smi unavailable"
if [ ! -d "$CEBRA_DATADIR" ]; then
  echo "ERROR: $CEBRA_DATADIR missing. Run 'bash setup_env.sh' on the login node first." >&2
  exit 1
fi
# Background GPU usage sampler (every 10 s) -> evidence of resource use
GPU_LOG="logs/gpu_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 10 > "$GPU_LOG" 2>/dev/null &
GPU_MON_PID=$!
echo "===================="

finish_job() {
  local out=$1
  kill "$GPU_MON_PID" 2>/dev/null || true
  cp "$GPU_LOG" "$out/" 2>/dev/null || true
  echo "===== JOB DONE ====="
  echo "end=$(date -Is) wall_seconds=$(( $(date +%s) - JOB_T0 ))"
  echo "outputs in $out:"; ls -la "$out"
  echo "After the job ends, record final accounting with:"
  echo "  sacct -j ${SLURM_JOB_ID} --format=JobID,JobName,Partition,AllocTRES%40,Elapsed,MaxRSS,State,ExitCode"
  echo "  seff ${SLURM_JOB_ID}"
}
