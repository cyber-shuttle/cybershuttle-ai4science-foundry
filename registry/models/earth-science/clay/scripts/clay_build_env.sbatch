#!/bin/bash
#SBATCH --job-name=clay-build-env
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/logs/%x-%j.out
#SBATCH --error=benchmarks/logs/%x-%j.err
#
# Step 1 of the Clay workflow: build the conda environment on a CPU node
# (login nodes kill large installs) and record exactly what was installed.
#
# Submit from the Clay model folder:
#   sbatch scripts/clay_build_env.sbatch
#
# What it does:
#   1. Checks that this node can reach GitHub, PyPI, the PyTorch wheel index
#      and conda-forge (fails in seconds with a clear message if not).
#   2. Clones Clay-foundation/model and checks out the pinned commit.
#   3. Creates the env from scripts/clay_environment.yml (Python + pip only),
#      then installs scripts/clay_requirements.txt (exact versions, sha256
#      hashes, no dependency resolution) and claymodel from the pinned source.
#   4. Runs scripts/clay_check_env.py and `pip check`.
#   5. Writes conda-env-export.yml and pip-freeze.txt to
#      benchmarks/outputs/env/.
#
# Re-running is safe. A finished env is reused unless the environment file,
# the lock, or the Clay commit changed; then it is rebuilt from scratch.

set -euo pipefail
trap 'echo "== clay-build-env FAILED at line $LINENO (exit $?)  $(date -Is)" >&2' ERR

: "${SLURM_SUBMIT_DIR:?Submit with sbatch from the Clay model folder.}"
cd "$SLURM_SUBMIT_DIR"
# shellcheck source=scripts/clay_common.sh
source scripts/clay_common.sh
clay_require_repo_root
clay_make_dirs
ENV_RECORD="$CLAY_EVIDENCE/outputs/env"
rm -rf "$ENV_RECORD" && mkdir -p "$ENV_RECORD"   # records of an earlier run are replaced, not mixed

echo "== clay-build-env  job=${SLURM_JOB_ID:-none}  node=$(hostname)  start=$(date -Is)"
echo "== partition=${SLURM_JOB_PARTITION:-?}  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "== $(clay_code_version)"
echo "== CLAY_WORKDIR=$CLAY_WORKDIR"
echo "== OS: $(cat /etc/redhat-release 2>/dev/null || uname -sr)  $(ldd --version | head -1)"

echo "== 1. Network check"
clay_check_url "https://github.com/Clay-foundation/model"
clay_check_url "https://pypi.org/simple/pip/"
clay_check_url "https://download.pytorch.org/whl/cu121/torch/"
clay_check_url "https://conda.anaconda.org/conda-forge/noarch/repodata.json.zst"

echo "== 2. Clay source at $CLAY_COMMIT"
if [[ ! -d "$CLAY_REPO/.git" ]]; then
    git clone --quiet "$CLAY_GIT_URL" "$CLAY_REPO"
fi
git -C "$CLAY_REPO" fetch --quiet origin
git -C "$CLAY_REPO" checkout --quiet --detach "$CLAY_COMMIT"
if [[ "$(git -C "$CLAY_REPO" rev-parse HEAD)" != "$CLAY_COMMIT" ]]; then
    echo "ERROR: $CLAY_REPO is not at $CLAY_COMMIT" >&2; exit 1
fi
if [[ -n "$(git -C "$CLAY_REPO" status --porcelain)" ]]; then
    echo "ERROR: $CLAY_REPO has local changes; delete it and resubmit." >&2; exit 1
fi
echo "   Clay source commit: $(git -C "$CLAY_REPO" rev-parse HEAD)"

echo "== 3. Conda environment at $CLAY_ENV"
clay_load_conda
echo "   $(conda --version)  (module miniforge/24.3.0-0)"
STAMP=$(cat scripts/clay_environment.yml scripts/clay_requirements.txt <(echo "$CLAY_COMMIT") | sha256sum | cut -d' ' -f1)
if [[ -f "$CLAY_ENV/.clay_env_complete" && "$(cat "$CLAY_ENV/.clay_env_complete")" == "$STAMP" ]]; then
    echo "   Reusing finished env (inputs unchanged)."
    set +u; conda activate "$CLAY_ENV"; set -u
else
    if [[ -e "$CLAY_ENV" ]]; then
        echo "   Removing old or unfinished env."
        rm -rf "${CLAY_WORKDIR:?}/envs/clay"
    fi
    conda env create --quiet -p "$CLAY_ENV" -f scripts/clay_environment.yml
    set +u; conda activate "$CLAY_ENV"; set -u
    python -m pip install --progress-bar off --require-hashes --no-deps -r scripts/clay_requirements.txt
    python -m pip install --progress-bar off --no-deps --no-build-isolation "$CLAY_REPO"
fi
echo "   python: $(command -v python)"

echo "== 4. Checks"
python scripts/clay_check_env.py --lock scripts/clay_requirements.txt --teacher "$CLAY_TEACHER"
python -m pip check

echo "== 5. Record the environment"
conda env export -p "$CLAY_ENV" --no-builds | grep -v '^prefix:' > "$ENV_RECORD/conda-env-export.yml"
{
    python -m pip freeze --exclude claymodel
    echo "claymodel @ git+${CLAY_GIT_URL}@${CLAY_COMMIT}"
} > "$ENV_RECORD/pip-freeze.txt"
echo "$STAMP" > "$CLAY_ENV/.clay_env_complete"
ls -l "$ENV_RECORD"
du -sh "$CLAY_ENV" | sed 's/^/   env size: /'

echo "== BUILD ENV COMPLETED  end=$(date -Is)"
