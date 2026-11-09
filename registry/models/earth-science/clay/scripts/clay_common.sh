# shellcheck shell=bash disable=SC2034  # variables are used by the scripts that source this file
# Shared settings for every Clay job script. Sourced, not run:
#   source scripts/clay_common.sh
#
# This is the single place for paths, pinned versions, URLs, and checksums.
# Every value marked "verified" was checked against the upstream source on
# 2026-10-11 (see registry/models/earth-science/clay/README.md).

# ---- Where things go -------------------------------------------------------
# Everything large lives under $CLAY_WORKDIR (scratch), never in this repo or
# in $HOME (30 GB quota). Override by exporting CLAY_WORKDIR before sbatch.
CLAY_WORKDIR="${CLAY_WORKDIR:-$HOME/scratch/clay}"
CLAY_REPO="$CLAY_WORKDIR/clay-model"       # Clay source at the pinned commit
CLAY_ENV="$CLAY_WORKDIR/envs/clay"         # conda environment
CLAY_WEIGHTS="$CLAY_WORKDIR/weights"       # clay-v1.5.ckpt
CLAY_DATA="$CLAY_WORKDIR/data"             # NAIP tile + reference parquet
CLAY_RESULTS="$CLAY_WORKDIR/results"       # embeddings written by jobs

# Evidence that gets committed (relative to the Clay model folder).
CLAY_EVIDENCE="benchmarks"

# ---- Pinned upstream artifacts (verified) ----------------------------------
CLAY_GIT_URL="https://github.com/Clay-foundation/model.git"
CLAY_COMMIT="f14e698f3c237cabf8d28dec669a362d66625381"

# Hugging Face made-with-clay/Clay, file v1.5/clay-v1.5.ckpt, pinned to a repo revision.
CLAY_HF_REVISION="70200ebcccdf67bf2a0cb9984c77ddee26c10ed2"
CLAY_CKPT_URL="https://huggingface.co/made-with-clay/Clay/resolve/${CLAY_HF_REVISION}/v1.5/clay-v1.5.ckpt"
CLAY_CKPT_SHA256="21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0"
CLAY_CKPT_BYTES=5158764629

# DINOv2 "teacher" named in the checkpoint's hyperparameters. Clay builds it
# with timm.create_model(..., pretrained=True) even for inference, so it is
# downloaded once here and read from the cache (offline) in GPU jobs.
CLAY_TEACHER="vit_large_patch14_reg4_dinov2.lvd142m"
CLAY_TEACHER_HF_REPO="timm/vit_large_patch14_reg4_dinov2.lvd142m"
CLAY_TEACHER_HF_REVISION="f3c408e77602bb412aa65fb03dfa0d5f95cb3832"
CLAY_TEACHER_SHA256="c893d72294d4c327e631ff92f428dbc14c4f93cb5581b6c5f9d89bb5d17def27"

# NAIP input tile, from Microsoft Planetary Computer (free, no account).
NAIP_STAC_API="https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_ITEM_ID="az_m_3211002_se_12_1_20150530_20151022"
NAIP_TILE_NAME="m_3211002_se_12_1_20150530.tif"
NAIP_TILE_SHA256="4bd22adff7877ef333561620a0b0d8e9c5da25adb9c3e131b865849679ade567"
NAIP_TILE_BYTES=188776628

# Clay's published embeddings for the same tile (rev2, CC-BY-4.0, source.coop).
REF_URL="https://data.source.coop/clay/clay-v1-5-naip-2/az/2015/100cm/rgbir_cog/32110/m_3211002_se_12_1_20150530.parquet"
REF_NAME="m_3211002_se_12_1_20150530.parquet"
REF_SHA256="a551efc22b3ab2e3f0317f9b97a821550baafcb1e0fa2bd5a36fc0c186bece54"

# ---- Keep caches and temp files out of $HOME --------------------------------
export CONDA_PKGS_DIRS="$CLAY_WORKDIR/cache/conda_pkgs"
export PIP_CACHE_DIR="$CLAY_WORKDIR/cache/pip"
export HF_HOME="$CLAY_WORKDIR/cache/huggingface"
export TORCH_HOME="$CLAY_WORKDIR/cache/torch"
export XDG_CACHE_HOME="$CLAY_WORKDIR/cache/xdg"
export MPLCONFIGDIR="$CLAY_WORKDIR/cache/matplotlib"
export TMPDIR="$CLAY_WORKDIR/tmp"
export PYTHONNOUSERSITE=1      # ignore packages in ~/.local
export PYTHONUNBUFFERED=1      # log lines appear immediately
export WANDB_MODE=disabled     # Clay depends on wandb; we never log to it
export CONDA_NOTIFY_OUTDATED_CONDA=false   # ICE's conda 23.7.4 otherwise nags in every .err log

clay_make_dirs() {
    mkdir -p "$CLAY_WORKDIR/envs" "$CLAY_WEIGHTS" "$CLAY_DATA" "$CLAY_RESULTS" \
        "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
        "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"
}

# Stop unless we are in the Clay model folder (sbatch jobs cd to $SLURM_SUBMIT_DIR).
clay_require_repo_root() {
    if [[ ! -f scripts/clay_common.sh || ! -d "$CLAY_EVIDENCE/logs" ]]; then
        echo "ERROR: run this from registry/models/earth-science/clay (the folder with scripts/ and benchmarks/)." >&2
        exit 1
    fi
}

# Slurm settings for people whose account is not the one in the job scripts
# (account coc, QOS coc-ice, partitions coc-cpu / coc-gpu, GPU gpu:l40s:1).
# Account and QOS: Slurm's own SBATCH_ACCOUNT and SBATCH_QOS variables
# override the #SBATCH lines. Partitions and GPU type: set these, and
# clay_run_all.sh and clay_preflight.sh pass them on the sbatch command line:
#   CLAY_CPU_PARTITION   for the env build and download jobs
#   CLAY_GPU_PARTITION   for the smoke test and inference jobs
#   CLAY_GPU_GRES        GPU request for the GPU jobs, e.g. gpu:a100:1
# Sets the array CLAY_SBATCH_OPTS for one job script.
clay_sbatch_opts() {
    local script="$1"
    CLAY_SBATCH_OPTS=()
    if grep -q '^#SBATCH --gres=gpu' "$script"; then
        if [[ -n "${CLAY_GPU_PARTITION:-}" ]]; then CLAY_SBATCH_OPTS+=(--partition="$CLAY_GPU_PARTITION"); fi
        if [[ -n "${CLAY_GPU_GRES:-}" ]]; then CLAY_SBATCH_OPTS+=(--gres="$CLAY_GPU_GRES"); fi
    else
        if [[ -n "${CLAY_CPU_PARTITION:-}" ]]; then CLAY_SBATCH_OPTS+=(--partition="$CLAY_CPU_PARTITION"); fi
    fi
}

# The partition a job script will run in, after any override.
clay_job_partition() {
    local script="$1"
    if grep -q '^#SBATCH --gres=gpu' "$script" && [[ -n "${CLAY_GPU_PARTITION:-}" ]]; then
        echo "$CLAY_GPU_PARTITION"
    elif ! grep -q '^#SBATCH --gres=gpu' "$script" && [[ -n "${CLAY_CPU_PARTITION:-}" ]]; then
        echo "$CLAY_CPU_PARTITION"
    else
        grep -m1 '^#SBATCH --partition=' "$script" | cut -d= -f2
    fi
}

# Which code ran. The scripts/ tree hash stays the same when only docs or
# evidence are amended into the commit, so logs can be matched to the final
# commit with: git rev-parse HEAD:./scripts
clay_code_version() {
    local dirty
    dirty=$(git status --porcelain -- scripts | wc -l | tr -d ' ')
    echo "foundry commit $(git rev-parse --short HEAD)  scripts tree $(git rev-parse HEAD:./scripts)  uncommitted script changes: $dirty"
}

# Load conda from the PACE module. Module and conda hooks break under `set -u`,
# so it is switched off around them and put back only if it was on before
# (the job scripts use it; an interactive shell usually does not). Errors
# `return` instead of `exit`, so a mistake does not close an interactive
# shell; the job scripts run with `set -e`, which still stops them.
clay_load_conda() {
    if ! type module >/dev/null 2>&1; then
        echo "ERROR: the 'module' command is not available in this shell (Lmod missing)." >&2
        return 1
    fi
    local nounset=$-
    set +u
    module load miniforge/24.3.0-0
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if [[ $nounset == *u* ]]; then set -u; fi
}

clay_activate() {
    clay_load_conda || return 1
    if [[ ! -f "$CLAY_ENV/.clay_env_complete" ]]; then
        echo "ERROR: no finished environment at $CLAY_ENV. Run: sbatch scripts/clay_build_env.sbatch" >&2
        return 1
    fi
    local nounset=$-
    set +u
    conda activate "$CLAY_ENV"
    if [[ $nounset == *u* ]]; then set -u; fi
}

# Setup shared by the GPU jobs: activate the env, refuse to run without the
# downloaded inputs, keep Hugging Face offline (the teacher is cached), and
# print the GPU and installed versions into the job log.
# Sets CLAY_EMBED_INPUTS for scripts/clay_embed.py.
clay_gpu_setup() {
    clay_activate
    export HF_HUB_OFFLINE=1
    local f
    for f in "$CLAY_WEIGHTS/clay-v1.5.ckpt" "$CLAY_DATA/$NAIP_TILE_NAME" "$CLAY_DATA/$REF_NAME" \
             "$CLAY_REPO/configs/metadata.yaml"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: missing $f. Run: sbatch scripts/clay_download.sbatch" >&2
            exit 1
        fi
    done
    echo "== GPU"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
    python scripts/clay_check_env.py --lock scripts/clay_requirements.txt --teacher "$CLAY_TEACHER"
    CLAY_EMBED_INPUTS=(--tile "$CLAY_DATA/$NAIP_TILE_NAME" --reference "$CLAY_DATA/$REF_NAME"
                       --checkpoint "$CLAY_WEIGHTS/clay-v1.5.ckpt"
                       --metadata "$CLAY_REPO/configs/metadata.yaml")
}

# Print "OK <code>" or fail with a clear message if a URL is unreachable.
clay_check_url() {
    local url="$1" code
    # Ask for 1 byte first (big files); some APIs (Planetary Computer STAC)
    # answer 502 to byte-range requests, so retry those with a plain GET.
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 -r 0-0 "$url" 2>/dev/null || true)
    if [[ ! "$code" =~ ^(200|206|301|302|307|308)$ ]]; then
        code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 "$url" 2>/dev/null || true)
    fi
    if [[ "$code" =~ ^(200|206|301|302|307|308)$ ]]; then
        echo "   reachable ($code)  $url"
    else
        echo "ERROR: cannot reach $url (HTTP '${code:-none}') from $(hostname)." >&2
        echo "       Compute nodes may not have internet access. See Troubleshooting in" >&2
        echo "       registry/models/earth-science/clay/README.md." >&2
        exit 1
    fi
}

# Verify a file's sha256. Prints the checksum line on success.
clay_verify_sha256() {
    local file="$1" expected="$2" actual
    actual=$(sha256sum "$file" | cut -d' ' -f1)
    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: checksum mismatch for $file" >&2
        echo "       expected $expected" >&2
        echo "       got      $actual" >&2
        return 1
    fi
    echo "$actual  $(basename "$file")"
}

# Add or replace one "sha256  name" line in the committed checksum record.
clay_record_checksum() {
    local line="$1" name record="$CLAY_EVIDENCE/outputs/checksums.sha256"
    name="${line#*  }"
    mkdir -p "$(dirname "$record")"
    touch "$record"
    { grep -v "  ${name}\$" "$record" || true; echo "$line"; } | sort -k2 > "$record.tmp"
    mv "$record.tmp" "$record"
}

# Download URL to FILE (resumable), then check size and sha256. Skips the
# download when FILE already exists with the right checksum. The URL is never
# printed, because signed URLs contain an access token.
clay_fetch() {
    local url="$1" file="$2" sha="$3" bytes="${4:-}" line
    if [[ -f "$file" ]] && line=$(clay_verify_sha256 "$file" "$sha" 2>/dev/null); then
        echo "   already present, checksum OK: $file"
    else
        echo "   downloading $(basename "$file") ..."
        curl -fL --retry 5 --retry-delay 10 --silent --show-error -C - -o "$file.part" "$url"
        if [[ -n "$bytes" && "$(stat -c %s "$file.part")" != "$bytes" ]]; then
            echo "ERROR: $file.part is $(stat -c %s "$file.part") bytes, expected $bytes. Delete it and rerun." >&2
            return 1
        fi
        if ! line=$(clay_verify_sha256 "$file.part" "$sha"); then
            rm -f "$file.part"
            echo "ERROR: removed the bad download; rerun to fetch it again." >&2
            return 1
        fi
        mv "$file.part" "$file"
        echo "   downloaded and verified: $file"
    fi
    # Record the path under $CLAY_WORKDIR, so `sha256sum -c` works from there.
    line="$sha  ${file#"$CLAY_WORKDIR"/}"
    clay_record_checksum "$line"
    echo "   sha256 $line"
}
