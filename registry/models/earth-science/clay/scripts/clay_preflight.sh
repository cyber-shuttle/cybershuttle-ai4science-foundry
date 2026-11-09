#!/bin/bash
# Quick checks on an ICE login node before submitting any Clay job.
# Takes a few seconds and submits nothing.
#
#   bash scripts/clay_preflight.sh        (from the Clay model folder)
#
# Checks: Clay model folder and branch, your Slurm account/QOS, that the CPU and GPU
# partitions used by the job scripts exist, that Slurm would accept each job
# (sbatch --test-only), the miniforge module, scratch space, and that the
# download hosts are reachable from the login node.

set -uo pipefail
# shellcheck source=scripts/clay_common.sh
source scripts/clay_common.sh
clay_require_repo_root
fail=0
ok()  { echo "  ok    $*"; }
bad() { echo "  FAIL  $*"; fail=1; }

echo "== Repo"
ok "branch $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD)"

echo "== Your Slurm associations (account|qos)"
sacctmgr -nP show assoc user="$USER" format=account,qos 2>/dev/null | sed 's/^/        /'
echo "        the jobs will use account ${SBATCH_ACCOUNT:-coc (from the scripts)}, qos ${SBATCH_QOS:-coc-ice (from the scripts)}"

echo "== Partitions the jobs will use"
for part in $(for job in scripts/clay_*.sbatch; do clay_job_partition "$job"; done | sort -u); do
    if sinfo -h -p "$part" -o '%P' 2>/dev/null | grep -q .; then ok "$part exists"; else bad "$part not found (see README, Setup step 2)"; fi
done

echo "== Would Slurm accept the jobs? (sbatch --test-only, nothing is submitted)"
for job in scripts/clay_*.sbatch; do
    clay_sbatch_opts "$job"
    if msg=$(sbatch --test-only ${CLAY_SBATCH_OPTS[@]+"${CLAY_SBATCH_OPTS[@]}"} "$job" 2>&1); then
        ok "$job: ${msg##*: }"
    else
        bad "$job: $msg"
    fi
done

echo "== Conda module"
if module -t avail miniforge/24.3.0-0 2>&1 | grep -q 'miniforge/24.3.0-0'; then ok "miniforge/24.3.0-0"; else bad "miniforge/24.3.0-0 not found (module spider miniforge)"; fi

echo "== Scratch ($CLAY_WORKDIR)"
if mkdir -p "$CLAY_WORKDIR"; then ok "writable"; else bad "cannot create $CLAY_WORKDIR"; fi
df -h "$CLAY_WORKDIR" | tail -1 | sed 's/^/        /'
echo "        needs about 13 GB: env 6.0 GB, checkpoint 4.9 GiB, teacher 1.2 GB, data 0.2 GB, plus pip/conda caches"

echo "== Download hosts (from the login node)"
for url in https://github.com/Clay-foundation/model https://pypi.org/simple/pip/ \
           https://download.pytorch.org/whl/cu121/torch/ "${NAIP_STAC_API}/collections/naip" \
           "https://huggingface.co/api/models/made-with-clay/Clay" "$REF_URL"; do
    (clay_check_url "$url") 2>/dev/null || bad "unreachable: $url"
done

echo
if [[ $fail -eq 0 ]]; then echo "PREFLIGHT PASSED"; else echo "PREFLIGHT FOUND PROBLEMS (see FAIL lines)"; fi
exit $fail
