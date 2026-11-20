# Shared settings; sourced by setup.sh and run_inference.sh.
# Large artifacts live under OF_ROOT (scratch), outside git.
export OF_ROOT=${OF_ROOT:-$HOME/scratch/openfold-repro}
export OF_REPO=$OF_ROOT/openfold      # upstream clone at OF_COMMIT
export OF_ENV=$OF_ROOT/env            # conda env from ../environment.yml
export OF_PARAMS=$OF_ROOT/params      # OpenFold + AF2 weights
export OF_DATA=$OF_ROOT/data          # inputs from prepare_data.py
export OF_RUNS=$OF_ROOT/runs          # outputs: <set>/<weight set>/
export OF_COMMIT=e938c184a291bf053af3b14c1e3e8bb29aee57e2   # tag v2.2.0
export MINIFORGE_VERSION=26.5.3-0
export MAX_TEMPLATE_DATE=2021-10-10   # OpenFold training template cutoff
export KMP_AFFINITY=none              # as upstream's installer sets
# Caches off the 30 GB home quota
export CONDA_PKGS_DIRS=$OF_ROOT/miniforge3/pkgs PIP_CACHE_DIR=$OF_ROOT/cache/pip TRITON_CACHE_DIR=$OF_ROOT/cache/triton

activate_env() {
    unset GCC_ROOT  # set by PACE gcc module; breaks conda g++ ("cannot execute 'cc1plus'")
    source "$OF_ROOT/miniforge3/etc/profile.d/conda.sh"
    conda activate "$OF_ENV"
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
}
