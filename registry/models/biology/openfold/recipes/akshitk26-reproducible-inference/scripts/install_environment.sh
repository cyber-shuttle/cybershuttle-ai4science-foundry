#!/usr/bin/env bash
# One-time setup: conda env, pinned OpenFold source, CUTLASS, and OpenFold's
# CUDA extension. Run inside a GPU allocation (see README step 2).
# Safe to re-run: finished steps are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

load_modules

# Remember the PACE module compiler before conda puts its own gcc on PATH.
MODULE_CC="$(command -v gcc)"
MODULE_CXX="$(command -v g++)"
echo "host compiler for the CUDA build: $MODULE_CC ($("$MODULE_CC" -dumpversion))"

if [[ "$OPENFOLD_WORK" == "$HOME/scratch/"* && ! -d "$HOME/scratch" ]]; then
  die "$HOME/scratch not found; set OPENFOLD_WORK to a scratch directory (see .env.example)"
fi
mkdir -p "$OPENFOLD_WORK" "$OPENFOLD_WORK/tmp" "$OPENFOLD_WORK/conda-pkgs"
export TMPDIR="$OPENFOLD_WORK/tmp"
# Keep the conda package cache in scratch so it does not fill the home quota.
export CONDA_PKGS_DIRS="$OPENFOLD_WORK/conda-pkgs"

echo "== 1/4 conda env at $OPENFOLD_ENV"
if [[ ! -d "$OPENFOLD_ENV" ]]; then
  conda env create -f "$RECIPE_DIR/environment.yml" -p "$OPENFOLD_ENV"
else
  echo "already exists, skipping"
fi
activate_env
python -m pip install --no-build-isolation flash-attn==2.8.3.post1

echo "== 2/4 OpenFold source at $OPENFOLD_DIR"
if [[ ! -d "$OPENFOLD_DIR/.git" ]]; then
  git clone https://github.com/aqlaboratory/openfold.git "$OPENFOLD_DIR"
fi
git -C "$OPENFOLD_DIR" checkout --quiet "$OPENFOLD_COMMIT"
[[ "$(git -C "$OPENFOLD_DIR" rev-parse HEAD)" == "$OPENFOLD_COMMIT" ]] \
  || die "OpenFold is not at the pinned commit $OPENFOLD_COMMIT"

echo "== 3/4 OpenFold resources and CUTLASS"
cd "$OPENFOLD_DIR"
if [[ ! -s openfold/resources/stereo_chemical_props.txt ]]; then
  wget -N --no-check-certificate -P openfold/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
fi
if [[ ! -d cutlass/.git ]]; then
  git clone https://github.com/NVIDIA/cutlass --branch v3.6.0 --depth 1 cutlass
fi
[[ "$(git -C cutlass rev-parse HEAD)" == "$CUTLASS_COMMIT" ]] \
  || die "CUTLASS is not at the pinned commit $CUTLASS_COMMIT"
conda env config vars set CUTLASS_PATH="$OPENFOLD_DIR/cutlass" KMP_AFFINITY=none >/dev/null

echo "== 4/4 build OpenFold CUDA extension (takes several minutes)"
if (cd "$RECIPE_DIR" && python -c "import openfold" 2>/dev/null); then
  echo "extension already built, skipping"
else
  export CC="$MODULE_CC" CXX="$MODULE_CXX" CUDAHOSTCXX="$MODULE_CXX"
  export CUTLASS_PATH="$OPENFOLD_DIR/cutlass"
  # Build for every NVIDIA GPU type on ICE (V100, RTX 6000, A100, A40, L40S,
  # H100/H200), so the build works no matter which GPU a later job gets.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0}"
  export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
  MAX_JOBS=4 python setup.py install
fi

cd "$RECIPE_DIR"
python -c "import openfold; print('openfold import ok:', openfold.__file__)"
echo "install finished; next: bash scripts/download_artifacts.sh"
