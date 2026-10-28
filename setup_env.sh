#!/usr/bin/env bash
# Build a conda environment for this repository.
#
#   ./setup_env.sh [claims|full] [--name NAME]
#
# claims (default)  A light CPU environment for checking the claims from the committed result tables:
#                   Python 3.12 from conda-forge, the pinned numpy / scipy / pandas / matplotlib / pyyaml / pillow /
#                   h5py / opencv-python-headless / pytest versions from environment.yml (no PyTorch, no models),
#                   the package itself (pip --no-deps -e .), then runs `python claims.py`.
#                   Default name: measured-space-claims.
# full              The full environment the results were produced with: `conda env create -f environment.yml`
#                   (PyTorch 2.14.1, a CUDA 13.0 build) plus requirements-nodeps.txt (MoGe-2).
#                   Default name: measured-space. The Cubify Anything and Sparse4D environments are separate
#                   and documented in REPRODUCE.md.
#
# The script only creates a new conda environment under your conda installation. It needs no sudo, changes nothing
# outside that environment, refuses to run if an environment of that name exists, and never removes anything. If a
# step fails, the partial environment is left in place; remove it with `conda env remove -n NAME`.
set -euo pipefail

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

MODE="claims"
NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        claims|full) MODE="$1"; shift ;;
        --name) NAME="${2:?--name needs a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -z "$NAME" ]]; then
    if [[ "$MODE" == "claims" ]]; then NAME="measured-space-claims"; else NAME="measured-space"; fi
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found on PATH; install Miniconda or Miniforge first." >&2
    exit 1
fi
if conda env list | awk '{print $1}' | grep -qx "$NAME"; then
    echo "a conda environment named '$NAME' already exists; choose another with --name, or remove it yourself." >&2
    exit 1
fi

# The version pinned in environment.yml for one PyPI package.
pin() {
    local spec
    spec="$(grep -E "^[[:space:]]*-[[:space:]]+$1==" environment.yml | head -n 1 | sed -E 's/^[[:space:]]*-[[:space:]]+//')"
    if [[ -z "$spec" ]]; then
        echo "no pin for $1 in environment.yml" >&2
        exit 1
    fi
    echo "$spec"
}

start=$(date +%s)
if [[ "$MODE" == "claims" ]]; then
    PKGS=()
    for p in numpy scipy pandas pillow pyyaml h5py opencv-python-headless matplotlib pytest; do
        PKGS+=("$(pin "$p")")
    done
    echo "== creating '$NAME' (Python 3.12, conda-forge only)"
    conda create -y -n "$NAME" --override-channels -c conda-forge "python=3.12" pip
    echo "== installing ${PKGS[*]}"
    conda run --no-capture-output -n "$NAME" python -m pip install --no-cache-dir "${PKGS[@]}"
    echo "== installing the measured-space package (no dependencies)"
    conda run --no-capture-output -n "$NAME" python -m pip install --no-cache-dir --no-deps -e .
    echo "== checking the claims"
    set +e
    conda run --no-capture-output -n "$NAME" python claims.py
    status=$?
    set -e
    echo "== done in $(( $(date +%s) - start )) s; claims.py exited with $status (0: every claim passed)"
    echo "   activate with: conda activate $NAME"
    exit "$status"
else
    echo "== creating '$NAME' from environment.yml (PyTorch 2.14.1, CUDA 13.0 build; several GB)"
    conda env create -n "$NAME" -f environment.yml
    echo "== installing requirements-nodeps.txt (MoGe-2, pinned commits)"
    conda run --no-capture-output -n "$NAME" python -m pip install --no-cache-dir --no-deps -r requirements-nodeps.txt
    echo "== done in $(( $(date +%s) - start )) s; activate with: conda activate $NAME"
fi
