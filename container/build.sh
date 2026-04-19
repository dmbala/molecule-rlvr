#!/usr/bin/env bash
# Build chem_rlvr.sif.
# Works with either `apptainer` (preferred) or `singularity` (Singularity-CE).
# On Kempner/FASRC: singularity-ce is available at /usr/bin/singularity; no
# apptainer binary is installed.
set -euo pipefail

cd "$(dirname "$0")"

SIF=${SIF:-chem_rlvr.sif}
DEF=${DEF:-chem_rlvr.def}

if [[ ! -f "$DEF" ]]; then
    echo "Missing $DEF" >&2
    exit 1
fi

# Pick the container runtime.
if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_CLI=apptainer
    TMPDIR_VAR=APPTAINER_TMPDIR
    CACHE_VAR=APPTAINER_CACHEDIR
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_CLI=singularity
    TMPDIR_VAR=SINGULARITY_TMPDIR
    CACHE_VAR=SINGULARITY_CACHEDIR
else
    echo "Neither apptainer nor singularity found on PATH" >&2
    exit 1
fi

# Build needs ~25 GB free in the tmpdir. Point it at scratch if nothing is set.
if [[ -z "${!TMPDIR_VAR:-}" ]]; then
    default_tmp=/n/netscratch/kempner_dev/Lab/bdesinghu/tmp/singularity-build
    mkdir -p "$default_tmp"
    export $TMPDIR_VAR=$default_tmp
fi
if [[ -z "${!CACHE_VAR:-}" ]]; then
    default_cache=/n/netscratch/kempner_dev/Lab/bdesinghu/tmp/singularity-cache
    mkdir -p "$default_cache"
    export $CACHE_VAR=$default_cache
fi

echo "Container runtime: $CONTAINER_CLI"
echo "Tmpdir ($TMPDIR_VAR): ${!TMPDIR_VAR}"
echo "Cache  ($CACHE_VAR):  ${!CACHE_VAR}"

echo "Building $SIF from $DEF ..."
BUILD_ARGS=(build --force)
# Use --fakeroot unless the user explicitly opts out (e.g. they have real root).
if [[ "${NO_FAKEROOT:-0}" != "1" ]]; then
    BUILD_ARGS+=(--fakeroot)
fi
"$CONTAINER_CLI" "${BUILD_ARGS[@]}" "$SIF" "$DEF"

echo "Running smoke test ..."
"$CONTAINER_CLI" exec --nv "$SIF" bash smoke_test.sh
