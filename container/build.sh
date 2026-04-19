#!/usr/bin/env bash
# Build chem_rlvr.sif on a login node with apptainer available.
# Use --fakeroot if the cluster requires it; check `apptainer --version` first.
set -euo pipefail

cd "$(dirname "$0")"

SIF=${SIF:-chem_rlvr.sif}
DEF=${DEF:-chem_rlvr.def}

if [[ ! -f "$DEF" ]]; then
    echo "Missing $DEF" >&2
    exit 1
fi

# Build (needs network + ~25 GB free in $APPTAINER_TMPDIR).
# On Kempner/FASRC: set APPTAINER_TMPDIR to scratch before invoking.
: "${APPTAINER_TMPDIR:?set APPTAINER_TMPDIR to a scratch path with >25GB free}"

echo "Building $SIF from $DEF (tmpdir=$APPTAINER_TMPDIR)"
apptainer build --force "$SIF" "$DEF"

echo "Running smoke test..."
apptainer exec --nv "$SIF" bash smoke_test.sh
