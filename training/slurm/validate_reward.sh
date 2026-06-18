#!/usr/bin/env bash
#SBATCH --job-name=validate_reward
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=sapphire
#SBATCH --time=04:00:00
#
# Reward sanity check (plan Fix 2; preregistration §9). Scores
# known Mpro inhibitors + ZINC drug-like + junk against the composite
# verifier reward, then asserts >= 80% of knowns land in the top decile.
#
# Wall time at exhaustiveness=16, n_seeds=3 on 64 cores: roughly
# (12 + n_random + n_junk) × ~6 min / 64 ≈ 3 min per 32 mols.
# The default 12 + 1000 + 1000 = 2012 mols → ~3 hours.
#
# Usage:
#   sbatch training/slurm/validate_reward.sh
#   sbatch --export=N_RANDOM=2000,N_JUNK=2000 training/slurm/validate_reward.sh

set -euo pipefail

WORKSPACE_DIR=${WORKSPACE_DIR:-/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/molecule-rlvr}
APPTAINER_SIF=${APPTAINER_SIF:-$WORKSPACE_DIR/container/chem_rlvr.sif}
CONFIG=${CONFIG:-$WORKSPACE_DIR/training/configs/verifier.yaml}
KNOWN=${KNOWN:-$WORKSPACE_DIR/data/reference/known_mpro_inhibitors.csv}
ZINC=${ZINC:-$WORKSPACE_DIR/data/reference/zinc_random.smi}
OUT=${OUT:-$WORKSPACE_DIR/analysis/reward_validation.json}
N_RANDOM=${N_RANDOM:-1000}
N_JUNK=${N_JUNK:-1000}
THRESHOLD=${THRESHOLD:-0.8}

cd "$WORKSPACE_DIR"
mkdir -p logs analysis

# The container has its `/workspace` bind already set up for the trainer
# launchers; do the same here so absolute paths in verifier.yaml resolve.
singularity exec \
    --bind "$WORKSPACE_DIR:/workspace" \
    --env  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env  CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    "$APPTAINER_SIF" \
    python /workspace/verifier/validate_reward.py \
        --config /workspace/$(realpath --relative-to="$WORKSPACE_DIR" "$CONFIG") \
        --known  /workspace/$(realpath --relative-to="$WORKSPACE_DIR" "$KNOWN") \
        --zinc   /workspace/$(realpath --relative-to="$WORKSPACE_DIR" "$ZINC") \
        --n-random "$N_RANDOM" \
        --n-junk "$N_JUNK" \
        --top-decile-min-known-frac "$THRESHOLD" \
        --n-workers "${SLURM_CPUS_PER_TASK:-64}" \
        --out /workspace/$(realpath --relative-to="$WORKSPACE_DIR" "$OUT")
