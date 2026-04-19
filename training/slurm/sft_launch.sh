#!/usr/bin/env bash
#SBATCH --job-name=chem_sft
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --partition=kempner_h100
#SBATCH --time=24:00:00
#
# Single-node SFT launcher for Stages 0a / 0b / 0c (plan Fix 8.1).
# Usage: sbatch --export=CONFIG=training/configs/sft_stage0a.yaml training/slurm/sft_launch.sh

set -euo pipefail

WORKSPACE_DIR=${WORKSPACE_DIR:-/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/molecule-rlvr}
APPTAINER_SIF=${APPTAINER_SIF:-$WORKSPACE_DIR/container/chem_rlvr.sif}
CONFIG=${CONFIG:?set CONFIG=training/configs/sft_stageXY.yaml}

cd "$WORKSPACE_DIR"
mkdir -p logs checkpoints

NPROC=${SLURM_GPUS_ON_NODE:-4}

apptainer exec --nv \
    --bind "$WORKSPACE_DIR:/workspace" \
    --env PYTHONPATH=/workspace/verifier:/workspace/training/sft:/opt/sascorer \
    "$APPTAINER_SIF" \
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
        /workspace/training/sft/train_sft.py \
        --config "/workspace/$CONFIG"
