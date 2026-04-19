#!/usr/bin/env bash
# Bring up a hetjob Ray cluster: GPU pool (trainer + vLLM rollouts) + CPU pool (Vina workers).
# Invoked by train_rlvr.sh; not run directly.
#
# Expects:
#   $RAY_HEAD_IP          — IP of the Ray head node (resolved in train_rlvr.sh)
#   $RAY_PORT             — 6379
#   $APPTAINER_SIF        — path to chem_rlvr.sif
#   $WORKSPACE_DIR        — bind-mounted as /workspace inside the container
#   $SLURM_HET_GROUP_ID   — 0 for GPU pool, 1 for CPU pool

set -euo pipefail

ROLE=${SLURM_HET_GROUP_ID:-0}
HOST_IP=$(hostname -I | awk '{print $1}')

BIND_ARGS=(--bind "$WORKSPACE_DIR:/workspace")

if [[ "$ROLE" == "0" ]]; then
    if [[ "$HOST_IP" == "$RAY_HEAD_IP" ]]; then
        echo "[ray_cluster] HEAD on $HOST_IP"
        apptainer exec --nv "${BIND_ARGS[@]}" "$APPTAINER_SIF" \
            ray start --head \
                --port="$RAY_PORT" \
                --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
                --num-cpus="${SLURM_CPUS_PER_TASK:-16}" \
                --dashboard-host=0.0.0.0 \
                --block
    else
        echo "[ray_cluster] GPU worker on $HOST_IP joining $RAY_HEAD_IP:$RAY_PORT"
        apptainer exec --nv "${BIND_ARGS[@]}" "$APPTAINER_SIF" \
            ray start --address="$RAY_HEAD_IP:$RAY_PORT" \
                --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
                --num-cpus="${SLURM_CPUS_PER_TASK:-16}" \
                --block
    fi
else
    echo "[ray_cluster] CPU worker on $HOST_IP joining $RAY_HEAD_IP:$RAY_PORT"
    apptainer exec "${BIND_ARGS[@]}" "$APPTAINER_SIF" \
        ray start --address="$RAY_HEAD_IP:$RAY_PORT" \
            --num-cpus="${SLURM_CPUS_PER_TASK:-32}" \
            --num-gpus=0 \
            --resources='{"vina": 32}' \
            --block
fi
