#!/usr/bin/env bash
#SBATCH --job-name=chem_rlvr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# --- GPU pool (trainer + vLLM rollouts) ---
#SBATCH --hetjob
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --partition=kempner_h100
#SBATCH --time=48:00:00
#SBATCH hetjob
# --- CPU pool (Vina Ray actors) ---
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=sapphire
#SBATCH --time=48:00:00
#
# --- Chemistry-RLVR training launch (plan Fix 3) ----------------------------
# GPU hetgroup 0: training + vLLM
# CPU hetgroup 1: Vina docking actors
#
# Requires: `apptainer` in PATH, chem_rlvr.sif built, workspace prepped.
# Tune --partition, --time and node counts to your cluster.

set -euo pipefail

WORKSPACE_DIR=${WORKSPACE_DIR:-/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/molecule-rlvr}
APPTAINER_SIF=${APPTAINER_SIF:-$WORKSPACE_DIR/container/chem_rlvr.sif}
CONFIG=${CONFIG:-$WORKSPACE_DIR/training/configs/grpo_7b.yaml}

cd "$WORKSPACE_DIR"
mkdir -p logs checkpoints

# --- Resolve the head node IP -----------------------------------------------
HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -n 1)
export RAY_HEAD_IP=$(srun --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_HOST" hostname -I | awk '{print $1}')
export RAY_PORT=6379

echo "Ray head will bind: $RAY_HEAD_IP:$RAY_PORT"
export APPTAINER_SIF WORKSPACE_DIR

# --- NCCL / network hygiene (Kempner FASRC uses ibN; override as needed) ----
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

# --- Start Ray workers in background (head + GPU + CPU pools) ---------------
srun --het-group=0 --nodes="$SLURM_JOB_NUM_NODES_HET_GROUP_0" --ntasks-per-node=1 \
    bash training/slurm/ray_cluster.sh &
RAY_GPU_PID=$!

srun --het-group=1 --nodes="$SLURM_JOB_NUM_NODES_HET_GROUP_1" --ntasks-per-node=1 \
    bash training/slurm/ray_cluster.sh &
RAY_CPU_PID=$!

# Give the cluster a moment to form
sleep 30

# --- Trainer launches from the head node only -------------------------------
ssh -o StrictHostKeyChecking=no "$HEAD_HOST" "cd $WORKSPACE_DIR && \
    apptainer exec --nv \
        --bind $WORKSPACE_DIR:/workspace \
        --env CHEM_VERIFIER_CONFIG=/workspace/training/configs/verifier.yaml \
        --env PYTHONPATH=/workspace/verifier \
        --env RAY_ADDRESS=$RAY_HEAD_IP:$RAY_PORT \
        $APPTAINER_SIF \
        python -m openrlhf.cli.train_ppo_ray \
            --config /workspace/$(basename $CONFIG)"

# Shut down Ray when trainer exits
kill $RAY_GPU_PID $RAY_CPU_PID || true
wait || true
