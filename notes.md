
***

# Project: Chemistry-RLVR (Reasoning vs. Standard LLMs)
**Target:** SARS-CoV-2 Mpro Inhibitor Discovery  
**Infrastructure:** Slurm Cluster (H100/H200 GPUs)  
**Stack:** OpenRLHF (GRPO) + Ray + Apptainer (Singularity)

---

## Phase 1: Environment & Sandbox Construction
The goal is to create a portable, high-performance container that includes both the deep learning stack and the chemical simulation suite.

### 1.1 Apptainer (Singularity) Definition
Create a file named `chem_rlvr.def`:

```bash
Bootstrap: docker
From: nvidia/cuda:12.4.1-devel-ubuntu22.04

%post
    apt-get update && apt-get install -y python3-pip python3-dev wget git openbabel
    
    # Install Chemical Suites
    pip install rdkit-pypi meeko scipy numpy
    
    # Install AutoDock Vina
    wget https://github.com/CCSB-VSR/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 -O /usr/local/bin/vina
    chmod +x /usr/local/bin/vina

    # Install RL Framework (OpenRLHF preferred for GRPO stability)
    pip install "openrlhf[vllm,ray]"
    pip install flash-attn --no-build-isolation

%environment
    export PATH="/usr/local/bin:$PATH"
    export PYTHONUNBUFFERED=1
```

**Build Command:**
```bash
apptainer build chem_rlvr.sif chem_rlvr.def
```

---

## Phase 2: Pilot Study (Baseline Evaluation)
Before RL, establish the performance floor for **Qwen-2.5-32B-Instruct** (Standard) vs. **DeepSeek-R1-Distill-Qwen-32B** (Reasoning).

### 2.1 The Multi-Turn Manual Loop
1.  **Prompting:** Provide the Mpro pocket description and ask for a SMILES string.
2.  **Verification:** Pass the SMILES through a Python script (using the `.sif` image) to get:
    * RDKit Validity
    * Lipinski Rule check
    * AutoDock Vina score ($kcal/mol$)
3.  **Feedback:** Re-insert these metrics into the prompt for Turn 2.
4.  **Metric:** Compare how many turns it takes for each model to reach a docking score $< -7.0$.

---

## Phase 3: RLVR Training via GRPO
Using **OpenRLHF** to train the models on a cluster of H100s. GRPO is ideal here because it doesn't require a separate Critic model, saving VRAM for the 32B model parameters.

### 3.1 Slurm Launch Script (`train_rlvr.sh`)
```bash
#!/bin/bash
#SBATCH --job-name=chem_rlvr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=h100

# Start Ray Head Node
srun --nodes=1 --ntasks=1 --exclusive ray start --head --port=6379 &

# Launch OpenRLHF GRPO
apptainer exec --nv chem_rlvr.sif openrlhf_train \
    --policy_path /models/Qwen-32B-Instruct \
    --dataset /data/mpro_questions.jsonl \
    --save_path /checkpoints/chem_agent \
    --strategy deepspeed_stage_3 \
    --learning_rate 1e-6 \
    --num_episodes 500 \
    --rollout_batch_size 512 \
    --micro_train_batch_size 2 \
    --reward_functions chemistry_verifier \
    --grpo
```

### 3.2 The Verifier Logic (`chemistry_verifier.py`)
This script resides inside the container and acts as the RLVR feedback source.

```python
def reward_fn(queries, responses):
    rewards = []
    for resp in responses:
        smiles = extract_smiles(resp)
        mol = Chem.MolFromSmiles(smiles)
        
        if not mol:
            rewards.append(-1.0) # Invalid Chemistry
            continue
            
        # 1. Check Lipinski
        if not passes_lipinski(mol):
            rewards.append(0.1) # Valid but not drug-like
            continue
            
        # 2. Run Vina Docking
        score = run_vina_docking(mol, target_pdb="6LU7")
        
        # 3. Calculate Reward
        # Reward increases as docking score becomes more negative
        reward = max(0, -score - 5.0) 
        rewards.append(reward)
    return rewards
```

---

## Phase 4: Analysis & Hypotheses Testing
The final phase is the comparative analysis of the two training runs.

### Key Metrics to Track:
1.  **Convergence Rate:** Does the Reasoning model reach the "High Affinity" threshold in fewer training steps?
2.  **SMILES Validity Ratio:** Track the percentage of "hallucinated" or invalid molecules generated over the course of training.
3.  **Qualitative Trace Analysis:** Extract `<thought>` blocks from the R1 model. 
    * *Target:* Does the model mention specific residues (e.g., "His41") when justifying an edit?
4.  **Synthesizability Trade-off:** Use the SAscore to see if the Reasoning model can maintain high affinity while keeping the molecule easy to synthesize.



---

## Research Hypothesis for the Paper
> "We posit that RLVR-trained models with native reasoning capabilities (CoT) internalize the rules of structural biology more effectively than standard policy-gradient models. By utilizing a multi-turn feedback loop with a physics-based verifier (AutoDock), the reasoning model learns to perform **spatial error correction**, leading to a $25\%+$ increase in successful lead generation within the Mpro pocket environment."


