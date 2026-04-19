# molecule-rlvr — Chemistry-RLVR on SARS-CoV-2 Mpro

GRPO-trained reasoning-vs-standard LLMs for SARS-CoV-2 main protease (Mpro)
inhibitor discovery, with a physics-based (AutoDock Vina) verifier as the
reward. Goal: test whether reasoning LLMs internalize structural-biology rules
more effectively than non-reasoning models when trained with a verifiable
chemistry reward.

The research design is documented in the approved plan at
`~/.claude/plans/dreamy-inventing-chipmunk.md` — 9 fixes against the original
`notes.md`, organized in execution-order with acceptance tests that gate each
stage. This README is the operational index; the plan is the source of truth.

---

## Layout

```
molecule-rlvr/
├── notes.md                          original research notes (kept for history)
├── container/                        Apptainer image + smoke test  (Fix 1)
│   ├── chem_rlvr.def
│   ├── build.sh                      build + smoke-test wrapper
│   └── smoke_test.sh                 imports + vina --version + prepare_receptor4
├── verifier/                         composite RLVR reward         (Fix 2)
│   ├── reward_components.py          PAINS/reactive, sigmoid dock, QED, SA, strain, Tanimoto
│   ├── chemistry_verifier.py         composite reward + curriculum + rank-dock alt + multi-turn bonus
│   ├── verifier_config_loader.py
│   └── validate_reward.py            three-group ranking test; gates RL start
├── data/
│   ├── receptors/7L13/               non-covalent Mpro receptor    (Fix 4)
│   │   └── prep_receptor.sh          reduce → prepare_receptor4 → grid box → redock < 2 Å gate
│   ├── prompts/                      prompt dataset                (Fix 6 + 8.2)
│   │   ├── schema.md
│   │   ├── examples.jsonl
│   │   └── generate_prompts.py       300 prompts, ≥80% lead-opt
│   └── reference/
│       └── known_mpro_inhibitors.csv 12 hand-curated actives
├── training/
│   ├── slurm/                        hetjob: GPU pool + CPU Vina pool  (Fix 3)
│   │   ├── train_rlvr.sh
│   │   └── ray_cluster.sh
│   ├── configs/
│   │   ├── verifier.yaml             reward weights + curriculum schedule
│   │   ├── sft_stage0a.yaml          Stage 0a — chemistry LM warm-start
│   │   └── grpo_7b.yaml              OpenRLHF GRPO config
│   └── tokenizer/
│       └── extend_vocab.py           atom-in-SMILES vocabulary extension (Fix 9.3)
└── analysis/
    └── preregistration.md            frozen metrics, ablations, stopping rule (Fix 5 + 7)
```

---

## Execution order (gated)

Each stage's acceptance test gates the next. Don't spend GPU-hours on stage *n*
if stage *n−1* hasn't passed.

| # | Stage | Where | Acceptance test |
|---|-------|-------|------------------|
| 1 | Build container | `container/build.sh` | `smoke_test.sh` passes (all chemistry imports + `vina --version` + `prepare_receptor4.py -h`) |
| 2 | Prepare receptor | `data/receptors/7L13/prep_receptor.sh` | `redock_rmsd.txt` reports < 2.0 Å |
| 3 | Validate reward | `verifier/validate_reward.py` | ≥ 80 % of known Mpro inhibitors land in the top decile vs. ZINC + junk |
| 4 | Generate prompts | `data/prompts/generate_prompts.py` | 240 train + 60 eval, ≥ 80 % lead-opt |
| 5 | Compute topology | `training/slurm/*.sh` | Measured docks/sec × CPU count ≥ rollouts-per-step ÷ target-step-seconds |
| 6 | Stage 0 SFT | `training/configs/sft_stage0a.yaml` (→ 0b, 0c) | Checkpoints saved; validity ratio > 0.8 on eval prompts |
| 7 | VS baseline | *not yet implemented* | Hit rate from SFT + re-rank recorded; this is what RL must beat |
| 8 | Pre-register | `analysis/preregistration.md` | Committed to git **before** RL step 1 |
| 9 | Pilot | *not yet implemented* | Per-model turns-to-threshold curves on 20 prompts |
| 10 | RL 7B | `sbatch training/slurm/train_rlvr.sh` (CONFIG=grpo_7b.yaml) | Reward ↑, validity ≥ 0.8, diversity stable, beats VS baseline |
| 11 | RL 14B / 32B | swap pretrain in config | Same as 10 at scale |
| 12 | Analysis & ablations | preregistration §7 | Hit-rate ratio C/A ≥ 1.25 with 95 % CI LB > 1.0 over ≥ 3 seeds |

---

## Quickstart

### Build the container
```bash
cd container
APPTAINER_TMPDIR=/n/netscratch/kempner_dev/Lab/bdesinghu/tmp ./build.sh
```

### Prepare the receptor
```bash
apptainer exec --nv container/chem_rlvr.sif \
    bash data/receptors/7L13/prep_receptor.sh
```

### Validate the reward (before anything else)
```bash
# Fetch ~2000 random ZINC drug-like SMILES first (see data/reference/README)
apptainer exec --nv container/chem_rlvr.sif \
    python verifier/validate_reward.py \
        --config training/configs/verifier.yaml \
        --known  data/reference/known_mpro_inhibitors.csv \
        --zinc   data/reference/zinc_random.smi \
        --n-random 1000 --n-junk 1000 \
        --out    analysis/reward_validation.json
```
Non-zero exit = reward is broken. Don't start RL.

### Generate the prompt dataset
```bash
apptainer exec container/chem_rlvr.sif \
    python data/prompts/generate_prompts.py \
        --known-inhibitors data/reference/known_mpro_inhibitors.csv \
        --zinc-smi         data/reference/zinc_random.smi \
        --out              data/prompts/mpro_prompts.jsonl
```

### Extend the tokenizer (run once; feeds Stage 0a SFT)
```bash
apptainer exec --nv container/chem_rlvr.sif \
    python training/tokenizer/extend_vocab.py \
        --base-model Qwen/Qwen3-8B \
        --out-dir    training/tokenizer/qwen3_atomInSmiles
```

### Launch RL (after Stages 1–8 pass)
```bash
cd training && sbatch slurm/train_rlvr.sh
```

---

## Model lineup (plan Fix 9)

| Role | Model | Rationale |
|------|-------|-----------|
| Iteration | Qwen3-8B (thinking mode on/off) | Fast debug, same family |
| Comparison arms A/B/C | Qwen3-14B → Qwen3-32B | Dual-mode toggle eliminates distillation confound |
| External reference | DeepSeek-R1-Distill-Qwen-32B (zero-shot) | Cross-family generalization |
| Chemistry-RL reference | ether0-24B (zero-shot) | Shows we're not below field baseline |

---

## What's not here yet

Per the current task list, these are the next planned additions:
- SFT Stage 0a/0b/0c training scripts (warm-start pipeline, plan Fix 8.1)
- Virtual-screening baseline (plan Fix 8.6) — the real yardstick for RL
- Multi-turn pilot harness (Phase 2 of `notes.md`)
- ZINC drug-like subset downloader (currently external prereq)

See the plan file for the rationale and approved design.

---

## References

Key prior art that shaped the design, all cited in the plan:

- **ether0** (FutureHouse, 2025) — multi-stage SFT → specialist-GRPO → merge recipe.
- **Augmented Memory** (Schaub et al., 2024) — SMILES-augmented experience replay; ~2× sample efficiency.
- **Mol-AIR** (2024) — intrinsic rewards for sparse molecular RL.
- **SHARP / FragDockRL / ReACT-Drug** (2025) — fragment / reaction-template action spaces (v2 direction).
- **MEDICO** (2025) — multiview graph generative model on Mpro; external benchmark.
