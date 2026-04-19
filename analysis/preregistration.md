# Pre-registration — Chemistry-RLVR Mpro Experiment

**Freeze date:** commit this file before the first RL training step.
**Plan reference:** `/n/home07/bdesinghu/.claude/plans/dreamy-inventing-chipmunk.md` (Fix 5, Fix 7, Fix 8).

---

## 1. Hypothesis

RLVR-trained reasoning LLMs internalize structural-biology rules more effectively than equivalent non-reasoning LLMs when trained with a physics-based (AutoDock Vina) verifier, measured by hit rate on a held-out prompt set.

## 2. Arms

| Arm | Model | Mode | Purpose |
|-----|-------|------|---------|
| A | Qwen3-14B (final: Qwen3-32B) | `enable_thinking=False` | Non-reasoning baseline |
| B | Qwen3-14B (final: Qwen3-32B) | `enable_thinking=False` + "think step-by-step" prompt | Elicited CoT only |
| C | Qwen3-14B (final: Qwen3-32B) | `enable_thinking=True` | Trained reasoning |

External reference arms (zero-shot eval, no RL training):
- R1: `DeepSeek-R1-Distill-Qwen-32B` — cross-family reasoning reference
- R2: `ether0-24B` — chemistry-RL reference
- **SFT+Rerank**: Stage-0 SFT checkpoint, sample 100k, re-rank. **This is the real baseline RL must beat.**

## 3. Training protocol

- Same Stage-0 SFT pipeline (plan Fix 8.1) for A/B/C.
- Same prompts (plan Fix 6), same reward config (plan Fix 2), same compute budget.
- ≥ 3 training seeds per arm.
- Curriculum (plan Fix 8.4): dock threshold shifts at steps 100 and 300.
- Replay (plan Fix 8.3) enabled.

## 4. Metrics

### Primary
- **Hit rate**: fraction of generated molecules on the held-out eval set (60 prompts, 8 samples per prompt, temperature 0.7) satisfying:
  - Vina dock score < −7.0 kcal/mol (mean of 3 seeds, receptor 7L13)
  - QED > 0.5
  - SA < 5.0
  - No PAINS matches
- Report mean ± 95% bootstrap CI over seeds × prompts.

### Secondary
- **Validity ratio**: fraction of parseable SMILES per 1k-step window.
- **Scaffold diversity**: unique Murcko scaffolds per 512-rollout training batch (monitor for collapse).
- **Turns-to-threshold**: median turns until a rollout crosses dock < −7 kcal/mol in multi-turn trajectories. Censor at 5 turns.
- **CoT-grounding** (Arms B and C only): per-trace count of correct mentions of {His41, Cys145, Glu166, Gln189}. Judge: fixed LLM-as-judge prompt + keyword rule; see `analysis/cot_judge_prompt.md` (to be committed).
- **Novelty**: median ECFP4 Tanimoto to the nearest training seed; target ≤ 0.4.

## 5. Statistical tests

- **Hit-rate comparison** (primary): two-proportion z-test on aggregated seeds; bootstrap 95% CI on the ratio `hit_rate(C) / hit_rate(A)`.
- **Docking-score distributions**: Mann–Whitney U.
- **CoT-grounding count**: Poisson regression with arm as the categorical covariate.
- Multiple-comparison adjustment: Benjamini–Hochberg at α = 0.05 across secondary metrics.

## 6. Stopping rule

The paper's hypothesis is supported iff **hit_rate(C) / hit_rate(A) ≥ 1.25** with a 95% bootstrap CI lower bound > 1.0, across ≥ 3 seeds, AND the C arm beats the SFT+Rerank baseline on per-call efficiency (hit rate at 10k oracle calls).

If hit_rate(C) / hit_rate(A) ∈ [1.0, 1.25): report as a null result; discuss in the paper.

If C does not beat SFT+Rerank: RL is not contributing; rewrite the story around what the reward signal reveals, not around RL's contribution.

## 7. Ablations (run on the best arm only, 7B scale)

| ID | Change | Isolates |
|----|--------|----------|
| a1 | `no-stage0` — skip SFT | Value of warm-start |
| a2 | `no-cot-sft` — drop R1-CoT Stage 0b | *Trained* vs. *elicited* CoT |
| a3 | `no-replay` — disable augmented memory | Value of experience replay |
| a4 | `no-curriculum` — fixed −7 sigmoid center | Value of staged threshold |
| a5 | `no-leadopt` — all prompts de novo | Value of seeded prompts |
| a6 | `no-multiturn` — single-turn rollouts | Value of feedback loop |
| a7 | `rank-reward` — rank-based dock reward | Noise-robustness of rank form |
| a8 | `sft-plus-rerank` — no RL at all | Whether RL adds anything |

## 8. Data and code freeze

Pinned artifacts (commit SHA before RL step 1):
- `data/receptors/7L13/receptor.pdbqt` + `receptor.config` + `redock_rmsd.txt`
- `data/prompts/mpro_prompts.jsonl` (240 train / 60 eval)
- `data/reference/known_mpro_inhibitors.csv`
- `training/configs/verifier.yaml`
- `verifier/chemistry_verifier.py` + `reward_components.py`

No changes to these files after freeze without an amendment to this document.

## 9. What would falsify this?

- If reward validation (`verifier/validate_reward.py`) fails the top-decile test, the experiment does not start.
- If 7B does not beat SFT+Rerank after 100 training steps with replay+curriculum, scale to 14B/32B is not authorized.
- If C does not beat A by ≥ 1.25× hit-rate ratio across ≥ 3 seeds, the reasoning hypothesis is not supported.

## 10. Deviations log

Record any in-training deviation from this document here with date, reason, and
new measurement plan. Do not silently modify protocol.
