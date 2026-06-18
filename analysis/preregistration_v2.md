# Pre-registration Amendment v2 — Arm C-PC (Proposer-Critic Self-Play)

**Freeze date:** commit this file before the first RL training step under
the C-PC configuration. Do not modify after freeze.
**Plan reference:** `~/.claude/plans/what-do-you-think-zesty-sifakis.md`.
**Supersedes:** nothing in `analysis/preregistration.md` — this amendment
**adds** a fourth arm; the three-arm comparisons in v1 remain unchanged.

---

## 1. Hypothesis (additional)

Explicit two-role proposer–critic decomposition on top of an
`enable_thinking=True` reasoning model improves hit rate beyond what a
single trained-thinking call achieves, at equal AutoDock Vina oracle-call
budget. The expected effect is non-trivially additive to thinking-mode
because the critic sees its parent's scalar reward breakdown — information
that a single-call thinker cannot reach.

## 2. New arm

| Arm | Model | Mode | Rollout shape | Oracle calls / prompt |
|-----|-------|------|----------------|------------------------|
| C-base | Qwen3-{8B → 14B → 32B} | `enable_thinking=True` | 8 single-call rollouts per GRPO group | 8 |
| **C-PC** | Same model + SFT init | `enable_thinking=True` | **4 proposer + 4 critic** rollouts per GRPO group (same weights, role selected by system prompt) | 8 |

C-base shares its SFT initialization with arm C in `preregistration.md` —
this amendment changes only the rollout shape, not the model, the prompts,
the reward, the optimizer, or the per-prompt oracle-call budget.

## 3. Critic reward

`r_critic(s2 | s1) = r(s2) + α · max(0, r(s2) − r(s1))`, with `α = 0.5`.

The bonus is applied **only** when both gates pass:

1. `dock_kcal(s2) − dock_kcal(s1) ≥ 0.5 kcal/mol`
   (mitigates Vina seed-lottery exploitation, ~0.5–1 kcal/mol stochasticity
   at `exhaustiveness=16, n_seeds=3`).
2. `Tanimoto(s1, s2) ≤ 0.7` (ECFP4, radius 2, 2048 bits)
   (prevents trivial-edit reward farming).

GRPO group-norm advantages are computed **within role**: the proposer-4 and
critic-4 of each prompt get separate baselines, so cross-role reward
asymmetry does not dominate.

## 4. Seed source

Initial seeds for lead-optimization prompts now include a frozen DiffSBDD
seed bank (`data/seeds/diffsbdd_7L13.jsonl`) at ~40 % of lead-opt slots.
The remaining 60 % stays with `known_inhibitor_fragment` and `zinc_random`
provenances exactly as in v1. The seed bank is pre-Vina-scored,
drug-likeness-filtered, and CrossDocked-leakage-filtered
(ECFP4 Tanimoto ≤ 0.7 to the bundled reference list).

## 5. Pre-registered primary comparison

**`hit_rate(C-PC) / hit_rate(C-base) ≥ 1.15` with 95 % bootstrap CI lower
bound > 1.0**, on the same 60-prompt eval split as Arm C (8 samples per
prompt, T=0.7), across ≥ 3 seeds. Hit-rate criteria are identical to
preregistration v1 §4 (dock < −7 kcal/mol AND QED > 0.5 AND SA < 5 AND no
PAINS).

If `1.0 ≤ ratio < 1.15`: report as a null result for C-PC; arm C remains
the recommended deployment.

If `ratio < 1.0` with significant CI: C-PC is harmful at this budget;
drop it from the experiment matrix and discuss in the paper.

## 6. Pilot acceptance gates (Qwen3-8B, 50 steps × 3 seeds)

Escalation from the 8B pilot to 14B / 32B C-PC is conditional on **all four**
gates passing:

| Gate | Threshold |
|---|---|
| Hit-rate ratio | `hit_rate(C-PC)/hit_rate(C-base) ≥ 1.15` mean over seeds; 95 % CI LB > 1.0 |
| Validity | C-PC parse rate ≥ 0.75 |
| Diversity | Murcko-scaffold count / 512-rollout batch ≥ 0.6× C-base AND top-1 Murcko ≤ 0.15 |
| Walltime | C-PC step ≤ 1.25× C-base |

If the ratio gate passes at 14B but with CI LB < 1.0, repeat the 14B run
with more seeds before escalating to 32B.

## 7. Secondary diagnostics (Amendment v2 only)

- **Critic credit rate** — fraction of critic rollouts whose `α · ReLU`
  bonus actually fired (both gates passed). Target: 0.15–0.40 across the
  training run. Outside that range signals reward-hacking (high) or
  critic-template collapse (low).
- **Critic Tanimoto distribution** — histogram of `Tanimoto(s1, s2)` over
  parsed critic rollouts; report mean ± std per save_step.
- **Cross-role evaluation parity** — at every `save_step`, evaluate the
  trained C-PC checkpoint **also** in single-call mode on the eval split;
  refuse to declare C-PC the winner unless C-PC mode wins *and* the
  single-call eval does not regress relative to C-base.
- **Think-block share** — `len(<think>...</think>) / len(full_response)`
  median per save_step on critic responses. Below 0.15 signals template
  paraphrasing instead of reasoning.

## 8. What would falsify this amendment?

- Critic credit rate < 0.05 sustained → the gates are too tight or the
  critic is collapsing onto isosteric edits; debug before reading the
  hit-rate result.
- Pilot ratio gate failure → do not pay for the 32B arm.
- Cross-role evaluation parity failure → the trained C-PC checkpoint is
  not a drop-in replacement for arm C at deployment; the paper's claim
  becomes "C-PC training improves C-PC inference" rather than "C-PC
  improves Mpro discovery," and we must re-evaluate whether that is the
  interesting result.

## 9. Data and code freeze (additional artifacts)

Pinned **before** the first RL step under C-PC:

- `data/seeds/diffsbdd_7L13.jsonl` + the leakage reference used to build it
- `training/configs/verifier_pc.yaml`
- `training/configs/grpo_8b_c_pc.yaml` and `grpo_8b_c_base.yaml`
- `training/pilot/role_templates.py`
- `verifier/chemistry_verifier.py` (specifically `verify_pair_group`,
  `_apply_critic_credit`, `reward_fn_openrlhf_pc`)

No changes to these files after freeze without a v3 amendment.

## 10. Deviations log

Record any runtime deviation from this document here, with date, reason,
and new measurement plan.
