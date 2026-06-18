"""chemistry_verifier: RLVR reward function for the GRPO trainer.

Implements the composite reward from the plan's Fix 2, plus the Fix 8.4
curriculum hook on the docking sigmoid center. Called by the OpenRLHF
trainer once per rollout group.

Reward (per molecule, per turn):
    r = 0.50 * r_dock  +  0.15 * r_qed  +  0.15 * r_sa  +  0.20 * r_div
with hard filter short-circuits for parsing / PAINS / reactive / strain.

Group-level calls:
    verify_group(prompts, responses, group_context, cfg) -> list[RewardRecord]
Each record carries the scalar reward plus enough metadata for logging.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from rdkit import Chem, RDLogger

from reward_components import (
    DockingConfig,
    descriptors,
    has_pains,
    has_reactive_group,
    max_tanimoto,
    murcko_scaffold_smiles,
    parse_smiles,
    qed_score,
    run_vina_docking,
    shaped_dock,
)

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("chem_verifier")

_LABEL_PREFIXES = ("Final:", "Answer:", "SMILES:", "Result:")


# --- Configuration ----------------------------------------------------------

@dataclass
class RewardWeights:
    dock: float = 0.50
    qed: float = 0.15
    sa: float = 0.15
    diversity: float = 0.20

    def validate(self) -> None:
        total = self.dock + self.qed + self.sa + self.diversity
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Reward weights must sum to 1.0, got {total}")


@dataclass
class CurriculumSchedule:
    """Shifts the sigmoid-dock center per training step (Fix 8.4)."""
    milestones: tuple[tuple[int, float], ...] = (
        (0, -5.0),
        (100, -7.0),
        (300, -8.0),
    )

    def center_at_step(self, step: int) -> float:
        current = self.milestones[0][1]
        for s, c in self.milestones:
            if step >= s:
                current = c
            else:
                break
        return current


@dataclass
class VerifierConfig:
    docking: DockingConfig
    weights: RewardWeights = field(default_factory=RewardWeights)
    curriculum: CurriculumSchedule = field(default_factory=CurriculumSchedule)
    # Hard-filter penalties (Fix 2).
    invalid_penalty: float = -1.0
    pains_penalty: float = -0.5
    reactive_penalty: float = -0.5
    strain_threshold_kcal: float = 60.0
    strain_penalty: float = -0.2
    # Rank-based alternative (Fix 2). Set True to ignore absolute dock and use
    # rank-within-group for the docking term only.
    rank_based_dock: bool = False
    # Multi-turn improvement bonus coefficient (Fix 2).
    multiturn_improvement_coef: float = 0.2
    # Proposer-critic (Amendment v2). When True, verify_pair_group applies the
    # non-zero-sum critic credit gated by dock improvement + Tanimoto.
    proposer_critic_enabled: bool = False
    critic_alpha: float = 0.5
    critic_min_dock_improvement_kcal: float = 0.5
    critic_tanimoto_max: float = 0.7
    within_role_baseline: bool = True


# --- Records ----------------------------------------------------------------

@dataclass
class RewardRecord:
    """Per-rollout reward + metadata. Log all fields to W&B / JSONL."""
    reward: float
    smiles: str | None
    parsed: bool
    failure_reason: str | None = None  # "parse", "pains", "reactive", "strain", None
    dock_kcal: float | None = None
    strain_kcal: float | None = None
    qed: float | None = None
    sa_raw: float | None = None
    r_dock: float | None = None
    r_qed: float | None = None
    r_sa: float | None = None
    r_div: float | None = None
    scaffold: str | None = None
    max_tanimoto_in_group: float | None = None
    descriptors: dict[str, float] | None = None
    step: int | None = None
    turn: int | None = None
    role: Literal["proposer", "critic"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- SMILES extraction ------------------------------------------------------

def extract_smiles(response: str) -> str | None:
    """Pull a SMILES out of a (possibly CoT-laden) model response.

    Rules:
    1. Prefer the last <answer>...</answer> span (matches Qwen3 thinking-mode).
    2. Else prefer the last fenced ```smiles ... ``` block.
    3. Else take the last non-empty line and strip quotes/punctuation.
    """
    if not response:
        return None

    for marker in ("</answer>", "</SMILES>"):
        if marker in response:
            tail = response.rsplit(marker, 1)[0]
            opener = marker.replace("/", "")
            if opener in tail:
                return tail.rsplit(opener, 1)[-1].strip() or None

    if "```" in response:
        parts = response.split("```")
        # Fenced blocks are at odd indices
        for block in reversed(parts[1::2]):
            lang, _, body = block.partition("\n")
            cand = (body if lang.lower() in {"smiles", "chem"} else block).strip()
            if cand:
                return cand.splitlines()[-1].strip()

    for line in reversed(response.splitlines()):
        s = line.strip().strip("`'\"")
        if not s or s.startswith("Turn "):
            continue
        for p in _LABEL_PREFIXES:
            if s.startswith(p):
                s = s[len(p):].strip().strip("`'\"")
                break
        if s:
            return s
    return None


# --- Core API ---------------------------------------------------------------

def _score_molecule(
    mol: Chem.Mol | None,
    group_mols: Sequence[Chem.Mol],
    cfg: VerifierConfig,
    step: int,
) -> RewardRecord:
    """Score one molecule against a group of siblings (for diversity)."""
    if mol is None:
        return RewardRecord(
            reward=cfg.invalid_penalty,
            smiles=None,
            parsed=False,
            failure_reason="parse",
            step=step,
        )

    smi = Chem.MolToSmiles(mol)

    if has_pains(mol):
        return RewardRecord(reward=cfg.pains_penalty, smiles=smi, parsed=True,
                            failure_reason="pains", step=step)
    if has_reactive_group(mol):
        return RewardRecord(reward=cfg.reactive_penalty, smiles=smi, parsed=True,
                            failure_reason="reactive", step=step)

    # Docking (dominant cost)
    dock_kcal, strain = run_vina_docking(mol, cfg.docking)

    if strain > cfg.strain_threshold_kcal:
        return RewardRecord(
            reward=cfg.strain_penalty, smiles=smi, parsed=True,
            failure_reason="strain", dock_kcal=dock_kcal, strain_kcal=strain,
            step=step,
        )

    # Reward components (absolute form; rank form applied later if enabled)
    center = cfg.curriculum.center_at_step(step)
    import sascorer
    sa_raw = float(sascorer.calculateScore(mol))
    r_dock = shaped_dock(dock_kcal, center=center)
    r_qed = qed_score(mol)
    r_sa = max(0.0, 1.0 - (sa_raw - 1.0) / 9.0)
    peers = [m for m in group_mols if m is not None and m is not mol]
    max_sim = max_tanimoto(mol, peers)
    r_div = 1.0 - max_sim

    w = cfg.weights
    reward = w.dock * r_dock + w.qed * r_qed + w.sa * r_sa + w.diversity * r_div

    return RewardRecord(
        reward=float(reward),
        smiles=smi,
        parsed=True,
        dock_kcal=float(dock_kcal),
        strain_kcal=float(strain),
        qed=float(r_qed),
        sa_raw=sa_raw,
        r_dock=float(r_dock),
        r_qed=float(r_qed),
        r_sa=float(r_sa),
        r_div=float(r_div),
        scaffold=murcko_scaffold_smiles(mol),
        max_tanimoto_in_group=float(max_sim),
        descriptors=descriptors(mol),
        step=step,
    )


def _apply_rank_dock(records: list[RewardRecord], cfg: VerifierConfig) -> None:
    """Replace `r_dock` with within-group rank-based reward (Fix 2 alternative)."""
    dockables = [r for r in records if r.dock_kcal is not None and math.isfinite(r.dock_kcal)]
    if len(dockables) < 2:
        return
    ordered = sorted(dockables, key=lambda r: r.dock_kcal)  # best (most negative) first
    G = len(ordered)
    for rank, r in enumerate(ordered):
        r_dock_rank = (G - 1 - rank) / (G - 1)
        old_r_dock = r.r_dock if r.r_dock is not None else 0.0
        # Recompose reward with rank-based dock term
        r.reward += cfg.weights.dock * (r_dock_rank - old_r_dock)
        r.r_dock = r_dock_rank


def _apply_critic_credit(
    critic_records: list[RewardRecord],
    parent_smiles: Sequence[str | None],
    parent_rewards: Sequence[float | None],
    parent_docks: Sequence[float | None],
    cfg: VerifierConfig,
) -> None:
    """Add `cfg.critic_alpha * ReLU(r2 − r1)` to each critic's reward iff
    BOTH gates pass — dock improvement >= cfg.critic_min_dock_improvement_kcal
    AND Tanimoto(s1, s2) <= cfg.critic_tanimoto_max. Risk #1 + #2 in plan.

    `parent_*` lists are aligned 1:1 with `critic_records`.
    """
    if not (len(critic_records) == len(parent_smiles) == len(parent_rewards) == len(parent_docks)):
        raise ValueError("parent_* lists must align 1:1 with critic_records")

    for r2, s1, r1, d1 in zip(critic_records, parent_smiles, parent_rewards, parent_docks):
        if not r2.parsed or r2.smiles is None:
            continue
        if s1 is None or r1 is None or d1 is None:
            continue
        if r2.dock_kcal is None or not math.isfinite(r2.dock_kcal):
            continue
        # Dock improvement: more negative is better, so improvement = d1 − d2
        if (d1 - r2.dock_kcal) < cfg.critic_min_dock_improvement_kcal:
            continue
        m1 = parse_smiles(s1)
        m2 = parse_smiles(r2.smiles)
        if m1 is None or m2 is None:
            continue
        if max_tanimoto(m2, [m1]) > cfg.critic_tanimoto_max:
            continue
        r2.reward += cfg.critic_alpha * max(0.0, r2.reward - r1)


def verify_group(
    responses: Sequence[str],
    cfg: VerifierConfig,
    step: int = 0,
    prior_turn_rewards: Sequence[float] | None = None,
) -> list[RewardRecord]:
    """Score a GRPO group of rollouts for one prompt.

    `prior_turn_rewards` is the list of first-turn rewards (one per trajectory)
    when scoring turn-T > 1; used for the multi-turn improvement bonus.
    """
    cfg.weights.validate()
    t0 = time.time()

    mols = [parse_smiles(extract_smiles(r) or "") for r in responses]
    records = [_score_molecule(m, mols, cfg, step) for m in mols]

    if cfg.rank_based_dock:
        _apply_rank_dock(records, cfg)

    if prior_turn_rewards is not None:
        assert len(prior_turn_rewards) == len(records)
        for r, r1 in zip(records, prior_turn_rewards):
            if r.parsed:
                r.reward += cfg.multiturn_improvement_coef * (r.reward - r1)

    dt = time.time() - t0
    parsed = sum(1 for r in records if r.parsed)
    log.info(
        "step=%d group=%d parsed=%d/%d wall=%.1fs mean_r=%.3f",
        step, len(records), parsed, len(records), dt,
        sum(r.reward for r in records) / len(records),
    )
    return records


def verify_pair_group(
    proposer_responses: Sequence[str],
    critic_responses: Sequence[str],
    parent_smiles: Sequence[str | None],
    parent_rewards: Sequence[float | None],
    parent_docks: Sequence[float | None],
    cfg: VerifierConfig,
    step: int = 0,
) -> list[RewardRecord]:
    """Score a proposer-critic GRPO sub-group for one prompt (Amendment v2).

    Returns 2G records — `len(proposer_responses)` proposer records first,
    then `len(critic_responses)` critic records. The critic-credit gate
    (dock improvement + Tanimoto) is applied to critic records only.

    `parent_*` lists align 1:1 with `critic_responses`. They name the
    proposer SMILES that each critic was tasked with improving. Use the
    proposer's reward and dock from the same RL step.
    """
    proposer_records = verify_group(proposer_responses, cfg, step=step)
    for r in proposer_records:
        r.role = "proposer"

    critic_records = verify_group(critic_responses, cfg, step=step)
    for r in critic_records:
        r.role = "critic"

    if cfg.proposer_critic_enabled:
        _apply_critic_credit(
            critic_records, parent_smiles, parent_rewards, parent_docks, cfg,
        )

    return list(proposer_records) + list(critic_records)


# --- OpenRLHF adapter -------------------------------------------------------

def reward_fn_openrlhf(queries, responses, **kwargs) -> list[float]:
    """Entry point wired into OpenRLHF via `--reward_functions chemistry_verifier`.

    OpenRLHF passes flat lists of queries + responses. We expect a per-batch
    VerifierConfig to be attached via env var CHEM_VERIFIER_CONFIG (YAML path).
    """
    from verifier_config_loader import load_verifier_config  # lazy import

    cfg_path = os.environ["CHEM_VERIFIER_CONFIG"]
    step = int(os.environ.get("CHEM_VERIFIER_STEP", "0"))
    cfg = load_verifier_config(cfg_path)

    records = verify_group(responses, cfg, step=step)
    # OpenRLHF wants a float list aligned with `responses`.
    return [r.reward for r in records]


# --- OpenRLHF adapter: proposer-critic --------------------------------------

_TAG_RE_CACHE: dict[str, re.Pattern] = {}


def _tag_pattern(name: str) -> re.Pattern:
    pat = _TAG_RE_CACHE.get(name)
    if pat is None:
        pat = re.compile(rf"<{re.escape(name)}>(.*?)</{re.escape(name)}>", re.DOTALL)
        _TAG_RE_CACHE[name] = pat
    return pat


def _parse_tag(text: str, name: str) -> str | None:
    """Return the inner text of the first <name>...</name> block, or None."""
    m = _tag_pattern(name).search(text or "")
    return m.group(1).strip() if m else None


def _parse_float_tag(text: str, name: str) -> float | None:
    raw = _parse_tag(text, name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def reward_fn_openrlhf_pc(queries, responses, **kwargs) -> list[float]:
    """OpenRLHF entry point for the C-PC arm.

    Each query string must carry these tags (the dataset preprocessor + the
    rollout collator are responsible for filling them in):
      <root_id>...</root_id>            shared between a proposer query and the matching critic query
      <role>proposer|critic</role>
      <parent_smiles>...</parent_smiles>      critic-only: s1 from the proposer
      <parent_reward>...</parent_reward>      critic-only: r(s1)
      <parent_dock>...</parent_dock>          critic-only: dock_kcal(s1)

    Responses are scored per (root_id, role) group via verify_pair_group.
    """
    from verifier_config_loader import load_verifier_config  # lazy

    cfg_path = os.environ["CHEM_VERIFIER_CONFIG"]
    step = int(os.environ.get("CHEM_VERIFIER_STEP", "0"))
    cfg = load_verifier_config(cfg_path)

    by_root: dict[str, dict[str, list[int]]] = {}
    for i, q in enumerate(queries):
        root_id = _parse_tag(q, "root_id") or f"_anon_{i}"
        role = (_parse_tag(q, "role") or "proposer").lower()
        if role not in {"proposer", "critic"}:
            role = "proposer"
        by_root.setdefault(root_id, {"proposer": [], "critic": []})[role].append(i)

    out = [0.0] * len(responses)
    for root_id, slots in by_root.items():
        prop_idxs = slots["proposer"]
        crit_idxs = slots["critic"]
        prop_resp = [responses[i] for i in prop_idxs]
        crit_resp = [responses[i] for i in crit_idxs]
        parent_smiles = [_parse_tag(queries[i], "parent_smiles") for i in crit_idxs]
        parent_rewards = [_parse_float_tag(queries[i], "parent_reward") for i in crit_idxs]
        parent_docks = [_parse_float_tag(queries[i], "parent_dock") for i in crit_idxs]
        recs = verify_pair_group(
            prop_resp, crit_resp,
            parent_smiles, parent_rewards, parent_docks,
            cfg, step=step,
        )
        for r, idx in zip(recs[: len(prop_idxs)], prop_idxs):
            out[idx] = r.reward
        for r, idx in zip(recs[len(prop_idxs):], crit_idxs):
            out[idx] = r.reward
    return out
