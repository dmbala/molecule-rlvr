"""Episode-level verifier for multi-turn tool-grounded chemistry RLVR.

This module wraps the existing one-turn chemistry verifier with episode-aware
logic: action parsing, previous-turn comparison, constraint checks, repetition
penalties, and terminal success detection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem

from chemistry_verifier import RewardRecord, VerifierConfig, extract_smiles, verify_group
from reward_components import has_pains, has_reactive_group, max_tanimoto, parse_smiles
from target_registry import TargetSpec, get_target_spec


@dataclass
class EpisodeStepInput:
    episode_id: str
    target: str
    turn: int
    response: str
    constraints: dict[str, Any]
    max_turns: int
    seed_smiles: str | None = None
    prior_smiles: list[str] | None = None
    prior_reward: float | None = None
    parent_smiles: str | None = None


@dataclass
class EpisodeStepResult:
    episode_id: str
    target: str
    turn: int
    action: str
    smiles: str | None
    parsed: bool
    base: RewardRecord
    reward_total: float
    delta_reward: float
    efficiency_bonus: float
    repeat_penalty: float
    constraint_passes: dict[str, bool]
    passes_all_constraints: bool
    success: bool
    done: bool
    summary_text: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["base"] = self.base.to_dict()
        return out


def extract_action(response: str) -> str:
    if not response:
        return "propose"
    open_tag = "<action>"
    close_tag = "</action>"
    if open_tag in response and close_tag in response:
        action = response.split(open_tag, 1)[1].split(close_tag, 1)[0].strip().lower()
        if action:
            return action
    return "propose"


def _extract_tag(response: str, tag: str) -> str | None:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag in response and close_tag in response:
        return response.split(open_tag, 1)[1].split(close_tag, 1)[0].strip() or None
    return None


def _constraint_passes(
    record: RewardRecord,
    mol: Chem.Mol | None,
    constraints: dict[str, Any],
    target_spec: TargetSpec,
    seed_mol: Chem.Mol | None,
) -> dict[str, bool]:
    passes: dict[str, bool] = {}

    passes["valid"] = record.parsed and mol is not None
    if not passes["valid"]:
        return passes

    desc = record.descriptors or {}
    passes["dock_max"] = (
        True if constraints.get("dock_max") is None else (record.dock_kcal is not None and record.dock_kcal <= constraints["dock_max"])
    )
    passes["qed_min"] = (
        True if constraints.get("qed_min") is None else (record.qed is not None and record.qed >= constraints["qed_min"])
    )
    passes["sa_max"] = (
        True if constraints.get("sa_max") is None else (record.sa_raw is not None and record.sa_raw <= constraints["sa_max"])
    )
    passes["mw_max"] = (
        True if constraints.get("mw_max") is None else desc.get("mw", float("inf")) <= constraints["mw_max"]
    )
    passes["logp_max"] = (
        True if constraints.get("logp_max") is None else desc.get("logp", float("inf")) <= constraints["logp_max"]
    )
    passes["hbd_max"] = (
        True if constraints.get("hbd_max") is None else desc.get("hbd", float("inf")) <= constraints["hbd_max"]
    )
    passes["hba_max"] = (
        True if constraints.get("hba_max") is None else desc.get("hba", float("inf")) <= constraints["hba_max"]
    )
    passes["forbid_pains"] = True if not constraints.get("forbid_pains", False) else not has_pains(mol)
    require_non_covalent = bool(
        constraints.get("non_covalent_only", target_spec.non_covalent_only)
    )
    passes["non_covalent_only"] = True if not require_non_covalent else not has_reactive_group(mol)

    if constraints.get("novelty_max_tanimoto_to_seed") is None or seed_mol is None:
        passes["novelty_max_tanimoto_to_seed"] = True
    else:
        sim = max_tanimoto(mol, [seed_mol])
        passes["novelty_max_tanimoto_to_seed"] = sim <= constraints["novelty_max_tanimoto_to_seed"]

    return passes


def _repeat_penalty(
    mol: Chem.Mol | None,
    prior_smiles: list[str] | None,
    parent_mol: Chem.Mol | None,
) -> float:
    if mol is None:
        return 0.0

    prior_mols = [parse_smiles(s) for s in (prior_smiles or [])]
    prior_mols = [m for m in prior_mols if m is not None]
    if parent_mol is not None:
        prior_mols.append(parent_mol)
    if not prior_mols:
        return 0.0

    max_sim = max_tanimoto(mol, prior_mols)
    if max_sim >= 0.995:
        return -0.25
    if max_sim >= 0.95:
        return -0.10
    return 0.0


def _efficiency_bonus(turn: int, max_turns: int, success: bool) -> float:
    if not success:
        return 0.0
    remaining = max(0, max_turns - turn)
    return 0.05 * (1 + remaining)


def _delta_reward(current: float, prior_reward: float | None) -> float:
    if prior_reward is None:
        return 0.0
    return 0.20 * (current - prior_reward)


def _passes_all_constraints(passes: dict[str, bool]) -> bool:
    return bool(passes) and all(passes.values())


def _build_summary(
    action: str,
    record: RewardRecord,
    passes_all_constraints: bool,
    success: bool,
    remaining_turns: int,
) -> str:
    if not record.parsed:
        return (
            f"Action={action}. The molecule could not be parsed. "
            f"Remaining turns: {remaining_turns}."
        )

    parts = [
        f"Action={action}",
        f"dock={record.dock_kcal:.2f}" if record.dock_kcal is not None else "dock=n/a",
        f"qed={record.qed:.2f}" if record.qed is not None else "qed=n/a",
        f"sa={record.sa_raw:.2f}" if record.sa_raw is not None else "sa=n/a",
    ]
    if record.failure_reason:
        parts.append(f"failure={record.failure_reason}")
    parts.append(f"constraints={'pass' if passes_all_constraints else 'fail'}")
    if success:
        parts.append("status=success")
    parts.append(f"remaining_turns={remaining_turns}")
    return "; ".join(parts)


def score_episode_step(
    step: EpisodeStepInput,
    cfg: VerifierConfig,
) -> EpisodeStepResult:
    target_spec = get_target_spec(step.target)
    action = extract_action(step.response)
    smiles = _extract_tag(step.response, "answer") or extract_smiles(step.response)

    base = verify_group([step.response], cfg, step.turn)[0]
    mol = parse_smiles(smiles or "")
    seed_mol = parse_smiles(step.seed_smiles or "")
    parent_mol = parse_smiles(step.parent_smiles or "")

    constraint_passes = _constraint_passes(
        base,
        mol,
        step.constraints,
        target_spec,
        seed_mol,
    )
    passes_all = _passes_all_constraints(constraint_passes)
    success = base.parsed and passes_all

    delta_reward = _delta_reward(base.reward, step.prior_reward)
    repeat_penalty = _repeat_penalty(mol, step.prior_smiles, parent_mol)
    efficiency_bonus = _efficiency_bonus(step.turn, step.max_turns, success)
    reward_total = base.reward + delta_reward + repeat_penalty + efficiency_bonus

    done = success or step.turn >= step.max_turns
    summary_text = _build_summary(
        action=action,
        record=base,
        passes_all_constraints=passes_all,
        success=success,
        remaining_turns=max(0, step.max_turns - step.turn),
    )

    return EpisodeStepResult(
        episode_id=step.episode_id,
        target=step.target,
        turn=step.turn,
        action=action,
        smiles=smiles,
        parsed=base.parsed,
        base=base,
        reward_total=float(reward_total),
        delta_reward=float(delta_reward),
        efficiency_bonus=float(efficiency_bonus),
        repeat_penalty=float(repeat_penalty),
        constraint_passes=constraint_passes,
        passes_all_constraints=passes_all,
        success=success,
        done=done,
        summary_text=summary_text,
    )
