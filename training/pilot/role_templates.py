"""Proposer / critic system prompts for Amendment v2 (Arm C-PC).

Both prompts assume the chat template will inject `<think>...</think>` blocks
under `enable_thinking=True` on Qwen3. The critic template asks for an
explicit articulation of what the proposer missed, then a single SMILES on
its own line — `extract_smiles` in `verifier/chemistry_verifier.py` picks up
the trailing line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PROPOSER_SYSTEM = (
    "You are an expert medicinal chemist designing non-covalent inhibitors "
    "of SARS-CoV-2 Mpro (PDB 7L13). You receive a pocket description and an "
    "optional seed SMILES. Reason briefly about which pocket residues to "
    "engage, then output ONE SMILES string on its own final line. Constraints: "
    "QED > 0.5, SA < 5, MW < 500, no PAINS alerts."
)


CRITIC_SYSTEM = (
    "You are critiquing a colleague's proposed SARS-CoV-2 Mpro inhibitor. "
    "You see the pocket context, the proposed SMILES, and its measured "
    "AutoDock Vina + property scores. Identify which pocket residue "
    "interaction the proposal missed (e.g. an H-bond to Glu166, a clash in "
    "S1, a hydrophobic mismatch with Met49), then output ONE improved SMILES "
    "string on its own final line. Your improvement must be structurally "
    "distinct (Tanimoto < 0.7) and must dock at least 0.5 kcal/mol better "
    "to receive credit."
)


CRITIC_USER_TEMPLATE = (
    "Pocket context:\n"
    "{context}\n\n"
    "Original task:\n"
    "{instruction}\n\n"
    "Colleague's proposed SMILES (s1): {parent_smiles}\n"
    "Measured scores for s1:\n"
    "  AutoDock Vina = {parent_dock:.2f} kcal/mol\n"
    "  QED           = {parent_qed:.2f}\n"
    "  SAscore       = {parent_sa:.2f}\n"
    "  Composite reward = {parent_reward:.3f}\n\n"
    "Critique s1: name the specific residue interaction it missed, then "
    "propose an improved SMILES s2 on its own final line."
)


# Same shape as CRITIC_USER_TEMPLATE but with literal sentinels in place of
# the parent-value format specs. Used at *dataset-prep* time, when the
# proposer's actual values aren't known yet. The OpenRLHF rollout collator
# text-replaces __PARENT_*__ with float strings between the two role passes.
CRITIC_USER_TEMPLATE_SENTINELS = (
    "Pocket context:\n"
    "{context}\n\n"
    "Original task:\n"
    "{instruction}\n\n"
    "Colleague's proposed SMILES (s1): __PARENT_SMILES__\n"
    "Measured scores for s1:\n"
    "  AutoDock Vina = __PARENT_DOCK__ kcal/mol\n"
    "  QED           = __PARENT_QED__\n"
    "  SAscore       = __PARENT_SA__\n"
    "  Composite reward = __PARENT_REWARD__\n\n"
    "Critique s1: name the specific residue interaction it missed, then "
    "propose an improved SMILES s2 on its own final line."
)


@dataclass
class ParentSummary:
    """Minimal subset of RewardRecord that the critic needs."""
    smiles: str
    reward: float
    dock_kcal: float | None
    qed: float | None
    sa_raw: float | None


def _fmt_or_na(v: float | None, kind: str = "f") -> str:
    if v is None:
        return "NA"
    if kind == "f":
        return f"{v:.2f}"
    return str(v)


def render_critic_user(prompt: dict, parent: ParentSummary) -> str:
    """Build the critic's user message for a given (prompt, parent)."""
    return CRITIC_USER_TEMPLATE.format(
        context=prompt.get("context", ""),
        instruction=prompt.get("instruction", ""),
        parent_smiles=parent.smiles or "INVALID",
        parent_dock=parent.dock_kcal if parent.dock_kcal is not None else float("nan"),
        parent_qed=parent.qed if parent.qed is not None else float("nan"),
        parent_sa=parent.sa_raw if parent.sa_raw is not None else float("nan"),
        parent_reward=parent.reward,
    )


def parent_summary_from_record(rec: Any) -> ParentSummary:
    """Build a ParentSummary from a verifier RewardRecord (duck-typed)."""
    return ParentSummary(
        smiles=getattr(rec, "smiles", "") or "",
        reward=float(getattr(rec, "reward", 0.0)),
        dock_kcal=getattr(rec, "dock_kcal", None),
        qed=getattr(rec, "qed", None),
        sa_raw=getattr(rec, "sa_raw", None),
    )
