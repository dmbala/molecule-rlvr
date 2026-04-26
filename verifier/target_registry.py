"""Central registry for target-specific verifier metadata.

The goal is to separate per-target assay geometry, file paths, and default
success thresholds from the generic reward code. This keeps the episode runner
and verifier target-agnostic while making it explicit which values are frozen
versus still placeholders.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetSpec:
    name: str
    display_name: str
    receptor_pdbqt: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    reference_ligands_path: str
    pocket_context: str
    family: str
    non_covalent_only: bool = True
    default_constraints: dict[str, float | bool | None] = field(default_factory=dict)
    notes: str = ""


_TARGETS: dict[str, TargetSpec] = {
    "bace1": TargetSpec(
        name="bace1",
        display_name="BACE1",
        receptor_pdbqt="/workspace/data/targets/bace1/receptor.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(22.0, 22.0, 22.0),
        reference_ligands_path="/workspace/data/targets/bace1/reference_ligands.csv",
        pocket_context=(
            "BACE1 catalytic site. Favor productive hydrogen-bonding patterns "
            "consistent with Asp32/Asp228 engagement while avoiding oversized "
            "or highly lipophilic substitutions."
        ),
        family="aspartyl_protease",
        non_covalent_only=True,
        default_constraints={
            "dock_max": -8.0,
            "qed_min": 0.5,
            "sa_max": 5.0,
            "mw_max": 550.0,
            "logp_max": 5.5,
            "novelty_max_tanimoto_to_seed": 0.85,
            "forbid_pains": True,
            "non_covalent_only": True,
        },
        notes=(
            "Primary benchmark target. Geometry fields are placeholders until "
            "the receptor-prep and redock gate are frozen."
        ),
    ),
    "mpro_noncovalent": TargetSpec(
        name="mpro_noncovalent",
        display_name="SARS-CoV-2 Mpro (non-covalent)",
        receptor_pdbqt="/workspace/data/receptors/7L13/receptor.pdbqt",
        center=(10.4, -5.8, 23.1),
        size=(22.0, 22.0, 22.0),
        reference_ligands_path="/workspace/data/reference/known_mpro_inhibitors.csv",
        pocket_context=(
            "SARS-CoV-2 Mpro active site. Catalytic dyad His41/Cys145 with S1 "
            "and S4 sub-pockets shaping non-covalent recognition."
        ),
        family="viral_cysteine_protease",
        non_covalent_only=True,
        default_constraints={
            "dock_max": -7.0,
            "qed_min": 0.5,
            "sa_max": 5.0,
            "mw_max": 500.0,
            "logp_max": 5.0,
            "novelty_max_tanimoto_to_seed": 0.85,
            "forbid_pains": True,
            "non_covalent_only": True,
        },
        notes=(
            "Transfer target. Reference ligand file should eventually be split "
            "into a non-covalent-only subset for reward validation."
        ),
    ),
    "plpro": TargetSpec(
        name="plpro",
        display_name="SARS-CoV-2 PLpro",
        receptor_pdbqt="/workspace/data/targets/plpro/receptor.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(24.0, 24.0, 24.0),
        reference_ligands_path="/workspace/data/targets/plpro/reference_ligands.csv",
        pocket_context=(
            "SARS-CoV-2 PLpro pocket. Prefer compact non-covalent inhibitors "
            "that maintain polar contacts while avoiding highly reactive motifs."
        ),
        family="viral_cysteine_protease",
        non_covalent_only=True,
        default_constraints={
            "dock_max": -7.5,
            "qed_min": 0.5,
            "sa_max": 5.0,
            "mw_max": 550.0,
            "logp_max": 5.5,
            "novelty_max_tanimoto_to_seed": 0.85,
            "forbid_pains": True,
            "non_covalent_only": True,
        },
        notes=(
            "Secondary transfer target. Geometry fields are placeholders until "
            "receptor prep is added under data/targets/plpro."
        ),
    ),
}


def list_targets() -> list[str]:
    return sorted(_TARGETS)


def get_target_spec(name: str) -> TargetSpec:
    try:
        return _TARGETS[name]
    except KeyError as exc:
        valid = ", ".join(list_targets())
        raise KeyError(f"Unknown target '{name}'. Valid targets: {valid}") from exc

