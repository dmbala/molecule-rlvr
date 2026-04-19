"""YAML → VerifierConfig loader. Kept separate so chemistry_verifier stays
importable in unit tests without a config file on disk."""
from __future__ import annotations

import yaml

from chemistry_verifier import (
    CurriculumSchedule,
    RewardWeights,
    VerifierConfig,
)
from reward_components import DockingConfig


def load_verifier_config(path: str) -> VerifierConfig:
    with open(path) as f:
        d = yaml.safe_load(f)

    dock_d = d["docking"]
    docking = DockingConfig(
        receptor_pdbqt=dock_d["receptor_pdbqt"],
        center=tuple(dock_d["center"]),
        size=tuple(dock_d.get("size", [22.0, 22.0, 22.0])),
        exhaustiveness=int(dock_d.get("exhaustiveness", 16)),
        num_modes=int(dock_d.get("num_modes", 9)),
        n_seeds=int(dock_d.get("n_seeds", 3)),
        vina_binary=dock_d.get("vina_binary", "vina"),
    )

    weights = RewardWeights(**d.get("weights", {}))

    curriculum = CurriculumSchedule(
        milestones=tuple(tuple(m) for m in d.get("curriculum", {}).get("milestones",
            [(0, -5.0), (100, -7.0), (300, -8.0)]))
    )

    return VerifierConfig(
        docking=docking,
        weights=weights,
        curriculum=curriculum,
        invalid_penalty=float(d.get("invalid_penalty", -1.0)),
        pains_penalty=float(d.get("pains_penalty", -0.5)),
        reactive_penalty=float(d.get("reactive_penalty", -0.5)),
        strain_threshold_kcal=float(d.get("strain_threshold_kcal", 60.0)),
        strain_penalty=float(d.get("strain_penalty", -0.2)),
        rank_based_dock=bool(d.get("rank_based_dock", False)),
        multiturn_improvement_coef=float(d.get("multiturn_improvement_coef", 0.2)),
    )
