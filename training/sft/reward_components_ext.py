"""CLI-side shim that lets build_stage0b_panel.py and the VS baseline run
Vina without importing the whole OpenRLHF RL stack."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_VERIFIER_DIR = Path(__file__).resolve().parent.parent.parent / "verifier"
sys.path.insert(0, str(_VERIFIER_DIR))

from reward_components import DockingConfig, run_vina_docking  # noqa: E402
from rdkit import Chem  # noqa: E402


@lru_cache(maxsize=8)
def load_docking_config(receptor_dir: str, *, n_seeds: int = 3,
                       exhaustiveness: int = 16) -> DockingConfig:
    """Parse receptor.config (Vina-style) and return a DockingConfig."""
    root = Path(receptor_dir)
    parsed: dict[str, float] = {}
    for line in (root / "receptor.config").read_text().splitlines():
        if "=" not in line:
            continue
        k, v = (x.strip() for x in line.split("=", 1))
        try:
            parsed[k] = float(v)
        except ValueError:
            parsed[k] = v
    return DockingConfig(
        receptor_pdbqt=str(root / "receptor.pdbqt"),
        center=(parsed["center_x"], parsed["center_y"], parsed["center_z"]),
        size=(parsed.get("size_x", 22.0), parsed.get("size_y", 22.0), parsed.get("size_z", 22.0)),
        exhaustiveness=int(parsed.get("exhaustiveness", exhaustiveness)),
        num_modes=int(parsed.get("num_modes", 9)),
        n_seeds=n_seeds,
    )


def run_docking_with_config(mol: Chem.Mol, receptor_dir: Path,
                             n_seeds: int = 3) -> tuple[float, float]:
    return run_vina_docking(mol, load_docking_config(str(receptor_dir), n_seeds=n_seeds))
