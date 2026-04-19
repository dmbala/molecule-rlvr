"""CLI-side shim that lets build_stage0b_panel.py and the VS baseline run
Vina without importing the whole OpenRLHF RL stack."""
from __future__ import annotations

import sys
from pathlib import Path

# Import the verifier package without running it as a full RL stack
_VERIFIER_DIR = Path(__file__).resolve().parent.parent.parent / "verifier"
sys.path.insert(0, str(_VERIFIER_DIR))

from reward_components import DockingConfig, run_vina_docking  # noqa: E402
from rdkit import Chem  # noqa: E402


def load_docking_config(receptor_dir: Path, *, n_seeds: int = 3,
                       exhaustiveness: int = 16) -> DockingConfig:
    """Parse receptor.config (Vina-style) and return a DockingConfig."""
    cfg_file = Path(receptor_dir) / "receptor.config"
    pdbqt = Path(receptor_dir) / "receptor.pdbqt"
    d: dict[str, float] = {}
    for line in cfg_file.read_text().splitlines():
        if "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return DockingConfig(
        receptor_pdbqt=str(pdbqt),
        center=(d["center_x"], d["center_y"], d["center_z"]),
        size=(d.get("size_x", 22.0), d.get("size_y", 22.0), d.get("size_z", 22.0)),
        exhaustiveness=int(d.get("exhaustiveness", exhaustiveness)),
        num_modes=int(d.get("num_modes", 9)),
        n_seeds=n_seeds,
    )


def run_docking_with_config(mol: Chem.Mol, receptor_dir: Path,
                             n_seeds: int = 3) -> tuple[float, float]:
    cfg = load_docking_config(Path(receptor_dir), n_seeds=n_seeds)
    return run_vina_docking(mol, cfg)
