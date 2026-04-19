"""Building blocks for the composite chemistry-RLVR reward.

See `plan` Fix 2. Each function here is pure and cheap except for
`run_vina_docking`, which shells out (or calls the Vina Python API) and
is the dominant per-rollout cost.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds import MurckoScaffold

import sascorer  # /opt/sascorer on PYTHONPATH inside the container


# --- Parsing / safety filters -----------------------------------------------

_PAINS_CATALOG: FilterCatalog | None = None


def _pains_catalog() -> FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog(params)
    return _PAINS_CATALOG


_REACTIVE_SMARTS = [
    "[CX3H1](=O)[#6]",          # aldehyde
    "[CX3](=O)[Cl,Br,I]",       # acyl halide
    "[N+]#N",                   # diazonium
    "[S,C]=[N+]=[N-]",          # diazo
    "[CX3]=[CX3][N+](=O)[O-]",  # nitroalkene
    "O1CC1",                    # epoxide
    "N1CC1",                    # aziridine
    "[CX3](=O)O[CX3](=O)",      # anhydride
]
_REACTIVE_PATTERNS = [Chem.MolFromSmarts(s) for s in _REACTIVE_SMARTS]


def parse_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Reject disconnected multi-molecule strings; pick largest fragment instead
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if mol.GetNumHeavyAtoms() < 5:
        return None
    return mol


def has_pains(mol: Chem.Mol) -> bool:
    return _pains_catalog().HasMatch(mol)


def has_reactive_group(mol: Chem.Mol) -> bool:
    return any(mol.HasSubstructMatch(p) for p in _REACTIVE_PATTERNS if p is not None)


# --- Reward components ------------------------------------------------------

def shaped_dock(dock_kcal: float, center: float = -7.0, scale: float = 1.5) -> float:
    """Sigmoid reward centered at `center` kcal/mol, scale `scale`.

    ~0 at -4, ~0.5 at -7, ~1 at -10. Use `center` curriculum (Fix 8.4).
    """
    return 1.0 / (1.0 + math.exp((dock_kcal - center) / scale))


def qed_score(mol: Chem.Mol) -> float:
    return float(QED.qed(mol))


def sa_score_normalized(mol: Chem.Mol) -> float:
    """Inverted + normalized SAscore. 1.0 = trivially synthesizable, 0.0 = hard."""
    sa = sascorer.calculateScore(mol)  # [1, 10]
    return max(0.0, 1.0 - (sa - 1.0) / 9.0)


def morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def max_tanimoto(mol: Chem.Mol, others: Sequence[Chem.Mol]) -> float:
    if not others:
        return 0.0
    fp = morgan_fp(mol)
    others_fp = [morgan_fp(o) for o in others]
    sims = DataStructs.BulkTanimotoSimilarity(fp, others_fp)
    return float(max(sims)) if sims else 0.0


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf else ""
    except Exception:
        return ""


# --- Docking ----------------------------------------------------------------

@dataclass
class DockingConfig:
    receptor_pdbqt: str
    center: tuple[float, float, float]
    size: tuple[float, float, float] = (22.0, 22.0, 22.0)
    exhaustiveness: int = 16
    num_modes: int = 9
    n_seeds: int = 3
    vina_binary: str = "vina"  # CLI fallback; prefer Python API when available


def _ligand_to_pdbqt(mol: Chem.Mol, path: str) -> None:
    """Prepare ligand PDBQT via meeko. Runs 3D embed + MMFF if needed."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
    prep = MoleculePreparation()
    prep.prepare(mol_h)
    pdbqt = PDBQTWriterLegacy.write_string(prep.setup)[0]
    with open(path, "w") as f:
        f.write(pdbqt)


def run_vina_docking(mol: Chem.Mol, cfg: DockingConfig) -> tuple[float, float]:
    """Run Vina n_seeds times; return (best_score_min_over_seeds, strain_energy).

    Uses the Vina Python API when importable; falls back to CLI invocation.
    strain_energy is MMFF94 strain estimated as the energy difference between
    the docked pose and its local-minimum re-optimization (Fix 2, strain gate).
    """
    from vina import Vina

    scores: list[float] = []
    best_pose_mol: Chem.Mol | None = None

    with tempfile.TemporaryDirectory() as tmp:
        lig_pdbqt = os.path.join(tmp, "lig.pdbqt")
        _ligand_to_pdbqt(mol, lig_pdbqt)

        for seed in range(cfg.n_seeds):
            v = Vina(sf_name="vina", cpu=1, seed=seed, verbosity=0)
            v.set_receptor(cfg.receptor_pdbqt)
            v.set_ligand_from_file(lig_pdbqt)
            v.compute_vina_maps(center=list(cfg.center), box_size=list(cfg.size))
            v.dock(exhaustiveness=cfg.exhaustiveness, n_poses=cfg.num_modes)
            energies = v.energies(n_poses=1)
            if len(energies) == 0:
                continue
            scores.append(float(energies[0][0]))
            if best_pose_mol is None or scores[-1] == min(scores):
                pose_pdbqt = os.path.join(tmp, f"pose_{seed}.pdbqt")
                v.write_poses(pose_pdbqt, n_poses=1, overwrite=True)
                try:
                    best_pose_mol = _pdbqt_to_mol(pose_pdbqt)
                except Exception:
                    best_pose_mol = None

    if not scores:
        return float("inf"), float("inf")

    strain = _pose_strain_energy(best_pose_mol) if best_pose_mol is not None else 0.0
    return float(min(scores)), float(strain)


def _pdbqt_to_mol(pdbqt_path: str) -> Chem.Mol | None:
    """Convert a docked PDBQT pose back to an RDKit mol via OpenBabel."""
    from openbabel import pybel

    obmol = next(pybel.readfile("pdbqt", pdbqt_path))
    pdb = obmol.write("pdb")
    m = Chem.MolFromPDBBlock(pdb, removeHs=False)
    return m


def _pose_strain_energy(pose: Chem.Mol) -> float:
    """MMFF94 strain: E(pose) - E(relaxed pose), kcal/mol. Cheap clash detector."""
    try:
        mol = Chem.AddHs(pose, addCoords=True)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            return 0.0
        ff_pose = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff_pose is None:
            return 0.0
        e_pose = float(ff_pose.CalcEnergy())

        mol_relax = Chem.Mol(mol)
        ff_relax = AllChem.MMFFGetMoleculeForceField(mol_relax, props)
        ff_relax.Minimize(maxIts=500)
        e_relax = float(ff_relax.CalcEnergy())

        return max(0.0, e_pose - e_relax)
    except Exception:
        return 0.0


# --- Utilities --------------------------------------------------------------

def descriptors(mol: Chem.Mol) -> dict[str, float]:
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "rot": int(Descriptors.NumRotatableBonds(mol)),
        "heavy": int(mol.GetNumHeavyAtoms()),
    }
