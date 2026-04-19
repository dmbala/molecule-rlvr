"""SMILES canonicalization + dedup helpers shared by the SFT data-prep scripts."""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED

RDLogger.DisableLog("rdApp.*")


def drug_like(mol: Chem.Mol,
              mw_range: tuple[float, float] = (150.0, 500.0),
              logp_max: float = 5.0,
              rotors_max: int = 10,
              qed_min: float = 0.4,
              heavy_atoms_min: int = 12) -> bool:
    """Loose drug-likeness filter used by Stage-0 corpus assembly."""
    if mol is None or mol.GetNumHeavyAtoms() < heavy_atoms_min:
        return False
    mw = Descriptors.MolWt(mol)
    if not (mw_range[0] <= mw <= mw_range[1]):
        return False
    try:
        logp = Descriptors.MolLogP(mol)
    except Exception:
        return False
    if logp > logp_max:
        return False
    if Descriptors.NumRotatableBonds(mol) > rotors_max:
        return False
    try:
        if QED.qed(mol) < qed_min:
            return False
    except Exception:
        return False
    return True


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        return max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return mol


def canonical_or_none(smiles: str, *, require_drug_like: bool = True,
                       ) -> tuple[str | None, str | None]:
    """Return (canonical_smiles, inchikey) after salt stripping, or (None, None)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    if require_drug_like and not drug_like(mol):
        return None, None
    mol = largest_fragment(mol)
    try:
        return Chem.MolToSmiles(mol, canonical=True), Chem.MolToInchiKey(mol)
    except Exception:
        return None, None


def canonicalize_and_dedup(smiles: list[str]) -> list[str]:
    """Canonicalize without drug-likeness filtering; dedup by canonical SMILES."""
    seen: set[str] = set()
    out: list[str] = []
    for s in smiles:
        canon, _ = canonical_or_none(s, require_drug_like=False)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def iter_smi_file(path: Path) -> Iterator[str]:
    """Iterate SMILES tokens from a plain or gzipped SMI/TXT file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for ln in f:
            s = ln.split()[0].strip() if ln.strip() else ""
            if s and not s.startswith("#") and not s.lower().startswith("smiles"):
                yield s


def iter_smi_dir(dirpath: Path) -> Iterator[str]:
    """Flatten all .smi* shards under a directory tree."""
    for p in dirpath.rglob("*.smi*"):
        yield from iter_smi_file(p)
