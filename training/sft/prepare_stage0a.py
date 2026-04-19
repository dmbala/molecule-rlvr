"""Assemble the Stage 0a SFT corpus: ~5M drug-like SMILES from ChEMBL + ZINC.

Pipeline:
  1. ChEMBL drug-like subset (MW < 500, logP < 5, rotors < 10, QED > 0.4,
     heavy atoms >= 12).
  2. Random sample from ZINC20 "drug-like" tranche(s).
  3. RDKit canonicalization + dedup by InChIKey.
  4. Shuffle, split into train.jsonl + val.jsonl (0.5%).

ChEMBL is the easier source — we pull from the ChEMBL FTP PostgreSQL SQL
dumps via RDKit's chembl loader, but for a hermetic run we prefer the
pre-processed ChEMBL canonical SMILES flatfile mirrored by the community.

Run in-container:
  apptainer exec --nv container/chem_rlvr.sif \
      python training/sft/prepare_stage0a.py \
          --chembl-smi /workspace/data/sft/raw/chembl_35.smi \
          --zinc-dir   /workspace/data/sft/raw/zinc20_druglike \
          --out-dir    /workspace/data/sft/stage0a \
          --target-size 5000000

Outputs:
  <out-dir>/train.jsonl
  <out-dir>/val.jsonl
  <out-dir>/corpus_stats.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED

RDLogger.DisableLog("rdApp.*")


# --- Filters ----------------------------------------------------------------

def drug_like(mol: Chem.Mol) -> bool:
    if mol is None or mol.GetNumHeavyAtoms() < 12:
        return False
    mw = Descriptors.MolWt(mol)
    if not (150 <= mw <= 500):
        return False
    try:
        logp = Descriptors.MolLogP(mol)
    except Exception:
        return False
    if logp > 5.0:
        return False
    if Descriptors.NumRotatableBonds(mol) > 10:
        return False
    try:
        if QED.qed(mol) < 0.4:
            return False
    except Exception:
        return False
    return True


def canonical_or_none(smi: str) -> tuple[str | None, str | None]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None or not drug_like(mol):
        return None, None
    # Drop salts: keep largest fragment
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        canon = Chem.MolToSmiles(mol, canonical=True)
        ikey = Chem.MolToInchiKey(mol)
    except Exception:
        return None, None
    return canon, ikey


# --- Sources ----------------------------------------------------------------

def iter_smi_file(path: Path) -> Iterator[str]:
    """Iterate SMILES from a plain .smi / .txt / .smi.gz file. Assumes first
    token on each line is the SMILES."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for ln in f:
            s = ln.split()[0].strip() if ln.strip() else ""
            if s and not s.startswith("#") and not s.lower().startswith("smiles"):
                yield s


def iter_zinc_dir(dirpath: Path) -> Iterator[str]:
    """ZINC20 tranches typically ship as many .smi / .smi.gz shards under a
    directory tree. We walk and flatten."""
    for p in dirpath.rglob("*.smi*"):
        yield from iter_smi_file(p)


# --- Driver -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl-smi", type=Path, required=True,
                    help="ChEMBL canonical SMILES flatfile (.smi or .smi.gz)")
    ap.add_argument("--zinc-dir", type=Path, default=None,
                    help="Root of a ZINC20 drug-like tranche directory tree")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-size", type=int, default=5_000_000)
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    seen_ikeys: set[str] = set()
    kept = 0
    source_counts = Counter()

    def _consume(src_name: str, smiles_iter: Iterable[str], limit: int | None):
        nonlocal kept
        for raw in smiles_iter:
            if limit is not None and source_counts[src_name] >= limit:
                return
            canon, ikey = canonical_or_none(raw)
            if canon is None or ikey in seen_ikeys:
                continue
            seen_ikeys.add(ikey)
            source_counts[src_name] += 1
            kept += 1
            yield canon
            if kept >= args.target_size:
                return

    out_path = args.out_dir / "all_canon.jsonl"
    n_chembl_target = args.target_size // 2
    n_zinc_target = args.target_size - n_chembl_target

    print(f"Writing canonicalized SMILES to {out_path} (target {args.target_size})")
    with out_path.open("w") as out:
        # ChEMBL
        for s in _consume("chembl", iter_smi_file(args.chembl_smi), n_chembl_target):
            out.write(json.dumps({"text": s, "src": "chembl"}) + "\n")
        # ZINC
        if args.zinc_dir and args.zinc_dir.exists():
            for s in _consume("zinc", iter_zinc_dir(args.zinc_dir), n_zinc_target):
                out.write(json.dumps({"text": s, "src": "zinc"}) + "\n")
        else:
            print("No ZINC dir supplied; ChEMBL-only corpus.")

    # Shuffle in place and split
    print("Shuffling + splitting...")
    lines = out_path.read_text().splitlines()
    rng.shuffle(lines)
    n_val = max(1, int(len(lines) * args.val_frac))
    val = lines[:n_val]
    train = lines[n_val:]
    (args.out_dir / "train.jsonl").write_text("\n".join(train) + "\n")
    (args.out_dir / "val.jsonl").write_text("\n".join(val) + "\n")

    stats = {
        "kept_total": len(lines),
        "train": len(train),
        "val": len(val),
        "source_counts": dict(source_counts),
        "unique_inchikeys": len(seen_ikeys),
    }
    (args.out_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
