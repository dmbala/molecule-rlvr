"""Stage 0c — assemble a small target-specific corpus: known Mpro inhibitors +
ChEMBL hits tagged against SARS-CoV-2 Mpro + published SARS / MERS inhibitor
series.

Goal per the plan: ~5k molecules. Keeps target-specific priors in the weights
without overfitting to 12 known inhibitors.

Two data paths:
  A) ChEMBL Mpro bioactivity dump (`chembl_mpro.tsv`) — pulled via the user's
     preferred ChEMBL fetch step; we read whatever is provided as a TSV with
     a `canonical_smiles` column.
  B) Fallback: expand the 12 known inhibitors into near-analogs via RDKit
     bond-rotation / R-group swap (simple enumeration, useful when offline).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS

RDLogger.DisableLog("rdApp.*")


def load_known(path: Path) -> list[str]:
    return [r["smiles"].strip()
            for r in csv.DictReader(path.open())
            if r.get("smiles", "").strip()]


def load_chembl_mpro(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    out = []
    with path.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            s = row.get("canonical_smiles", "").strip()
            if s:
                out.append(s)
    return out


def brics_enumerate(seeds: list[str], n_per_seed: int, rng: random.Random) -> list[str]:
    """Fragment each seed via BRICS and recombine to generate analogs.

    Not chemistry-reactant-aware (so not every output is synthesizable), but
    produces plausible graph-variants that keep the scaffold's pharmacophore.
    """
    out: set[str] = set()
    all_frags: set[str] = set()
    for s in seeds:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            frags = BRICS.BRICSDecompose(m, keepNonLeafNodes=True)
            all_frags.update(frags)
        except Exception:
            continue

    frag_list = list(all_frags)
    if len(frag_list) < 4:
        return []

    for s in seeds:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        local_frags = list(BRICS.BRICSDecompose(m))
        pool = set(local_frags) | set(rng.sample(frag_list, min(len(frag_list), 30)))
        pool_mols = [Chem.MolFromSmiles(f) for f in pool]
        pool_mols = [x for x in pool_mols if x is not None]
        try:
            builder = BRICS.BRICSBuild(pool_mols, onlyUseReactionsOnce=True)
            for i, built in enumerate(builder):
                if i >= n_per_seed:
                    break
                built.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(built, catchErrors=True)
                out.add(Chem.MolToSmiles(built))
        except Exception:
            continue

    return list(out)


def canonicalize_and_dedup(smiles: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        frags = Chem.GetMolFrags(m, asMols=True)
        if len(frags) > 1:
            m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        canon = Chem.MolToSmiles(m, canonical=True)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=Path, required=True)
    ap.add_argument("--chembl-mpro-tsv", type=Path, default=None,
                    help="Optional TSV with `canonical_smiles` column for CHEMBL Mpro activities")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-size", type=int, default=5000)
    ap.add_argument("--analogs-per-seed", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    known = load_known(args.known)
    chembl = load_chembl_mpro(args.chembl_mpro_tsv)

    pool: list[str] = list(known)
    pool.extend(chembl)

    if len(pool) < args.target_size:
        need = args.target_size - len(pool)
        print(f"Augmenting with BRICS analogs; need {need} more.")
        analogs = brics_enumerate(known, args.analogs_per_seed, rng)
        pool.extend(analogs[:need])

    canon = canonicalize_and_dedup(pool)
    rng.shuffle(canon)
    canon = canon[:args.target_size]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for s in canon:
            f.write(json.dumps({"text": s, "src": "stage0c"}) + "\n")

    print(f"Wrote {len(canon)} SMILES to {args.out}")


if __name__ == "__main__":
    main()
