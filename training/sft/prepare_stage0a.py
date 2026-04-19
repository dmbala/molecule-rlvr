"""Assemble the Stage 0a SFT corpus: ~5M drug-like SMILES from ChEMBL + ZINC.

Pipeline:
  1. ChEMBL drug-like subset (MW 150-500, logP <= 5, rotors <= 10, QED >= 0.4).
  2. Random sample from ZINC20 "drug-like" tranche(s).
  3. RDKit canonicalization + dedup by InChIKey.
  4. Shuffle, split into train.jsonl + val.jsonl (0.5%).

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
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from smiles_utils import canonical_or_none, iter_smi_dir, iter_smi_file


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
    records: list[tuple[str, str]] = []      # (canonical_smiles, src)
    source_counts: Counter[str] = Counter()

    def _consume(src_name: str, smiles_iter: Iterable[str], limit: int | None) -> None:
        for raw in smiles_iter:
            if limit is not None and source_counts[src_name] >= limit:
                return
            canon, ikey = canonical_or_none(raw)
            if canon is None or ikey in seen_ikeys:
                continue
            seen_ikeys.add(ikey)
            source_counts[src_name] += 1
            records.append((canon, src_name))
            if len(records) >= args.target_size:
                return

    n_chembl_target = args.target_size // 2
    n_zinc_target = args.target_size - n_chembl_target

    print(f"Canonicalizing (target {args.target_size})")
    _consume("chembl", iter_smi_file(args.chembl_smi), n_chembl_target)
    if args.zinc_dir and args.zinc_dir.exists():
        _consume("zinc", iter_smi_dir(args.zinc_dir), n_zinc_target)
    else:
        print("No ZINC dir supplied; ChEMBL-only corpus.")

    print(f"Collected {len(records)}; shuffling + splitting")
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    val_recs, train_recs = records[:n_val], records[n_val:]

    def _dump(path: Path, recs: list[tuple[str, str]]) -> None:
        with path.open("w") as f:
            for smi, src in recs:
                f.write(json.dumps({"text": smi, "src": src}) + "\n")

    _dump(args.out_dir / "train.jsonl", train_recs)
    _dump(args.out_dir / "val.jsonl", val_recs)

    stats = {
        "kept_total": len(records),
        "train": len(train_recs),
        "val": len(val_recs),
        "source_counts": dict(source_counts),
        "unique_inchikeys": len(seen_ikeys),
    }
    (args.out_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
