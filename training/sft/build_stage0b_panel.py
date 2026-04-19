"""Build the (context, SMILES, vina_score, label) panel that feeds
generate_r1_cot.py. This is the labor-intensive precursor: it actually runs
Vina across the sampled molecules so R1 has real numbers to explain.

Inputs:
  --known    data/reference/known_mpro_inhibitors.csv     (strong binders)
  --zinc-smi data/reference/zinc_random.smi               (pulled, will be weak/mixed)
  --chembl   data/sft/stage0a/train.jsonl                 (mixed; sampled for "medium" items)
  --receptor data/receptors/7L13/receptor.pdbqt           (+ receptor.config for grid)
  --out      data/sft/stage0b/panel.jsonl

Emits ~5 000 items (tunable). Requires the receptor to already be prepped
and redock-validated. Running this against 5k ligands on 64 cores takes
~4–8 hours; schedule as a CPU-only Slurm job.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from rdkit import Chem

from reward_components_ext import run_docking_with_config

# NB: keep this importable separately from the RL training path so Stage 0b
# can run without the full OpenRLHF stack loaded. `reward_components_ext`
# is a thin shim that wraps `verifier/reward_components.py` for CLI use.


MPRO_CONTEXT = (
    "SARS-CoV-2 Mpro active site. Catalytic dyad His41 / Cys145. "
    "S1 pocket: His163, Glu166, Phe140, Leu141, Asn142. "
    "S2 pocket: Met49, Tyr54, His41. "
    "S4 sub-pocket: Gln192, Thr190, Ala191, Leu167."
)

LABEL_STRONG = "strong"
LABEL_MEDIUM = "medium"
LABEL_WEAK = "weak"
LABELS = (LABEL_STRONG, LABEL_MEDIUM, LABEL_WEAK)
STRONG_THRESHOLD = -8.0
MEDIUM_THRESHOLD = -6.0


def iter_jsonl(path: Path):
    for ln in path.read_text().splitlines():
        if ln.strip():
            yield json.loads(ln)


def label_for(score: float) -> str:
    if score <= STRONG_THRESHOLD:
        return LABEL_STRONG
    if score <= MEDIUM_THRESHOLD:
        return LABEL_MEDIUM
    return LABEL_WEAK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=Path, required=True)
    ap.add_argument("--zinc-smi", type=Path, default=None)
    ap.add_argument("--chembl", type=Path, default=None)
    ap.add_argument("--receptor-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-strong", type=int, default=200)
    ap.add_argument("--n-medium", type=int, default=2000)
    ap.add_argument("--n-weak", type=int, default=2800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Collect candidate SMILES per bucket ---------------------------------
    known = [r["smiles"] for r in csv.DictReader(args.known.open())]
    zinc = ([ln.split()[0].strip() for ln in args.zinc_smi.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
            if args.zinc_smi else [])
    chembl = ([d["text"] for d in iter_jsonl(args.chembl)] if args.chembl else [])

    candidates = {
        LABEL_STRONG: known,                                      # expected strong
        LABEL_MEDIUM: rng.sample(chembl, min(len(chembl), 20000)) if chembl else [],
        LABEL_WEAK:   rng.sample(zinc, min(len(zinc), 20000)) if zinc else [],
    }

    targets = {
        LABEL_STRONG: args.n_strong,
        LABEL_MEDIUM: args.n_medium,
        LABEL_WEAK: args.n_weak,
    }
    kept: dict[str, list[dict]] = {k: [] for k in targets}

    # 2. Dock each candidate until we have enough in each bucket -------------
    with args.out.open("w") as fout:
        idx = 0
        for bucket, smiles_list in candidates.items():
            need = targets[bucket]
            for smi in smiles_list:
                if len(kept[bucket]) >= need:
                    break
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                try:
                    score, strain = run_docking_with_config(mol, args.receptor_dir, n_seeds=3)
                except Exception as e:
                    print(f"[skip {smi}] {e}")
                    continue
                if score == float("inf"):
                    continue
                obs_bucket = label_for(score)
                # Keep only if it falls into the bucket we're currently filling,
                # OR if we still need more of its actual bucket.
                if obs_bucket == bucket or len(kept[obs_bucket]) < targets[obs_bucket]:
                    item = {
                        "id": f"panel_{idx:06d}",
                        "context": MPRO_CONTEXT,
                        "smiles": Chem.MolToSmiles(mol),
                        "vina_score": float(score),
                        "label": obs_bucket,
                        "source": bucket,
                    }
                    fout.write(json.dumps(item) + "\n")
                    kept[obs_bucket].append(item)
                    idx += 1
                    if idx % 100 == 0:
                        print("kept: " + " ".join(f"{k}={len(v)}" for k, v in kept.items()))

    print("Final:", {k: len(v) for k, v in kept.items()}, "->", args.out)


if __name__ == "__main__":
    main()
