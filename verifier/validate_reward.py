"""Three-group ranking test (plan Fix 2 validation).

Runs the full composite reward over:
  1. Known Mpro inhibitors  — expected top decile
  2. N_random  random drug-like ZINC molecules — expected middle
  3. N_junk    junk molecules (alkanes, small fragments) — expected bottom

Acceptance criterion:
  top-decile(known ∪ random ∪ junk) ⊇ ≥ 80% of known inhibitors.

Run in-container:
  apptainer exec --nv chem_rlvr.sif python verifier/validate_reward.py \
      --config training/configs/verifier.yaml \
      --known data/reference/known_mpro_inhibitors.csv \
      --zinc data/reference/zinc_random.smi \
      --n-random 1000 --n-junk 1000 \
      --out analysis/reward_validation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import string
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rdkit import Chem

from chemistry_verifier import verify_group
from verifier_config_loader import load_verifier_config

log = logging.getLogger("validate_reward")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# --- Junk molecule generator ------------------------------------------------

def generate_junk_smiles(n: int, rng: random.Random) -> list[str]:
    """Adversarial negative controls: structurally implausible for drug-likeness.

    Mix of (a) long alkanes, (b) random atom chains, (c) single fragments,
    (d) nonsense-but-valid SMILES.
    """
    out: list[str] = []
    for _ in range(n):
        kind = rng.randint(0, 3)
        if kind == 0:  # long alkane
            out.append("C" * rng.randint(20, 60))
        elif kind == 1:  # random halogenated chain
            atoms = "".join(rng.choice("CCCCCCCOFNClBr") for _ in range(rng.randint(5, 15)))
            out.append(atoms)
        elif kind == 2:  # tiny fragment
            out.append(rng.choice(["C", "CC", "O", "CCO", "N", "CN", "CCCN", "C=C"]))
        else:  # random permutation of plausible atoms + bonds
            parts = ["".join(rng.choices(string.ascii_uppercase + "12()=", k=rng.randint(4, 10)))
                     for _ in range(1)]
            out.append(parts[0])
    return out


def load_smiles_list(path: Path, column: str | None = None, limit: int | None = None) -> list[str]:
    """Accepts .csv (optionally with named column), .smi, .txt (one SMILES per line)."""
    lines = path.read_text().splitlines()
    if not lines:
        return []
    smiles: list[str]
    if path.suffix.lower() == ".csv":
        header = lines[0].split(",")
        col_idx = header.index(column) if column and column in header else 0
        smiles = []
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) > col_idx:
                smiles.append(parts[col_idx].strip())
    else:
        smiles = [ln.split()[0].strip() for ln in lines if ln.strip()]
    smiles = [s for s in smiles if s and not s.startswith("#")]
    if limit is not None:
        smiles = smiles[:limit]
    return smiles


# --- Evaluation -------------------------------------------------------------

def wrap_as_response(smi: str) -> str:
    """Match the format extract_smiles() expects."""
    return f"<answer>{smi}</answer>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--known", required=True, type=Path,
                    help="CSV of known Mpro inhibitors with a `smiles` column")
    ap.add_argument("--zinc", type=Path, default=None,
                    help="Optional .smi file of random ZINC drug-like molecules")
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--n-junk", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-decile-min-known-frac", type=float, default=0.8,
                    help="Acceptance threshold (Fix 2).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = load_verifier_config(str(args.config))

    known = load_smiles_list(args.known, column="smiles")
    if not known:
        raise SystemExit(f"No SMILES loaded from {args.known}")

    random_smi: list[str] = []
    if args.zinc and args.zinc.exists():
        random_smi = load_smiles_list(args.zinc, limit=args.n_random)
    if len(random_smi) < args.n_random:
        log.warning("Only %d ZINC SMILES available; proceeding.", len(random_smi))

    junk_smi = generate_junk_smiles(args.n_junk, rng)

    all_smi = known + random_smi + junk_smi
    labels = (["known"] * len(known)
              + ["random"] * len(random_smi)
              + ["junk"] * len(junk_smi))
    log.info("Scoring: known=%d random=%d junk=%d", len(known), len(random_smi), len(junk_smi))

    # Score each molecule as its own singleton group (no diversity signal).
    # For the sanity test we care about the other reward components.
    records = []
    for smi, lbl in zip(all_smi, labels):
        rec = verify_group([wrap_as_response(smi)], cfg, step=9999)[0]
        d = asdict(rec)
        d["label"] = lbl
        d["source_smiles"] = smi
        records.append(d)

    rewards = np.array([r["reward"] for r in records])
    order = np.argsort(-rewards)  # descending
    top_decile = set(order[: max(1, len(rewards) // 10)].tolist())

    known_idx = {i for i, r in enumerate(records) if r["label"] == "known"}
    known_in_top = len(known_idx & top_decile)
    known_frac = known_in_top / max(1, len(known_idx))

    summary = {
        "n_known": len(known),
        "n_random": len(random_smi),
        "n_junk": len(junk_smi),
        "top_decile_size": len(top_decile),
        "known_in_top_decile": known_in_top,
        "known_top_decile_fraction": known_frac,
        "acceptance_threshold": args.top_decile_min_known_frac,
        "passed": known_frac >= args.top_decile_min_known_frac,
        "mean_reward_by_label": {
            lbl: float(np.mean([r["reward"] for r in records if r["label"] == lbl]))
            for lbl in ("known", "random", "junk") if any(r["label"] == lbl for r in records)
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)

    log.info("Summary: %s", json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(
            f"FAIL: only {known_in_top}/{len(known_idx)} known inhibitors landed "
            f"in the top decile (need {args.top_decile_min_known_frac:.0%}). "
            "Fix the reward before starting RL."
        )
    log.info("PASS")


if __name__ == "__main__":
    main()
