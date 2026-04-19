"""Generate the 300-prompt dataset (240 train / 60 eval) per plan Fix 6 + 8.2.

Usage:
    python data/prompts/generate_prompts.py \
        --known-inhibitors data/reference/known_mpro_inhibitors.csv \
        --zinc-smi         data/reference/zinc_random.smi \
        --out              data/prompts/mpro_prompts.jsonl \
        --n-train 240 --n-eval 60 --seed 42

Implements the four variation axes and the ≥80% lead-optimization bias.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


POCKET_CONTEXTS = {
    "all": "SARS-CoV-2 Mpro active site. Key residues: His41, Cys145, Met49, Glu166, Gln189, His163, Phe140.",
    "catalytic_dyad": "SARS-CoV-2 Mpro active site. Catalytic dyad: His41 and Cys145. Nearby: Met49, Met165.",
    "s1": "SARS-CoV-2 Mpro S1 pocket. Residues: His163, Glu166, Phe140, Leu141, Asn142. Prefer H-bond partners for Glu166.",
    "s4": "SARS-CoV-2 Mpro S4 sub-pocket. Residues: Gln192, Thr190, Ala191, Leu167. Hydrophobic contacts favored.",
}

STRICTNESS_THRESHOLDS = {"loose": -6.0, "medium": -7.0, "tight": -8.0}

INSTR_TEMPLATES = {
    "terse": {
        "de_novo": "Propose a SMILES for a non-covalent Mpro inhibitor. Target Vina score < {th} kcal/mol.",
        "lead_opt": "Modify the SMILES above to improve Vina docking against Mpro (target < {th} kcal/mol).",
    },
    "detailed": {
        "de_novo": (
            "Design a non-covalent SARS-CoV-2 Mpro inhibitor. "
            "Keep QED > 0.5, SA < 5, and MW < 500. Target Vina score < {th} kcal/mol. "
            "Briefly justify your design using the pocket residues before giving the final SMILES on its own line."
        ),
        "lead_opt": (
            "Starting from the scaffold above, propose a modified SMILES that improves Vina docking "
            "score against SARS-CoV-2 Mpro (target < {th} kcal/mol) while keeping QED > 0.5 and SA < 5. "
            "Briefly justify your edit using specific pocket residues."
        ),
    },
    "constraints_first": {
        "de_novo": (
            "Constraints: QED > 0.5; SA < 5; MW < 500; no PAINS alerts.\n"
            "Goal: Vina dock < {th} kcal/mol against SARS-CoV-2 Mpro.\n\n"
            "Task: propose a valid SMILES satisfying the constraints."
        ),
        "lead_opt": (
            "Constraints: QED > 0.5; SA < 5; MW < 500; no PAINS alerts.\n"
            "Goal: Vina dock < {th} kcal/mol against SARS-CoV-2 Mpro.\n\n"
            "Task: modify the seed SMILES above to satisfy the constraints."
        ),
    },
}


@dataclass
class Prompt:
    id: str
    split: str
    context: str
    seed_smiles: str | None
    seed_provenance: str
    instruction: str
    reward_strictness: str
    pocket_focus: str
    instruction_style: str
    turn: int = 1


def load_column(path: Path, column: str) -> list[str]:
    rows = list(csv.DictReader(path.open()))
    return [r[column].strip() for r in rows if r.get(column, "").strip()]


def load_smiles(path: Path, limit: int | None = None) -> list[str]:
    if not path or not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.split()[0].strip() if line.strip() else ""
        if s and not s.startswith("#"):
            out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def murcko_fragment(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf) if scaf and scaf.GetNumAtoms() >= 5 else None


def build_seed_pool(known: list[str], zinc: list[str], rng: random.Random) -> dict[str, list[str]]:
    """Seed pool per provenance type. Known-inhibitor fragments use Murcko scaffolds."""
    known_frags = [f for f in (murcko_fragment(s) for s in known) if f]
    # Drop duplicates while preserving order
    seen = set()
    known_frags = [f for f in known_frags if not (f in seen or seen.add(f))]
    rng.shuffle(known_frags)
    rng.shuffle(zinc)
    return {
        "known_inhibitor_fragment": known_frags,
        "zinc_random": zinc,
        "none": [""],
        "prior_turn": [],  # filled at rollout time during multi-turn training
    }


def generate(
    known: list[str],
    zinc: list[str],
    *,
    n_train: int,
    n_eval: int,
    leadopt_frac: float,
    rng: random.Random,
) -> list[Prompt]:
    seed_pool = build_seed_pool(known, zinc, rng)
    provenances_leadopt = ["known_inhibitor_fragment", "zinc_random"]
    pocket_foci = list(POCKET_CONTEXTS.keys())
    strictnesses = list(STRICTNESS_THRESHOLDS.keys())
    styles = list(INSTR_TEMPLATES.keys())

    combos = list(product(pocket_foci, strictnesses, styles))
    prompts: list[Prompt] = []
    idx = 1

    n_total = n_train + n_eval
    n_leadopt = int(round(leadopt_frac * n_total))

    # Fill lead-opt slots first
    for i in range(n_leadopt):
        prov = provenances_leadopt[i % len(provenances_leadopt)]
        pool = seed_pool[prov]
        if not pool:
            continue
        seed = pool[i % len(pool)]
        pf, strict, style = combos[i % len(combos)]
        split = "train" if i < int(leadopt_frac * n_train) else "eval"
        instr = INSTR_TEMPLATES[style]["lead_opt"].format(th=STRICTNESS_THRESHOLDS[strict])
        full_instr = (
            f"Seed SMILES: {seed}\n\n{instr}"
            if style != "constraints_first"
            else f"Seed SMILES: {seed}\n\n{instr}"
        )
        prompts.append(
            Prompt(
                id=f"mpro_{idx:04d}",
                split=split,
                context=POCKET_CONTEXTS[pf],
                seed_smiles=seed,
                seed_provenance=prov,
                instruction=full_instr,
                reward_strictness=strict,
                pocket_focus=pf,
                instruction_style=style,
            )
        )
        idx += 1

    # Fill remaining slots as de novo
    n_remaining = n_total - len(prompts)
    for i in range(n_remaining):
        pf, strict, style = combos[(i + idx) % len(combos)]
        split = "train" if i < max(0, n_train - sum(p.split == "train" for p in prompts)) else "eval"
        instr = INSTR_TEMPLATES[style]["de_novo"].format(th=STRICTNESS_THRESHOLDS[strict])
        prompts.append(
            Prompt(
                id=f"mpro_{idx:04d}",
                split=split,
                context=POCKET_CONTEXTS[pf],
                seed_smiles=None,
                seed_provenance="none",
                instruction=instr,
                reward_strictness=strict,
                pocket_focus=pf,
                instruction_style=style,
            )
        )
        idx += 1

    # Final split rebalancing — ensure exactly n_train / n_eval
    rng.shuffle(prompts)
    train = [p for p in prompts if p.split == "train"]
    evals = [p for p in prompts if p.split == "eval"]
    # Trim/pad
    while len(train) > n_train and evals:
        p = train.pop()
        p.split = "eval"
        evals.append(p)
    while len(evals) > n_eval and train:
        p = evals.pop()
        p.split = "train"
        train.append(p)
    return train + evals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-inhibitors", type=Path, required=True)
    ap.add_argument("--zinc-smi", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--leadopt-frac", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    known = load_column(args.known_inhibitors, "smiles")
    zinc = load_smiles(args.zinc_smi, limit=2000) if args.zinc_smi else []
    if not zinc:
        print(f"Warning: no ZINC SMILES found at {args.zinc_smi}; using known-inhibitor fragments only for lead-opt seeds.")

    prompts = generate(known, zinc,
                       n_train=args.n_train, n_eval=args.n_eval,
                       leadopt_frac=args.leadopt_frac, rng=rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in prompts:
            f.write(json.dumps(asdict(p)) + "\n")

    n_train = sum(1 for p in prompts if p.split == "train")
    n_eval = sum(1 for p in prompts if p.split == "eval")
    n_leadopt = sum(1 for p in prompts if p.seed_smiles)
    print(f"Wrote {len(prompts)} prompts to {args.out}")
    print(f"  train={n_train} eval={n_eval} leadopt={n_leadopt} ({n_leadopt / len(prompts):.0%})")


if __name__ == "__main__":
    main()
