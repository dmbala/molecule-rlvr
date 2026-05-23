"""Generate the 300-prompt dataset (240 train / 60 eval) per plan Fix 6 + 8.2.

Usage:
    python data/prompts/generate_prompts.py \
        --known-inhibitors data/reference/known_mpro_inhibitors.csv \
        --zinc-smi         data/reference/zinc_random.smi \
        --out              data/prompts/mpro_prompts.jsonl \
        --n-train 240 --n-eval 60 --seed 42

For Amendment v2 (Arm C-PC) emit role-paired prompts and optionally mix in a
DiffSBDD-generated seed bank:

    python data/prompts/generate_prompts.py ... \
        --diffsbdd-seeds   data/seeds/diffsbdd_7L13.jsonl \
        --diffsbdd-frac    0.40 \
        --emit-roles       pc \
        --out              data/prompts/mpro_prompts_pc.jsonl

Implements the four variation axes and the ≥80% lead-optimization bias.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# role_templates lives under training/pilot; add it to sys.path so we can
# reuse CRITIC_USER_TEMPLATE without copying the string.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "training" / "pilot"))
from role_templates import CRITIC_USER_TEMPLATE_SENTINELS  # noqa: E402


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
    # Amendment v2 (Arm C-PC). When --emit-roles pc, each Prompt is expanded
    # into a (proposer, critic) pair sharing a `root_id`. The critic's
    # instruction text carries `__PARENT_*__` sentinels that the rollout
    # collator replaces at RL time with the proposer's actual SMILES / dock /
    # reward. Both records carry an `<rlvr_tags>` block consumed by
    # `reward_fn_openrlhf_pc`.
    root_id: str | None = None
    role: str | None = None


_RLVR_TAG_FMT_PROPOSER = (
    "\n\n<rlvr_tags>"
    "<root_id>{root_id}</root_id>"
    "<role>proposer</role>"
    "</rlvr_tags>"
)


_RLVR_TAG_FMT_CRITIC = (
    "\n\n<rlvr_tags>"
    "<root_id>{root_id}</root_id>"
    "<role>critic</role>"
    "<parent_smiles>__PARENT_SMILES__</parent_smiles>"
    "<parent_reward>__PARENT_REWARD__</parent_reward>"
    "<parent_dock>__PARENT_DOCK__</parent_dock>"
    "</rlvr_tags>"
)


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


def build_seed_pool(known: list[str], zinc: list[str], diffsbdd: list[str],
                     rng: random.Random) -> dict[str, list[str]]:
    """Seed pool per provenance type. Known-inhibitor fragments use Murcko scaffolds."""
    known_frags = [f for f in (murcko_fragment(s) for s in known) if f]
    # Drop duplicates while preserving order
    seen = set()
    known_frags = [f for f in known_frags if not (f in seen or seen.add(f))]
    rng.shuffle(known_frags)
    rng.shuffle(zinc)
    rng.shuffle(diffsbdd)
    return {
        "known_inhibitor_fragment": known_frags,
        "zinc_random": zinc,
        "diffsbdd": diffsbdd,
        "none": [""],
        "prior_turn": [],  # filled at rollout time during multi-turn training
    }


def load_diffsbdd_seeds(path: Path | None, top_k: int | None = None) -> list[str]:
    """Read SMILES from the DiffSBDD seed bank JSONL.

    Schema: `{smiles, dock_kcal, qed, sa_raw, scaffold, source, cd2020_max_tani}`.
    Sort by `dock_kcal` ascending (more negative = better) and optionally
    take the top-K so the seed bank biases toward strong starting molecules.
    """
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    rows = [r for r in rows if r.get("smiles") and r.get("dock_kcal") is not None]
    rows.sort(key=lambda r: r["dock_kcal"])
    if top_k is not None:
        rows = rows[: top_k]
    return [r["smiles"] for r in rows]


def build_leadopt_schedule(base: list[str], extra: str,
                            n_slots: int, n_extra: int) -> list[str]:
    """Return an n_slots-long list of provenance tags. `extra` takes exactly
    n_extra slots interleaved across the span; the remainder cycles through
    `base`."""
    if n_extra <= 0:
        return [base[i % len(base)] for i in range(n_slots)]
    n_extra = min(n_extra, n_slots)
    extra_positions = set(
        round(i * n_slots / n_extra) for i in range(n_extra)
    )
    schedule: list[str] = []
    base_cursor = 0
    for i in range(n_slots):
        if i in extra_positions:
            schedule.append(extra)
        else:
            schedule.append(base[base_cursor % len(base)])
            base_cursor += 1
    return schedule


def generate(
    known: list[str],
    zinc: list[str],
    diffsbdd: list[str],
    *,
    n_train: int,
    n_eval: int,
    leadopt_frac: float,
    diffsbdd_frac: float,
    rng: random.Random,
) -> list[Prompt]:
    """Build the prompt set. `diffsbdd_frac` caps the share of *lead-opt* slots
    that draw seeds from the DiffSBDD bank; the rest is split between
    known-inhibitor fragments and ZINC random seeds. Set diffsbdd_frac=0 to
    reproduce the pre-Amendment-v2 distribution.
    """
    seed_pool = build_seed_pool(known, zinc, diffsbdd, rng)
    base_leadopt = ["known_inhibitor_fragment", "zinc_random"]
    pocket_foci = list(POCKET_CONTEXTS.keys())
    strictnesses = list(STRICTNESS_THRESHOLDS.keys())
    styles = list(INSTR_TEMPLATES.keys())

    combos = list(product(pocket_foci, strictnesses, styles))
    prompts: list[Prompt] = []
    idx = 1

    n_total = n_train + n_eval
    n_leadopt = int(round(leadopt_frac * n_total))

    # Cap the diffsbdd fraction at what the seed bank can actually deliver.
    n_diffsbdd_target = int(round(diffsbdd_frac * n_leadopt))
    if not seed_pool["diffsbdd"]:
        n_diffsbdd_target = 0

    # Provenance schedule across lead-opt slots: interleave diffsbdd with the
    # base provenances so the split rebalancer doesn't strip diffsbdd seeds.
    provenances_leadopt = build_leadopt_schedule(
        base_leadopt, "diffsbdd", n_leadopt, n_diffsbdd_target,
    )

    # Fill lead-opt slots first
    for i in range(n_leadopt):
        prov = provenances_leadopt[i]
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


def expand_to_role_pairs(prompts: list[Prompt]) -> list[Prompt]:
    """Take a list of (single-role) Prompts and return twice as many — each
    original prompt yields a `_p` proposer record and a `_c` critic record
    sharing a `root_id` field. Used by --emit-roles pc."""
    out: list[Prompt] = []
    for p in prompts:
        root_id = p.id
        # Proposer: keep the original instruction + rlvr_tags
        proposer = Prompt(
            id=f"{root_id}_p",
            split=p.split,
            context=p.context,
            seed_smiles=p.seed_smiles,
            seed_provenance=p.seed_provenance,
            instruction=p.instruction + _RLVR_TAG_FMT_PROPOSER.format(root_id=root_id),
            reward_strictness=p.reward_strictness,
            pocket_focus=p.pocket_focus,
            instruction_style=p.instruction_style,
            turn=p.turn,
            root_id=root_id,
            role="proposer",
        )
        # Critic: render the critic user template with sentinels. The collator
        # at RL time replaces __PARENT_*__ with the proposer's actual output.
        critic_instruction = CRITIC_USER_TEMPLATE_SENTINELS.format(
            context=p.context,
            instruction=p.instruction,
        ) + _RLVR_TAG_FMT_CRITIC.format(root_id=root_id)
        critic = Prompt(
            id=f"{root_id}_c",
            split=p.split,
            context=p.context,
            seed_smiles=p.seed_smiles,
            seed_provenance=p.seed_provenance,
            instruction=critic_instruction,
            reward_strictness=p.reward_strictness,
            pocket_focus=p.pocket_focus,
            instruction_style=p.instruction_style,
            turn=p.turn,
            root_id=root_id,
            role="critic",
        )
        out.append(proposer)
        out.append(critic)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-inhibitors", type=Path, required=True)
    ap.add_argument("--zinc-smi", type=Path, default=None)
    ap.add_argument("--diffsbdd-seeds", type=Path, default=None,
                    help="Optional JSONL seed bank from data/seeds/build_seed_bank.py (Amendment v2)")
    ap.add_argument("--diffsbdd-frac", type=float, default=0.40,
                    help="Fraction of lead-opt slots seeded from the DiffSBDD bank")
    ap.add_argument("--diffsbdd-top-k", type=int, default=2000,
                    help="Use the K best-docked DiffSBDD seeds (sorted by Vina)")
    ap.add_argument("--emit-roles", choices=["single", "pc"], default="single",
                    help="`pc` emits paired proposer/critic records for Arm C-PC")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--leadopt-frac", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    known = load_column(args.known_inhibitors, "smiles")
    zinc = load_smiles(args.zinc_smi, limit=2000) if args.zinc_smi else []
    diffsbdd = load_diffsbdd_seeds(args.diffsbdd_seeds, top_k=args.diffsbdd_top_k)
    if not zinc:
        print(f"Warning: no ZINC SMILES found at {args.zinc_smi}; using known-inhibitor fragments only for lead-opt seeds.")
    if args.diffsbdd_seeds and not diffsbdd:
        print(f"Warning: no DiffSBDD seeds loaded from {args.diffsbdd_seeds}; diffsbdd_frac will be ignored.")

    prompts = generate(known, zinc, diffsbdd,
                       n_train=args.n_train, n_eval=args.n_eval,
                       leadopt_frac=args.leadopt_frac,
                       diffsbdd_frac=args.diffsbdd_frac, rng=rng)

    if args.emit_roles == "pc":
        prompts = expand_to_role_pairs(prompts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in prompts:
            f.write(json.dumps(asdict(p)) + "\n")

    n_train = sum(1 for p in prompts if p.split == "train")
    n_eval = sum(1 for p in prompts if p.split == "eval")
    n_leadopt = sum(1 for p in prompts if p.seed_smiles)
    n_diffsbdd = sum(1 for p in prompts if p.seed_provenance == "diffsbdd")
    print(f"Wrote {len(prompts)} prompts to {args.out}")
    print(f"  train={n_train} eval={n_eval} leadopt={n_leadopt} ({n_leadopt / len(prompts):.0%})")
    if diffsbdd:
        print(f"  diffsbdd-seeded={n_diffsbdd} ({n_diffsbdd / max(1, n_leadopt):.0%} of lead-opt)")
    if args.emit_roles == "pc":
        print(f"  emit-roles=pc → {len(prompts) // 2} root prompts × 2 (proposer + critic)")


if __name__ == "__main__":
    main()
