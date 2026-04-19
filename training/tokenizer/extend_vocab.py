"""Extend a base tokenizer with atom-in-SMILES tokens (plan Fix 9.3 Option A).

Standard BPE tokenizers fragment `[C@@H](=O)` etc. into meaningless sub-tokens.
We add a curated set of atom-in-SMILES tokens (elements + common valence /
charge / chirality / bracket patterns) and mean-init their embeddings so the
model can learn them quickly during Stage 0a continued pre-training.

Usage:
    python training/tokenizer/extend_vocab.py \
        --base-model Qwen/Qwen3-8B \
        --out-dir    /workspace/training/tokenizer/qwen3_atomInSmiles

The output dir is loadable via:
    AutoTokenizer.from_pretrained("<out-dir>")
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    model.resize_token_embeddings(len(tokenizer))  # done in train_sft.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


# --- Vocabulary ------------------------------------------------------------

# Atoms as they appear in SMILES (bracketed + bare). Covers the majority of
# drug-like chemistry; anything outside falls back to byte-level BPE.
_ATOMS_BARE = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B"]
_ATOMS_AROM = ["c", "n", "o", "s", "p"]

# Common bracketed atoms with valence / charge / chirality.
_ATOMS_BRACKET = [
    "[H]", "[C]", "[N]", "[O]", "[S]",
    "[NH]", "[NH2]", "[NH3+]", "[NH+]", "[N+]", "[N-]",
    "[OH]", "[O-]", "[O+]",
    "[S-]", "[S+]", "[SH]",
    "[C@H]", "[C@@H]", "[C@]", "[C@@]",
    "[nH]", "[nH+]", "[n+]", "[n-]",
    "[c-]", "[c+]",
    "[CH2]", "[CH3]", "[CH]",
    "[P+]", "[P@]", "[P@@]",
    # Metals / ions that appear occasionally in drug-like sets
    "[Na+]", "[K+]", "[Mg+2]", "[Ca+2]", "[Fe+2]", "[Fe+3]", "[Zn+2]",
    # Deuterium / tritium
    "[2H]", "[3H]",
]

# SMILES structural tokens (bonds, rings, branches, stereo). Most tokenizers
# already have these, but adding them explicitly guarantees single-token
# representation.
_STRUCTURAL = [
    "(", ")", "[", "]",
    "=", "#", "/", "\\", "@", "@@",
    ".", ":",  # disconnection, aromatic bond
    # Ring-closure digits 1-9 + %10..%20
    *[str(i) for i in range(1, 10)],
    *[f"%{i}" for i in range(10, 21)],
]


def build_vocabulary() -> list[str]:
    """Return the ordered vocabulary to add. Duplicates against the base
    tokenizer are filtered inside main()."""
    vocab = _ATOMS_BARE + _ATOMS_AROM + _ATOMS_BRACKET + _STRUCTURAL
    # Dedupe preserving order
    seen = set()
    out: list[str] = []
    for t in vocab:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                    help="HF repo id or local path for the base tokenizer")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    start = len(tok)

    candidate = build_vocabulary()
    existing = set(tok.get_vocab().keys())
    to_add = [t for t in candidate if t not in existing]

    print(f"Base tokenizer size: {start}")
    print(f"Candidates: {len(candidate)}  (already present: {len(candidate) - len(to_add)})")
    print(f"Adding     : {len(to_add)}")
    if args.dry_run:
        print("Dry run — first 40 tokens that would be added:")
        print(to_add[:40])
        return

    added = tok.add_tokens(to_add)
    end = len(tok)
    print(f"Added {added} tokens. New size: {end}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(args.out_dir)

    # Record what was added, so train_sft.py can mean-init exactly these ids.
    added_tokens = to_add[:added]
    added_ids = tok.convert_tokens_to_ids(added_tokens)
    with open(args.out_dir / "added_tokens.json", "w") as f:
        json.dump({"tokens": added_tokens, "ids": added_ids, "base_size": start}, f, indent=2)
    print(f"Saved tokenizer + added_tokens.json to {args.out_dir}")


if __name__ == "__main__":
    main()
