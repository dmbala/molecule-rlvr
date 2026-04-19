"""Stage 0b — generate DeepSeek-R1 CoT traces about Mpro docking (plan Fix 8.1).

Pipeline:
  1. Build a (context, molecule, vina_score, label) panel from ChEMBL + known
     Mpro inhibitors + a random ZINC subset scored against 7L13.
  2. Prompt DeepSeek-R1 to *explain* why each molecule docks well or poorly,
     referencing specific Mpro residues (His41, Cys145, Glu166, Gln189).
  3. Filter traces for (a) valid final SMILES, (b) at least one correct
     residue mention + plausible structural rationale.
  4. Package as SFT conversations in HF ChatML format.

The traces are used by train_sft.py via the Stage 0b config (to be written once
the panel is generated and R1 is reachable). The R1 call abstracts over either
DeepSeek's API or a local vLLM endpoint so this can run in either environment.

Usage:
  apptainer exec --nv container/chem_rlvr.sif \
      python training/sft/generate_r1_cot.py \
          --panel     /workspace/data/sft/stage0b/panel.jsonl \
          --out       /workspace/data/sft/stage0b/r1_cot.jsonl \
          --endpoint  https://api.deepseek.com \
          --model     deepseek-reasoner \
          --api-key-env DEEPSEEK_API_KEY \
          --n-per-item 2

Panel file schema (one JSON per line):
  {"id", "context", "smiles", "vina_score", "label": "strong"|"weak", "source"}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


REQUIRED_RESIDUES = ("His41", "Cys145", "Glu166", "Gln189", "His163", "Met49", "Phe140")
SYSTEM_PROMPT = (
    "You are an expert medicinal chemist reasoning about SARS-CoV-2 Mpro "
    "(main protease) inhibitors. Given a pocket context and a candidate "
    "molecule with its AutoDock Vina score, explain why the molecule does "
    "or does not bind well. Your reasoning must cite specific pocket "
    "residues (e.g. His41, Cys145, Glu166, Gln189) and describe what "
    "interactions are made or missed. Close with a single line containing "
    "only the molecule's canonical SMILES prefixed by `SMILES: `."
)


@dataclass
class PanelItem:
    id: str
    context: str
    smiles: str
    vina_score: float
    label: str
    source: str


@dataclass
class Trace:
    id: str
    prompt: str
    reasoning: str
    smiles: str
    residues_mentioned: list[str] = field(default_factory=list)
    valid_smiles: bool = False
    panel: dict[str, Any] = field(default_factory=dict)


# --- Panel --------------------------------------------------------------

def read_panel(path: Path) -> list[PanelItem]:
    items: list[PanelItem] = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        items.append(PanelItem(**{k: d[k] for k in (
            "id", "context", "smiles", "vina_score", "label", "source"
        )}))
    return items


def render_user_prompt(item: PanelItem) -> str:
    verdict = "binds well" if item.label == "strong" else "binds poorly"
    return (
        f"Pocket context:\n{item.context}\n\n"
        f"Candidate molecule: {item.smiles}\n"
        f"AutoDock Vina score: {item.vina_score:.2f} kcal/mol.\n"
        f"This molecule {verdict}.\n\n"
        "Explain why, citing the specific Mpro residues involved. End with a "
        "single line `SMILES: <canonical SMILES>`."
    )


# --- R1 client ----------------------------------------------------------

def call_r1(endpoint: str, model: str, api_key: str, messages: list[dict]) -> str:
    """OpenAI-compatible chat completion. Works for DeepSeek's API and for a
    local vLLM server started with `--served-model-name`."""
    import requests

    url = endpoint.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 4096,
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# --- Trace parsing ------------------------------------------------------

_SMILES_TAIL = re.compile(r"SMILES:\s*([^\s`]+)\s*$")


def parse_trace(response: str, item: PanelItem) -> Trace:
    smiles = ""
    m = _SMILES_TAIL.search(response.strip())
    if m:
        smiles = m.group(1).strip().strip("`")
    residues = [r for r in REQUIRED_RESIDUES if r in response]
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Trace(
        id=item.id,
        prompt=render_user_prompt(item),
        reasoning=response.strip(),
        smiles=smiles,
        residues_mentioned=residues,
        valid_smiles=mol is not None,
        panel=asdict(item),
    )


def trace_is_good(tr: Trace) -> bool:
    return tr.valid_smiles and len(tr.residues_mentioned) >= 1


# --- ChatML packaging --------------------------------------------------

def to_chatml(tr: Trace) -> dict[str, Any]:
    """Emit the HF conversations format expected by train_sft.py when used
    with a chat-template tokenizer (Qwen3 default)."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": tr.prompt},
            {"role": "assistant", "content": tr.reasoning},
        ],
        "id": tr.id,
        "residues_mentioned": tr.residues_mentioned,
        "final_smiles": tr.smiles,
        "panel_label": tr.panel.get("label"),
        "panel_score": tr.panel.get("vina_score"),
    }


# --- Main ---------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rejects-out", type=Path, default=None)
    ap.add_argument("--endpoint", default=os.environ.get("R1_ENDPOINT", "https://api.deepseek.com"))
    ap.add_argument("--model", default="deepseek-reasoner")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--n-per-item", type=int, default=2,
                    help="Number of independent traces to request per panel item")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--retry", type=int, default=3)
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing {args.api_key_env} in env")

    items = read_panel(args.panel)
    if args.max_items:
        items = items[:args.max_items]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = args.rejects_out or args.out.with_suffix(".rejects.jsonl")

    kept = rejected = 0
    with args.out.open("w") as f_ok, rejects_path.open("w") as f_bad:
        for i, item in enumerate(items):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_user_prompt(item)},
            ]
            for k in range(args.n_per_item):
                resp = None
                for attempt in range(args.retry):
                    try:
                        resp = call_r1(args.endpoint, args.model, api_key, messages)
                        break
                    except Exception as e:
                        wait = 2 ** attempt
                        print(f"[{item.id} try {k}.{attempt}] {e}; sleep {wait}s")
                        time.sleep(wait)
                if resp is None:
                    continue
                tr = parse_trace(resp, item)
                if trace_is_good(tr):
                    f_ok.write(json.dumps(to_chatml(tr)) + "\n")
                    kept += 1
                else:
                    f_bad.write(json.dumps(asdict(tr)) + "\n")
                    rejected += 1

            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(items)}] kept={kept} rejected={rejected}")

    print(f"Done. kept={kept} rejected={rejected} -> {args.out}")


if __name__ == "__main__":
    main()
