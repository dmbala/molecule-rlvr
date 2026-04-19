"""Virtual-screening baseline (plan Fix 8.6).

This is the *real* baseline RL must beat. Procedure:
  1. Sample K molecules from the Stage-0 SFT checkpoint at temperature T
     against the held-out eval prompt set.
  2. Re-rank them by the full `chemistry_verifier` reward (Fix 2).
  3. Report hit-rate curves as a function of oracle-call budget — this is the
     RL-vs-VS comparison the paper hinges on.

We use vLLM for sampling because 100k samples from a 7B/14B model at
temperature 0.7 would be hours on HF `generate()` but minutes on vLLM.

Usage (inside container):
  apptainer exec --nv container/chem_rlvr.sif \
      python analysis/vs_baseline.py \
          --model    /workspace/checkpoints/sft_stage0c \
          --prompts  /workspace/data/prompts/mpro_prompts.jsonl \
          --config   /workspace/training/configs/verifier.yaml \
          --n-samples-per-prompt 1600 \
          --temperature 0.9 --top-p 0.95 \
          --out      /workspace/analysis/vs_baseline_result.json

Budget note: 60 eval prompts × 1600 samples = 96k oracle calls, closely
matching the RL budget (256k across 500 steps × 512 rollouts, but RL reuses
via replay). Record the number of Vina calls alongside the hit rate so the
comparison is per-oracle-call.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

# --- Wire up the verifier package ------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "verifier"))

from chemistry_verifier import (  # noqa: E402
    RewardRecord, verify_group, extract_smiles,
)
from verifier_config_loader import load_verifier_config  # noqa: E402

log = logging.getLogger("vs_baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_prompts(path: Path, split: str = "eval") -> list[dict]:
    items = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        if d.get("split") == split:
            items.append(d)
    return items


def render_prompt(item: dict) -> str:
    head = f"{item['context']}\n\n{item['instruction']}"
    if item.get("seed_smiles"):
        head = f"Seed SMILES: {item['seed_smiles']}\n\n" + head
    return head


def sample_vllm(model_path: str, prompts: list[str], n: int, temperature: float,
                top_p: float, max_tokens: int) -> list[list[str]]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, trust_remote_code=True,
              gpu_memory_utilization=0.85, dtype="bfloat16")
    params = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                            max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    return [[o.text for o in out.outputs] for out in outputs]


def evaluate_group(responses: list[str], cfg, step: int) -> list[RewardRecord]:
    return verify_group(responses, cfg, step=step)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n-samples-per-prompt", type=int, default=1600)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch-size-prompts", type=int, default=8,
                    help="Number of prompts to sample per vLLM batch; trade memory vs. speed")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_verifier_config(str(args.config))
    items = load_prompts(args.prompts, split=args.split)
    log.info("Loaded %d %s prompts", len(items), args.split)

    t0 = time.time()

    rendered = [render_prompt(it) for it in items]
    all_records: list[dict] = []
    budget_oracle_calls = 0

    for i in range(0, len(items), args.batch_size_prompts):
        sub = items[i:i + args.batch_size_prompts]
        sub_rendered = rendered[i:i + args.batch_size_prompts]
        log.info("Sampling prompts %d..%d", i, i + len(sub))
        samples = sample_vllm(args.model, sub_rendered,
                              n=args.n_samples_per_prompt,
                              temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens)

        for it, resps in zip(sub, samples):
            records = verify_group(resps, cfg, step=9999)
            budget_oracle_calls += sum(1 for r in records if r.dock_kcal is not None)
            for resp, rec in zip(resps, records):
                d = asdict(rec)
                d["prompt_id"] = it["id"]
                d["response"] = resp
                all_records.append(d)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate metrics ----------------------------------------------------
    rewards = [r["reward"] for r in all_records]
    hits_dock7 = sum(1 for r in all_records
                     if r.get("dock_kcal") is not None
                     and r["dock_kcal"] < -7.0
                     and (r.get("qed") or 0.0) > 0.5
                     and (r.get("sa_raw") or 10.0) < 5.0)
    n_parsed = sum(1 for r in all_records if r.get("parsed"))
    n_total = len(all_records)

    # Hit-rate curve by oracle-call rank: sort records by reward desc, step
    # through in chunks of 100 evaluated molecules and report cumulative hits.
    sorted_by_rew = sorted(
        [r for r in all_records if r.get("dock_kcal") is not None],
        key=lambda r: -r["reward"],
    )
    curve = []
    cum_hits = 0
    for i, r in enumerate(sorted_by_rew, 1):
        is_hit = (r["dock_kcal"] < -7.0
                  and (r.get("qed") or 0.0) > 0.5
                  and (r.get("sa_raw") or 10.0) < 5.0)
        if is_hit:
            cum_hits += 1
        if i % 100 == 0 or i == len(sorted_by_rew):
            curve.append({"n_evaluated": i, "cum_hits": cum_hits,
                          "hit_rate": cum_hits / i})

    summary = {
        "n_prompts": len(items),
        "n_samples_per_prompt": args.n_samples_per_prompt,
        "n_total_responses": n_total,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / max(1, n_total),
        "n_oracle_calls": budget_oracle_calls,
        "hit_rate_overall": hits_dock7 / max(1, n_total),
        "hit_rate_curve": curve,
        "wall_clock_sec": time.time() - t0,
    }

    with args.out.open("w") as f:
        json.dump({"summary": summary, "records": all_records}, f, indent=2, default=str)

    log.info("Summary: %s", json.dumps(summary, indent=2))
    log.info("This is the baseline RL must beat at equal oracle-call budget.")


if __name__ == "__main__":
    main()
