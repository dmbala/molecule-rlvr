"""Multi-turn pilot (Phase 2 of notes.md).

Manual multi-turn loop: propose SMILES → verify → feed metrics back → repeat.
Measures *turns-to-threshold* per model for a small set of prompts. This is
the quick-and-dirty comparison that runs BEFORE RL to see whether the
reasoning model has any advantage at all under test-time chain-of-thought.

Runs each model via vLLM (OpenAI-compatible endpoint) or HuggingFace
`pipeline` locally. All generation goes through a unified chat interface so
the only per-arm variable is the model and its reasoning-mode flag.

Usage:
  apptainer exec --nv container/chem_rlvr.sif \
      python training/pilot/multiturn_pilot.py \
          --prompts  data/prompts/mpro_prompts.jsonl \
          --config   training/configs/verifier.yaml \
          --arms     training/pilot/arms.json \
          --out      analysis/pilot_results.jsonl \
          --n-prompts 20 --max-turns 5 \
          --threshold-reward 0.6

`arms.json` example:
  [
    {"name": "A_standard",     "model": "Qwen/Qwen3-8B",
     "thinking": false, "system": null},
    {"name": "B_elicited_cot", "model": "Qwen/Qwen3-8B",
     "thinking": false, "system": "Think step-by-step before answering."},
    {"name": "C_reasoning",    "model": "Qwen/Qwen3-8B",
     "thinking": true,  "system": null}
  ]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "verifier"))

from chemistry_verifier import verify_group  # noqa: E402
from verifier_config_loader import load_verifier_config  # noqa: E402

log = logging.getLogger("pilot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


@dataclass
class ArmSpec:
    name: str
    model: str
    thinking: bool = False
    system: str | None = None


@dataclass
class TurnRecord:
    turn: int
    prompt: str
    response: str
    reward: float | None
    dock_kcal: float | None
    qed: float | None
    sa_raw: float | None
    parsed: bool
    smiles: str | None


@dataclass
class TrajectoryRecord:
    arm: str
    prompt_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    reached_threshold_at_turn: int | None = None
    best_reward: float = 0.0
    best_dock: float | None = None


# --- Model adapter ----------------------------------------------------------

class LocalChatModel:
    """vLLM-backed chat model; lazy-loaded per-arm so we don't hold multiple
    70GB models in GPU memory at once."""

    def __init__(self, model_path: str, thinking: bool) -> None:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path, trust_remote_code=True,
                       gpu_memory_utilization=0.85, dtype="bfloat16")
        self.thinking = thinking
        self.sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
        # Qwen3 chat template accepts `enable_thinking`; on non-Qwen3 templates
        # the kwarg is silently ignored.
        self._supports_thinking_kw = "enable_thinking" in self.tok.apply_chat_template.__doc__ or "enable_thinking" in self.tok.init_kwargs

    def render(self, messages: list[dict]) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tok.apply_chat_template(messages, enable_thinking=self.thinking, **kwargs)
        except TypeError:
            return self.tok.apply_chat_template(messages, **kwargs)

    def chat(self, messages: list[dict]) -> str:
        text = self.render(messages)
        out = self.llm.generate([text], self.sampling)
        return out[0].outputs[0].text


# --- Multi-turn loop --------------------------------------------------------

def format_feedback(rec: TurnRecord) -> str:
    """Turn the verifier's scalar breakdown into a user-visible message."""
    if not rec.parsed:
        return "I could not parse a SMILES from your response. Please respond with a valid SMILES."
    parts = [f"Your previous SMILES: {rec.smiles}"]
    if rec.dock_kcal is not None:
        parts.append(f"Vina docking score: {rec.dock_kcal:.2f} kcal/mol.")
    if rec.qed is not None:
        parts.append(f"QED: {rec.qed:.2f}")
    if rec.sa_raw is not None:
        parts.append(f"SAscore (1=easy, 10=hard): {rec.sa_raw:.2f}")
    parts.append(f"Composite reward: {rec.reward:.3f}.")
    parts.append("Propose an improved SMILES — aim for a more negative Vina score while keeping QED > 0.5 and SAscore < 5.")
    return " ".join(parts)


def run_trajectory(arm: ArmSpec, prompt: dict, model: LocalChatModel,
                   cfg, max_turns: int, threshold_reward: float) -> TrajectoryRecord:
    traj = TrajectoryRecord(arm=arm.name, prompt_id=prompt["id"])

    history: list[dict] = []
    if arm.system:
        history.append({"role": "system", "content": arm.system})

    instruction = prompt["instruction"]
    if prompt.get("seed_smiles"):
        instruction = f"Seed SMILES: {prompt['seed_smiles']}\n\n{instruction}"
    user_msg = f"{prompt['context']}\n\n{instruction}"

    for t in range(1, max_turns + 1):
        history.append({"role": "user", "content": user_msg})
        response = model.chat(history)
        history.append({"role": "assistant", "content": response})

        records = verify_group([response], cfg, step=0)
        rec = records[0]
        turn = TurnRecord(
            turn=t, prompt=user_msg, response=response,
            reward=float(rec.reward),
            dock_kcal=rec.dock_kcal,
            qed=rec.qed,
            sa_raw=rec.sa_raw,
            parsed=rec.parsed,
            smiles=rec.smiles,
        )
        traj.turns.append(turn)
        if turn.reward > traj.best_reward:
            traj.best_reward = turn.reward
            traj.best_dock = turn.dock_kcal
        if turn.reward >= threshold_reward and traj.reached_threshold_at_turn is None:
            traj.reached_threshold_at_turn = t
            break

        user_msg = format_feedback(turn)

    return traj


# --- Main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--arms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--threshold-reward", type=float, default=0.6)
    ap.add_argument("--split", default="eval")
    args = ap.parse_args()

    cfg = load_verifier_config(str(args.config))

    items = [json.loads(ln) for ln in args.prompts.read_text().splitlines() if ln.strip()]
    items = [it for it in items if it.get("split") == args.split][: args.n_prompts]
    log.info("Using %d %s prompts", len(items), args.split)

    arms: list[ArmSpec] = [ArmSpec(**a) for a in json.loads(args.arms.read_text())]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fout:
        for arm in arms:
            log.info("=== Arm %s (%s, thinking=%s) ===", arm.name, arm.model, arm.thinking)
            model = LocalChatModel(arm.model, thinking=arm.thinking)
            for it in items:
                t0 = time.time()
                traj = run_trajectory(arm, it, model, cfg,
                                      max_turns=args.max_turns,
                                      threshold_reward=args.threshold_reward)
                out = asdict(traj)
                out["wall_sec"] = time.time() - t0
                fout.write(json.dumps(out) + "\n")
            # Free GPU memory before loading the next arm. vLLM has no clean
            # shutdown in 0.4.x; process recycling at the shell level is
            # the robust choice if memory becomes an issue.
            del model

    # --- Summary ------------------------------------------------------------
    log.info("Summary by arm:")
    by_arm: dict[str, dict[str, Any]] = {}
    for ln in args.out.read_text().splitlines():
        d = json.loads(ln)
        a = by_arm.setdefault(d["arm"], {"n_prompts": 0, "n_solved": 0,
                                         "turns_to_threshold": [],
                                         "best_reward": [],
                                         "best_dock": []})
        a["n_prompts"] += 1
        if d["reached_threshold_at_turn"] is not None:
            a["n_solved"] += 1
            a["turns_to_threshold"].append(d["reached_threshold_at_turn"])
        a["best_reward"].append(d["best_reward"])
        if d["best_dock"] is not None:
            a["best_dock"].append(d["best_dock"])

    for arm, s in by_arm.items():
        tts = s["turns_to_threshold"]
        log.info(
            "  %-18s solved=%d/%d  median_turns=%s  mean_best_reward=%.3f  mean_best_dock=%.2f",
            arm, s["n_solved"], s["n_prompts"],
            f"{sorted(tts)[len(tts)//2]}" if tts else "NA",
            sum(s["best_reward"]) / len(s["best_reward"]),
            (sum(s["best_dock"]) / len(s["best_dock"])) if s["best_dock"] else float("nan"),
        )


if __name__ == "__main__":
    main()
