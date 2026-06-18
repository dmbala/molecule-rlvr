"""Evaluate the Amendment v2 pilot (C-base vs C-PC) against the four
acceptance gates defined in `analysis/preregistration_v2.md` §6:

  1. Hit-rate ratio   `hit_rate(C-PC)/hit_rate(C-base) ≥ 1.15`, 95% CI LB > 1.0
  2. Validity         C-PC parse rate ≥ 0.75
  3. Diversity        Murcko count / batch ≥ 0.6× C-base AND top-1 ≤ 0.15
  4. Walltime         C-PC step ≤ 1.25× C-base

Reads the JSONL emitted by `training/pilot/multiturn_pilot.py` and reports
per-gate pass/fail. Re-uses the `is_hit` predicate from `vs_baseline.py`
(plus thresholds) so hit definitions stay consistent across the project.

Usage (in-container or with rdkit + scipy + numpy on PATH):
    python analysis/pilot_pc_eval.py \
        --pilot-jsonl analysis/pilot_pc_results.jsonl \
        --base-arm    C_base_8B_pilot \
        --pc-arm      C_pc_8B_pilot \
        --out         analysis/pilot_pc_report.json
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "analysis"))
sys.path.insert(0, str(_REPO / "verifier"))

from vs_baseline import HIT_DOCK_THRESHOLD, HIT_QED_MIN, HIT_SA_MAX, is_hit  # noqa: E402

log = logging.getLogger("pilot_pc_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_pilot(path: Path) -> dict[str, list[dict]]:
    """Read the pilot JSONL and bucket trajectories by `arm`."""
    by_arm: dict[str, list[dict]] = collections.defaultdict(list)
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        by_arm[rec["arm"]].append(rec)
    return dict(by_arm)


# ----------------------------------------------------------------------------
# Gate 1 — hit rate + bootstrap CI on the ratio
# ----------------------------------------------------------------------------

def trajectory_is_hit(traj: dict) -> bool:
    """A trajectory hits if any of its turns satisfies the v1 §4 hit criteria."""
    for turn in traj.get("turns", []):
        if turn.get("parsed") and is_hit(turn):
            return True
    return False


def best_dock_kcal(traj: dict) -> float | None:
    docks = [t.get("dock_kcal") for t in traj.get("turns", [])
             if t.get("dock_kcal") is not None]
    return min(docks) if docks else None


def hit_rate(trajs: list[dict]) -> float:
    if not trajs:
        return 0.0
    return sum(1 for t in trajs if trajectory_is_hit(t)) / len(trajs)


def parse_rate(trajs: list[dict]) -> float:
    """Fraction of trajectories with at least one parsed turn."""
    if not trajs:
        return 0.0
    return sum(1 for t in trajs if any(turn.get("parsed") for turn in t.get("turns", []))) / len(trajs)


def bootstrap_ratio_ci(base: list[dict], pc: list[dict], *,
                         n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Return (point_ratio, ci_low, ci_high) for hit_rate(pc) / hit_rate(base).
    Resamples trajectories with replacement within each arm."""
    rng = np.random.default_rng(seed)
    base_arr = np.array([trajectory_is_hit(t) for t in base], dtype=int)
    pc_arr = np.array([trajectory_is_hit(t) for t in pc], dtype=int)
    if base_arr.size == 0 or pc_arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    point = (pc_arr.mean() / base_arr.mean()) if base_arr.mean() > 0 else float("inf")
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        b = base_arr[rng.integers(0, base_arr.size, base_arr.size)]
        p = pc_arr[rng.integers(0, pc_arr.size, pc_arr.size)]
        bmean = b.mean()
        ratios[i] = (p.mean() / bmean) if bmean > 0 else float("inf")
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ----------------------------------------------------------------------------
# Gate 3 — scaffold diversity
# ----------------------------------------------------------------------------

def scaffold_diagnostics(trajs: list[dict]) -> dict[str, Any]:
    """Count unique Murcko scaffolds across all turns and report the top-1
    scaffold frequency. Uses RDKit only on parsed turns."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    scaffolds: list[str] = []
    for t in trajs:
        for turn in t.get("turns", []):
            smi = turn.get("smiles")
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                s = MurckoScaffold.GetScaffoldForMol(mol)
                scaf_smi = Chem.MolToSmiles(s) if s else ""
            except Exception:
                scaf_smi = ""
            if scaf_smi:
                scaffolds.append(scaf_smi)
    counts = collections.Counter(scaffolds)
    total = sum(counts.values())
    return {
        "n_scaffold_observations": total,
        "n_unique_scaffolds": len(counts),
        "top1_freq": (max(counts.values()) / total) if total else 0.0,
        "top1_scaffold": counts.most_common(1)[0][0] if counts else None,
    }


# ----------------------------------------------------------------------------
# Gate 4 — walltime
# ----------------------------------------------------------------------------

def mean_walltime(trajs: list[dict]) -> float:
    vals = [t.get("wall_sec") for t in trajs if isinstance(t.get("wall_sec"), (int, float))]
    return float(np.mean(vals)) if vals else float("nan")


# ----------------------------------------------------------------------------
# Mann-Whitney U on best-dock per trajectory
# ----------------------------------------------------------------------------

def mw_best_dock(base: list[dict], pc: list[dict]) -> dict[str, float]:
    from scipy.stats import mannwhitneyu

    bd = [best_dock_kcal(t) for t in base]
    pd = [best_dock_kcal(t) for t in pc]
    bd = [v for v in bd if v is not None]
    pd = [v for v in pd if v is not None]
    if len(bd) < 5 or len(pd) < 5:
        return {"u_stat": float("nan"), "p_value": float("nan"),
                "median_base": float("nan"), "median_pc": float("nan")}
    u, p = mannwhitneyu(pd, bd, alternative="less")  # PC docks more negative
    return {
        "u_stat": float(u),
        "p_value": float(p),
        "median_base": float(np.median(bd)),
        "median_pc": float(np.median(pd)),
    }


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

GATE_THRESHOLDS = {
    "hit_rate_ratio_min": 1.15,
    "hit_rate_ratio_ci_lb_min": 1.0,
    "pc_parse_rate_min": 0.75,
    "diversity_unique_ratio_min": 0.6,
    "diversity_top1_max": 0.15,
    "walltime_ratio_max": 1.25,
}


def evaluate(base: list[dict], pc: list[dict]) -> dict[str, Any]:
    point, lo, hi = bootstrap_ratio_ci(base, pc)
    base_div = scaffold_diagnostics(base)
    pc_div = scaffold_diagnostics(pc)
    base_wall = mean_walltime(base)
    pc_wall = mean_walltime(pc)
    mw = mw_best_dock(base, pc)

    div_ratio = (pc_div["n_unique_scaffolds"] / max(1, base_div["n_unique_scaffolds"]))
    walltime_ratio = pc_wall / base_wall if base_wall > 0 else float("inf")

    gates = {
        "hit_rate_ratio_passes": (point >= GATE_THRESHOLDS["hit_rate_ratio_min"]
                                  and lo > GATE_THRESHOLDS["hit_rate_ratio_ci_lb_min"]),
        "validity_passes": parse_rate(pc) >= GATE_THRESHOLDS["pc_parse_rate_min"],
        "diversity_passes": (div_ratio >= GATE_THRESHOLDS["diversity_unique_ratio_min"]
                              and pc_div["top1_freq"] <= GATE_THRESHOLDS["diversity_top1_max"]),
        "walltime_passes": walltime_ratio <= GATE_THRESHOLDS["walltime_ratio_max"],
    }
    gates["all_pass"] = all(gates.values())

    return {
        "thresholds": GATE_THRESHOLDS,
        "hit_criteria": {
            "dock_kcal_max": HIT_DOCK_THRESHOLD,
            "qed_min": HIT_QED_MIN,
            "sa_max": HIT_SA_MAX,
        },
        "n_trajectories": {"base": len(base), "pc": len(pc)},
        "hit_rate": {"base": hit_rate(base), "pc": hit_rate(pc),
                     "ratio_point": point, "ratio_ci_low": lo, "ratio_ci_high": hi},
        "validity": {"base_parse_rate": parse_rate(base),
                      "pc_parse_rate": parse_rate(pc)},
        "diversity": {
            "base": base_div,
            "pc": pc_div,
            "pc_to_base_unique_ratio": div_ratio,
        },
        "walltime": {"base_mean_sec": base_wall, "pc_mean_sec": pc_wall,
                      "ratio": walltime_ratio},
        "best_dock_mann_whitney": mw,
        "gates": gates,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-jsonl", type=Path, required=True)
    ap.add_argument("--base-arm", default="C_base_8B_pilot")
    ap.add_argument("--pc-arm", default="C_pc_8B_pilot")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_arm = load_pilot(args.pilot_jsonl)
    if args.base_arm not in by_arm:
        raise SystemExit(f"Base arm {args.base_arm!r} not found in {args.pilot_jsonl}; arms: {list(by_arm)}")
    if args.pc_arm not in by_arm:
        raise SystemExit(f"PC arm {args.pc_arm!r} not found in {args.pilot_jsonl}; arms: {list(by_arm)}")

    report = evaluate(by_arm[args.base_arm], by_arm[args.pc_arm])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    log.info("Hit rate: base=%.3f  pc=%.3f  ratio=%.2f [%.2f, %.2f]",
             report["hit_rate"]["base"], report["hit_rate"]["pc"],
             report["hit_rate"]["ratio_point"],
             report["hit_rate"]["ratio_ci_low"],
             report["hit_rate"]["ratio_ci_high"])
    log.info("Gates:")
    for k, v in report["gates"].items():
        log.info("  %-30s %s", k, "PASS" if v else "FAIL")
    if not report["gates"]["all_pass"]:
        raise SystemExit("One or more pilot gates failed. Do not escalate to 14B/32B.")


if __name__ == "__main__":
    main()
