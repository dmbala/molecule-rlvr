"""Build the DiffSBDD seed bank for Arm C-PC (Amendment v2).

Pipeline:
  1. Read SDF from DiffSBDD inference output (pocket-conditioned 3D ligands).
  2. openbabel-decode each entry to SMILES; RDKit-canonicalize.
  3. Drug-likeness filter via `training/sft/smiles_utils.drug_like`.
  4. CrossDocked-leakage filter: drop seeds with ECFP4 Tanimoto > threshold
     to any ligand in `--leakage-reference` (defaults to the known-Mpro CSV;
     expand this with a CrossDocked2020-derived list before any novelty claim).
  5. Pre-score the survivors with AutoDock Vina at `n_seeds=1` (the verifier
     still uses `n_seeds=3` at training/eval time).
  6. Emit a JSONL — one record per kept seed — with the schema
     `{smiles, dock_kcal, qed, sa_raw, scaffold, source, cd2020_max_tani}`.

Run in-container:
    apptainer exec --nv container/chem_rlvr.sif \
        python data/seeds/build_seed_bank.py \
            --sdf data/seeds/diffsbdd_raw.sdf \
            --receptor-dir data/receptors/7L13 \
            --out data/seeds/diffsbdd_7L13.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "training" / "sft"))
sys.path.insert(0, str(_REPO / "verifier"))

from smiles_utils import canonical_or_none, canonicalize_and_dedup  # noqa: E402
from reward_components_ext import run_docking_with_config  # noqa: E402

# sascorer is shipped via /opt/sascorer inside the container.
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("build_seed_bank")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# --- SDF decoding -----------------------------------------------------------

def iter_sdf_smiles(sdf_path: Path):
    """Decode 3D ligands in an SDF to SMILES strings.

    Uses openbabel (already in the container) because DiffSBDD outputs SDFs
    with hydrogens placed and bond orders sometimes inferred differently than
    RDKit's SDF reader expects.
    """
    from openbabel import pybel

    for mol in pybel.readfile("sdf", str(sdf_path)):
        smi_field = mol.write("smi").split("\t")[0].strip()
        if smi_field:
            yield smi_field


# --- Leakage filter ---------------------------------------------------------

def load_leakage_reference(path: Path) -> list:
    """Return ECFP4 fingerprints for the SMILES in `path`. Accepts a CSV with
    a `smiles` column or a plain `.smi`/`.txt` file (one SMILES per line)."""
    if not path.exists():
        log.warning("Leakage reference not found at %s; running with empty filter.", path)
        return []

    smiles: list[str] = []
    if path.suffix.lower() == ".csv":
        for r in csv.DictReader(path.open()):
            s = (r.get("smiles") or "").strip()
            if s:
                smiles.append(s)
    else:
        for ln in path.read_text().splitlines():
            s = ln.split()[0].strip() if ln.strip() else ""
            if s and not s.startswith("#") and not s.lower().startswith("smiles"):
                smiles.append(s)

    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    log.info("Loaded %d reference fingerprints from %s", len(fps), path)
    return fps


def max_tanimoto_to_ref(mol: Chem.Mol, ref_fps: list) -> float:
    if not ref_fps:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
    return float(max(sims))


# --- Main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", type=Path, required=True,
                    help="DiffSBDD inference output (3D ligands).")
    ap.add_argument("--receptor-dir", type=Path, required=True,
                    help="Path to data/receptors/7L13/; must contain receptor.pdbqt + receptor.config.")
    ap.add_argument("--leakage-reference", type=Path,
                    default=_REPO / "data" / "reference" / "known_mpro_inhibitors.csv",
                    help="SMILES file (CSV with `smiles` column, or one-per-line). Drop any seed with Tanimoto > threshold to any of these.")
    ap.add_argument("--leakage-threshold", type=float, default=0.7)
    ap.add_argument("--n-seeds-vina", type=int, default=1,
                    help="Vina n_seeds for the one-shot pre-score (verifier uses 3 at RL time).")
    ap.add_argument("--max-seeds", type=int, default=None,
                    help="Optional cap on the number of seeds kept after filtering.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Receptor-prep sanity check
    pdbqt = args.receptor_dir / "receptor.pdbqt"
    config = args.receptor_dir / "receptor.config"
    if not pdbqt.exists() or not config.exists():
        raise SystemExit(
            f"Receptor not prepared. Run data/receptors/<pdb>/prep_receptor.sh first.\n"
            f"Missing: {[p for p in (pdbqt, config) if not p.exists()]}"
        )

    # 1. SDF → SMILES
    raw = list(iter_sdf_smiles(args.sdf))
    log.info("Read %d SMILES from %s", len(raw), args.sdf)

    # 2-3. Canonicalize + drug-likeness filter, then dedup
    canon_kept: list[str] = []
    for s in raw:
        canon, _ = canonical_or_none(s, require_drug_like=True)
        if canon:
            canon_kept.append(canon)
    canon_kept = canonicalize_and_dedup(canon_kept)
    log.info("After canon + drug-like + dedup: %d", len(canon_kept))

    # 4. Leakage filter (precomputed reference fps)
    ref_fps = load_leakage_reference(args.leakage_reference)

    # 5-6. Vina pre-score and emit
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w") as fout:
        for canon in canon_kept:
            if args.max_seeds and written >= args.max_seeds:
                break
            mol = Chem.MolFromSmiles(canon)
            if mol is None:
                continue
            tani = max_tanimoto_to_ref(mol, ref_fps)
            if tani > args.leakage_threshold:
                continue
            try:
                dock_kcal, _ = run_docking_with_config(
                    mol, args.receptor_dir, n_seeds=args.n_seeds_vina,
                )
            except Exception as e:
                log.warning("Vina failed for %s: %s", canon, e)
                continue
            if dock_kcal == float("inf"):
                continue
            try:
                qed_v = float(QED.qed(mol))
                sa_v = float(sascorer.calculateScore(mol))
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                scaf_smi = Chem.MolToSmiles(scaf) if scaf else ""
            except Exception:
                qed_v, sa_v, scaf_smi = None, None, ""

            fout.write(json.dumps({
                "smiles": canon,
                "dock_kcal": float(dock_kcal),
                "qed": qed_v,
                "sa_raw": sa_v,
                "scaffold": scaf_smi,
                "source": "diffsbdd",
                "cd2020_max_tani": tani,
            }) + "\n")
            written += 1
            if written % 100 == 0:
                log.info("Wrote %d / target %s", written, args.max_seeds or "all")

    log.info("Done. Wrote %d seeds to %s", written, args.out)


if __name__ == "__main__":
    main()
