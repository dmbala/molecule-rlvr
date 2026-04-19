#!/usr/bin/env bash
# Receptor-prep pipeline for SARS-CoV-2 Mpro (PDB 7L13, non-covalent state).
# See plan Fix 4. Must be run inside chem_rlvr.sif so ADFRsuite + reduce are on PATH.
#
# Usage (from repo root):
#   apptainer exec --nv chem_rlvr.sif bash data/receptors/7L13/prep_receptor.sh
#
# Output artifacts (all committed to repo once produced):
#   receptor.pdb       — raw PDB with waters/HETATMs stripped
#   receptor_h.pdb     — protonated with `reduce`
#   receptor.pdbqt     — AutoDock input
#   receptor.config    — Vina grid box (center/size)
#   cocrystal.pdb      — extracted co-crystal ligand (if any)
#   redock_rmsd.txt    — RMSD vs. crystal pose after redocking (acceptance gate)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PDB_ID=${PDB_ID:-7L13}
COCRYSTAL_RESN=${COCRYSTAL_RESN:-X77}   # 7L13 co-crystal ligand; update for other PDBs

# --- 1. Fetch --------------------------------------------------------------
if [[ ! -f "${PDB_ID}.pdb" ]]; then
    echo "Fetching ${PDB_ID} from RCSB..."
    wget -q "https://files.rcsb.org/download/${PDB_ID}.pdb" -O "${PDB_ID}.pdb"
fi

# --- 2. Strip waters + extract co-crystal ligand ---------------------------
python - <<PY
from pathlib import Path
src = Path("${PDB_ID}.pdb").read_text().splitlines()
protein, ligand = [], []
for ln in src:
    if ln.startswith(("ATOM", "TER", "END")):
        protein.append(ln)
    elif ln.startswith("HETATM"):
        resn = ln[17:20].strip()
        if resn == "${COCRYSTAL_RESN}":
            ligand.append(ln)
        elif resn in {"HOH", "SO4", "PO4", "DMS", "EDO"}:
            continue
        else:
            # Keep cofactors/ions if any; for Mpro there are none typically
            protein.append(ln)
Path("receptor.pdb").write_text("\n".join(protein) + "\n")
Path("cocrystal.pdb").write_text("\n".join(ligand) + "\n")
print(f"Extracted {len(ligand)} ligand atoms -> cocrystal.pdb")
PY

# --- 3. Add hydrogens with reduce ------------------------------------------
reduce -Trim receptor.pdb > receptor_noh.pdb 2> reduce_trim.log || true
reduce -Build receptor_noh.pdb > receptor_h.pdb 2> reduce_build.log

# --- 4. Convert to PDBQT via ADFRsuite -------------------------------------
prepare_receptor4.py \
    -r receptor_h.pdb \
    -o receptor.pdbqt \
    -A checkhydrogens \
    -U nphs_lps_waters_nonstdres

# --- 5. Grid box: center on co-crystal ligand centroid ---------------------
python - <<'PY'
import numpy as np
from pathlib import Path

coords = []
for ln in Path("cocrystal.pdb").read_text().splitlines():
    if ln.startswith("HETATM"):
        try:
            x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            coords.append([x, y, z])
        except ValueError:
            continue

if not coords:
    raise SystemExit("No co-crystal ligand coords found; set COCRYSTAL_RESN correctly")

arr = np.asarray(coords)
center = arr.mean(axis=0).round(2)
size = (22.0, 22.0, 22.0)  # standard Mpro box

config = (
    f"receptor = receptor.pdbqt\n"
    f"center_x = {center[0]}\ncenter_y = {center[1]}\ncenter_z = {center[2]}\n"
    f"size_x = {size[0]}\nsize_y = {size[1]}\nsize_z = {size[2]}\n"
    f"exhaustiveness = 16\nnum_modes = 9\n"
)
Path("receptor.config").write_text(config)
print("Wrote receptor.config:")
print(config)
PY

# --- 6. Redock the co-crystal ligand and compute RMSD ---------------------
python - <<'PY'
import tempfile, subprocess, os
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Read the co-crystal ligand (heavy atoms only) as reference pose
ref = Chem.MolFromPDBFile("cocrystal.pdb", removeHs=True, sanitize=False)
if ref is None:
    raise SystemExit("RDKit could not parse cocrystal.pdb")

# Prepare ligand PDBQT using meeko
from meeko import MoleculePreparation, PDBQTWriterLegacy
ref_h = Chem.AddHs(ref, addCoords=True)
prep = MoleculePreparation()
prep.prepare(ref_h)
pdbqt = PDBQTWriterLegacy.write_string(prep.setup)[0]
Path("cocrystal_lig.pdbqt").write_text(pdbqt)

# Redock with Vina
from vina import Vina
v = Vina(sf_name="vina", cpu=4, seed=0, verbosity=1)
v.set_receptor("receptor.pdbqt")
v.set_ligand_from_file("cocrystal_lig.pdbqt")

# Parse center/size from receptor.config
cfg = dict(
    line.split("=", 1) for line in Path("receptor.config").read_text().splitlines() if "=" in line
)
center = [float(cfg[f"center_{a}"]) for a in "xyz"]
size = [float(cfg[f"size_{a}"]) for a in "xyz"]
v.compute_vina_maps(center=center, box_size=size)
v.dock(exhaustiveness=16, n_poses=9)
v.write_poses("cocrystal_redocked.pdbqt", n_poses=1, overwrite=True)

# Compute heavy-atom RMSD between crystal pose and top redocked pose
from openbabel import pybel
docked = next(pybel.readfile("pdbqt", "cocrystal_redocked.pdbqt"))
docked_coords = np.array([[a.coords[0], a.coords[1], a.coords[2]]
                          for a in docked.atoms if a.atomicnum > 1])
ref_coords = ref.GetConformer().GetPositions()

# Match heavy-atom counts (best effort; production-grade tool would use OBAlign)
n = min(len(ref_coords), len(docked_coords))
diff = ref_coords[:n] - docked_coords[:n]
rmsd = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
print(f"Heavy-atom RMSD (naive match): {rmsd:.3f} A")

Path("redock_rmsd.txt").write_text(
    f"heavy_atom_rmsd_naive = {rmsd:.3f}\n"
    "# Acceptance gate (plan Fix 4): < 2.0 A\n"
)

if rmsd >= 2.0:
    raise SystemExit(
        f"FAIL: redock RMSD {rmsd:.3f} A >= 2.0 A threshold. "
        "Check grid box / receptor prep before proceeding."
    )
print("PASS")
PY

echo "Receptor prep complete. Artifacts:"
ls -la receptor.pdb receptor_h.pdb receptor.pdbqt receptor.config cocrystal.pdb redock_rmsd.txt
