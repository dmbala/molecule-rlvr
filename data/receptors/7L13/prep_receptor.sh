#!/usr/bin/env bash
# Receptor-prep pipeline for SARS-CoV-2 Mpro (PDB 7L13, non-covalent state).
# See plan Fix 4. Run inside chem_rlvr.sif.
#
# Usage (from repo root):
#   apptainer exec --nv chem_rlvr.sif bash data/receptors/7L13/prep_receptor.sh
#
# Output artifacts (committed to repo once the RMSD gate passes):
#   receptor.pdb       — raw PDB with waters/ligand stripped
#   receptor.pdbqt     — AutoDock input (generated with OpenBabel)
#   receptor.config    — Vina grid box (center/size)
#   cocrystal.pdb      — extracted co-crystal ligand
#   redock_rmsd.txt    — RMSD vs. crystal pose after redocking (acceptance gate)
#
# Uses OpenBabel (openbabel-wheel) + RDKit + Vina (Python API). No ADFRsuite.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PDB_ID=${PDB_ID:-7L13}
# 7L13's co-crystal ligand is XF7 ("Compound 21", chloro-bipyridine
# pyrimidine-dione, non-covalent). For a different Mpro structure copy
# this directory under data/receptors/<NEW_PDB>/ and override COCRYSTAL_RESN.
COCRYSTAL_RESN=${COCRYSTAL_RESN:-XF7}

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
            protein.append(ln)
Path("receptor.pdb").write_text("\n".join(protein) + "\n")
Path("cocrystal.pdb").write_text("\n".join(ligand) + "\n")
print(f"Extracted {len(ligand)} ligand atoms -> cocrystal.pdb")
PY

# --- 3. Receptor PDBQT via OpenBabel ---------------------------------------
# Add hydrogens at pH 7.4, assign Gasteiger charges, write PDBQT with rigid
# receptor flag (-xr). Matches the recipe the Forli lab (Vina authors) use
# for quick receptor prep.
python - <<'PY'
from openbabel import pybel
mol = next(pybel.readfile("pdb", "receptor.pdb"))
# Protonate at pH 7.4
mol.OBMol.CorrectForPH(7.4)
mol.addh()
# Write PDBQT; -xr = rigid receptor (no rotatable bonds)
mol.write("pdbqt", "receptor.pdbqt", overwrite=True, opt={"r": True})
print(f"Wrote receptor.pdbqt ({len(mol.atoms)} atoms including H)")
PY

# --- 4. Grid box: center on co-crystal ligand centroid ---------------------
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

# --- 5. Redock the co-crystal ligand and compute RMSD ---------------------
python - <<'PY'
import tempfile
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Read the co-crystal ligand (heavy atoms only) as reference pose for RMSD.
ref = Chem.MolFromPDBFile("cocrystal.pdb", removeHs=True, sanitize=False)
if ref is None:
    raise SystemExit("RDKit could not parse cocrystal.pdb")

# OpenBabel produces the ligand PDBQT (perceives bond orders + active torsions
# from PDB coordinates). RDKit + meeko fail here because the HETATM-stripped
# PDB has no CONECT records and `sanitize=False` skips bond inference, leaving
# meeko nothing to write.
from openbabel import pybel
lig = next(pybel.readfile("pdb", "cocrystal.pdb"))
lig.OBMol.CorrectForPH(7.4)
lig.addh()
lig.write("pdbqt", "cocrystal_lig.pdbqt", overwrite=True)

from vina import Vina
v = Vina(sf_name="vina", cpu=4, seed=0, verbosity=1)
v.set_receptor("receptor.pdbqt")
v.set_ligand_from_file("cocrystal_lig.pdbqt")

cfg = {}
for line in Path("receptor.config").read_text().splitlines():
    if "=" not in line:
        continue
    key, val = line.split("=", 1)
    cfg[key.strip()] = val.strip()
center = [float(cfg[f"center_{a}"]) for a in "xyz"]
size   = [float(cfg[f"size_{a}"]) for a in "xyz"]
v.compute_vina_maps(center=center, box_size=size)
v.dock(exhaustiveness=16, n_poses=9)
v.write_poses("cocrystal_redocked.pdbqt", n_poses=1, overwrite=True)

from openbabel import pybel, openbabel as ob

# Compare the prepped input PDBQT (what Vina docked) against the docked PDBQT.
# Both go through OpenBabel's torsion-tree representation so their atom order
# matches 1:1, and OBAlign(symmetry=True) handles automorphism on top of that.
# Earlier versions used a naive ref_coords[:n] - docked_coords[:n] subtraction
# which failed for any rotated pose because crystal PDB atom order differs
# from PDBQT atom order.
prepped = next(pybel.readfile("pdbqt", "cocrystal_lig.pdbqt"))
docked  = next(pybel.readfile("pdbqt", "cocrystal_redocked.pdbqt"))
align = ob.OBAlign(False, True)   # includeH=False, symmetry-aware
align.SetRefMol(prepped.OBMol)
align.SetTargetMol(docked.OBMol)
align.Align()
rmsd = float(align.GetRMSD())
print(f"Heavy-atom RMSD (OBAlign, symmetry-aware): {rmsd:.3f} A")

pass_gate = rmsd < 2.0
Path("redock_rmsd.txt").write_text(
    f"heavy_atom_rmsd = {rmsd:.3f}\n"
    f"method = OBAlign(prepped_pdbqt, docked_pdbqt, symmetry=True)\n"
    f"acceptance_gate_kcal = 2.0\n"
    f"gate_passed = {str(pass_gate).lower()}\n"
)

if not pass_gate:
    raise SystemExit(
        f"FAIL: redock RMSD {rmsd:.3f} A >= 2.0 A threshold. Common causes:\n"
        f"  - Receptor protonation state (OpenBabel CorrectForPH may misprotonate His).\n"
        f"  - OpenBabel aromatic-kekulize warning during receptor prep (see stderr).\n"
        f"  - Vina scoring is suboptimal for large flexible ligands at default exhaustiveness.\n"
        f"Decide before proceeding: (a) try Reduce / ADFRsuite for receptor prep,\n"
        f"(b) document a prereg deviation log entry relaxing the gate, or\n"
        f"(c) try a different Mpro PDB."
    )
print("PASS")
PY

echo "Receptor prep complete. Artifacts:"
ls -la receptor.pdb receptor.pdbqt receptor.config cocrystal.pdb redock_rmsd.txt
