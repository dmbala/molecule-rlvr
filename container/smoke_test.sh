#!/usr/bin/env bash
# Smoke test for chem_rlvr.sif. Gate Phase 2 of the plan on this passing.
# Usage: apptainer exec chem_rlvr.sif bash container/smoke_test.sh
set -euo pipefail

echo "== Python version =="
python --version

echo "== Core chemistry imports =="
python - <<'PY'
import importlib, sys
mods = ["rdkit", "meeko", "vina", "torch", "flash_attn", "numpy", "scipy", "pandas", "selfies"]
for m in mods:
    mod = importlib.import_module(m)
    v = getattr(mod, "__version__", "?")
    print(f"  {m:15s} {v}")

from openbabel import pybel  # noqa: F401
print("  openbabel.pybel  ok")

import sascorer  # noqa: F401
print("  sascorer         ok (from /opt/sascorer)")
PY

echo "== Vina CLI =="
vina --version

echo "== Receptor-prep path (OpenBabel Python + RDKit) =="
python - <<'PY'
from openbabel import pybel
print(f"  openbabel formats available: {len(pybel.informats)} in / {len(pybel.outformats)} out")
assert "pdb" in pybel.informats and "pdbqt" in pybel.outformats, \
    "OpenBabel build is missing PDB/PDBQT support"
print("  pdb/pdbqt formats: OK")
PY

echo "== CUDA visibility =="
python - <<'PY'
import torch
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"  device_count = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  device 0 = {torch.cuda.get_device_name(0)}")
PY

echo "== PASS =="
