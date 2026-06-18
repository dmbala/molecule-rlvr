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

echo "== Ligand prep (meeko MoleculePreparation + PDBQTWriterLegacy) =="
# Catches the meeko-0.6.0 packaging bug where data/params/ad4_types.json is
# missing. If MoleculePreparation() can construct AND prep.prepare() runs on a
# real molecule, the docking path is wired correctly.
python - <<'PY'
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem

prep = MoleculePreparation()
mol = Chem.MolFromSmiles("O=C(NC1CCOCC1)c1ccc(F)cc1")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
prep.prepare(mol)
out = PDBQTWriterLegacy.write_string(prep.setup)
pdbqt = out[0] if isinstance(out, tuple) else out
assert "ATOM" in pdbqt or "HETATM" in pdbqt, "PDBQT output missing atom records"
print(f"  meeko prep + PDBQTWriterLegacy: OK ({len(pdbqt)} chars)")
PY

echo "== Receptor-prep path (OpenBabel Python + RDKit) =="
python - <<'PY'
from openbabel import pybel
print(f"  openbabel formats available: {len(pybel.informats)} in / {len(pybel.outformats)} out")
assert "pdb" in pybel.informats and "pdbqt" in pybel.outformats, \
    "OpenBabel build is missing PDB/PDBQT support"
print("  pdb/pdbqt formats: OK")
PY

echo "== HF transformers + Qwen3 tokenizer compat =="
# Catches the transformers==4.44 / Qwen3-tokenizer.json mismatch that blocked
# extend_vocab.py and train_sft.py on the previous build. We only need the
# tokenizer to *parse* here, not the model weights, so this is fast.
python - <<'PY'
import transformers, tokenizers
print(f"  transformers {transformers.__version__}, tokenizers {tokenizers.__version__}")
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 51), \
    "transformers must be >= 4.51 to parse Qwen3's tokenizer.json"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
print(f"  Qwen3 tokenizer loaded: {type(tok).__name__} vocab_size={tok.vocab_size}")
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
