# DiffSBDD seed bank — Amendment v2 (Arm C-PC)

Pocket-conditioned ligand seeds for SARS-CoV-2 Mpro (PDB 7L13), generated
once offline by **DiffSBDD** (Schneuing et al., *Nature Computational
Science*, 2024) and committed as a frozen artifact that the prompt
generator samples from at training time.

## Schema (`diffsbdd_7L13.jsonl`)

One JSON record per line:

| Field | Type | Notes |
|---|---|---|
| `smiles` | str | Canonical SMILES (drug-like, dedup'd) |
| `dock_kcal` | float | AutoDock Vina docking score at `n_seeds=1` against `data/receptors/7L13/receptor.pdbqt` |
| `qed` | float \| null | RDKit QED |
| `sa_raw` | float \| null | SA score (1=easy, 10=hard) via `/opt/sascorer` |
| `scaffold` | str | Murcko scaffold SMILES |
| `source` | str | Always `"diffsbdd"` |
| `cd2020_max_tani` | float | Max ECFP4 Tanimoto to any ligand in the CrossDocked-leakage reference set |

## Runbook

Prerequisites:
- `container/chem_rlvr.sif` built (smoke test passed).
- `data/receptors/7L13/prep_receptor.sh` already run, with
  `receptor.pdbqt`, `receptor.config`, and `redock_rmsd.txt < 2.0 Å` on disk.
- DiffSBDD weights downloaded (HuggingFace or upstream release).

```bash
# 1. Install DiffSBDD inference into the container (one-time):
apptainer exec --nv container/chem_rlvr.sif \
    pip install --no-deps diffsbdd  # or vendor weights via huggingface-cli

# 2. Sample 8k molecules conditioned on the prepped 7L13 receptor:
CENTER=$(awk -F'=' '/^center_/ {gsub(/ /,"",$2); printf "%s,", $2}' \
           data/receptors/7L13/receptor.config | sed 's/,$//')
apptainer exec --nv container/chem_rlvr.sif \
    python -m diffsbdd.inference \
        --receptor data/receptors/7L13/receptor.pdb \
        --pocket-center "$CENTER" \
        --n-samples 8000 \
        --out data/seeds/diffsbdd_raw.sdf

# 3. Decode + filter + Vina pre-score (writes diffsbdd_7L13.jsonl):
apptainer exec --nv container/chem_rlvr.sif \
    python data/seeds/build_seed_bank.py \
        --sdf          data/seeds/diffsbdd_raw.sdf \
        --receptor-dir data/receptors/7L13 \
        --out          data/seeds/diffsbdd_7L13.jsonl
```

Expected yield on Mpro pockets: ~55–65 % RDKit-valid after openbabel decode,
then ~50–70 % of those pass drug-likeness, then ~95 % pass the
leakage-Tanimoto gate. Vina pre-score time at `n_seeds=1` is ~10 s/seed on
one CPU core, so 4 000 seeds × 10 s ÷ 256 cores ≈ 3 h on one sapphire
allocation.

## Receptor-prep consistency check

Our `prep_receptor.sh` uses pure-Python hydrogen placement (ADFRsuite was
intentionally dropped). DiffSBDD's reference inference uses Reduce +
ADFRsuite. Before committing the seed bank, redock five random DiffSBDD
samples against (a) our `receptor.pdbqt` and (b) a DiffSBDD-prep-style
receptor; if mean Vina differs by more than 0.5 kcal/mol, regenerate the
seed bank with the same receptor file the verifier uses. **Never train RL
on seeds derived from a different receptor file than the verifier uses.**

## CrossDocked-leakage filter

DiffSBDD was trained on CrossDocked2020, which contains Mpro-similar
proteases. Seeds that paraphrase CrossDocked training ligands would
contaminate the novelty claim. `build_seed_bank.py --leakage-reference`
defaults to `data/reference/known_mpro_inhibitors.csv`, which is the
minimum acceptable reference set. **Before any publication claim about
novelty, expand this reference set with the CrossDocked2020 pocket-
neighbours of 7L13** (sequence-similar 3CLpro / Mpro / NS3 entries) and
re-run with `--leakage-reference <expanded.smi>`.

## Falling back to scaffold-only mode

If DiffSBDD's pocket conditioning misbehaves on 7L13 (sample diversity
collapses, 3D→SMILES yield drops below 30 %, or pose RMSD against 7L13's
co-crystal is consistently high), fall back to using its Murcko scaffolds
only and instantiate substituents from `data/reference/zinc_random.smi`.
This turns the diffusion model into a scaffold generator and dodges the
3D→SMILES failure mode without giving up the pocket-conditioned prior.
