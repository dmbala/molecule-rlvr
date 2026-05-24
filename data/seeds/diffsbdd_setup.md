# DiffSBDD setup runbook — separate conda env (Amendment v2)

DiffSBDD ships no PyPI wheel and requires `pytorch-scatter` compiled against
the same CUDA/PyTorch ABI as the container — and the container uses
PyTorch 2.3.0+cu121 while DiffSBDD's reference env is PyTorch 2.0.1+cu118.
Keeping DiffSBDD in its own conda env is cleaner than rebuilding chem_rlvr
to satisfy the diffusion side. DiffSBDD inference is a one-shot offline
job; the seed bank artifact is what the RL container actually consumes.

## One-time setup

```bash
# 1. Pick the right Anaconda / Miniforge / Mamba on the cluster.
module load Anaconda2/2019.10-fasrc01   # or whatever your site uses

# 2. Clone DiffSBDD upstream (504 stars, last commit 2026-05-12; MIT).
mkdir -p ~/envs && cd ~/envs
git clone https://github.com/arneschneuing/DiffSBDD.git
cd DiffSBDD

# 3. Build the env. The conda solve takes ~10–20 min on first run.
conda env create -f environment.yaml -n diffsbdd
conda activate diffsbdd

# 4. Download a pre-trained checkpoint. The CrossDocked full-atom
#    conditional model is what we use for pocket-conditioned generation
#    in the 7L13 use case (it was trained on CrossDocked2020, which
#    contains Mpro-like proteases — see the leakage filter in
#    build_seed_bank.py).
mkdir -p ckpts
curl -L -o ckpts/crossdocked_fullatom_cond.ckpt \
    'https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt?download=1'
```

## Inference for 7L13

```bash
REPO=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/molecule-rlvr
cd ~/envs/DiffSBDD
conda activate diffsbdd

# Generate ~8000 candidate ligands conditioned on the XF7 pocket.
# --ref_ligand: use the prepped XF7 to define the binding site (atoms
#   within ~8 Å of any ref_ligand atom define the pocket). This matches
#   how Vina sees the pocket in our verifier.
python generate_ligands.py \
    ckpts/crossdocked_fullatom_cond.ckpt \
    --pdbfile     $REPO/data/receptors/7L13/receptor.pdb \
    --ref_ligand  $REPO/data/receptors/7L13/cocrystal.pdb \
    --outfile     $REPO/data/seeds/diffsbdd_raw.sdf \
    --n_samples   8000 \
    --batch_size  100 \
    --sanitize
```

Expect ~6 GPU-hours on a single H100/A100. Submit as a Slurm job if
nodes are queued.

## Decode + filter + Vina pre-score

Back inside the chem_rlvr container — this is where the seed bank gets
turned into a JSONL that the prompt generator can consume.

```bash
cd $REPO
singularity exec --nv \
    --bind "$PWD:/workspace" \
    container/chem_rlvr.sif \
    python data/seeds/build_seed_bank.py \
        --sdf          data/seeds/diffsbdd_raw.sdf \
        --receptor-dir data/receptors/7L13 \
        --out          data/seeds/diffsbdd_7L13.jsonl
```

Expected yield: ~3–5 k drug-like, Vina-pre-scored, CrossDocked-leakage-
filtered seeds.

## Fallback: skip DiffSBDD for the first pilot

If conda + pytorch-scatter + checkpoint download isn't feasible right
now, the pilot can still run. Pass `--diffsbdd-frac 0` to
`generate_prompts.py` and the prompt generator falls back to
`known_inhibitor_fragment` + `zinc_random` seeds at the originally-planned
ratios. The C-PC arm's scientific claim does **not** depend on the
DiffSBDD seed bank — it depends on the proposer-critic structure on top
of any seed source. Document the deviation under
`analysis/preregistration_v2.md` §10.

## Receptor-prep consistency check

DiffSBDD's reference inference uses Reduce + ADFRsuite for the protein
PDB. Our `prep_receptor.sh` uses OpenBabel `CorrectForPH(7.4)` instead.
Before committing the seed bank, redock 5 random DiffSBDD samples against
both protein files and confirm mean Vina differs by < 0.5 kcal/mol. If
the gap is larger, regenerate the seed bank with the chem_rlvr receptor
file. **Never train RL on seeds derived from a different receptor file
than the verifier actually uses.**
