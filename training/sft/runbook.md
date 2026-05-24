# SFT runbook — Stage 0a / 0b / 0c

Stage 0 SFT is the warm-start pipeline (plan Fix 8.1). It is what arms A,
B, C, and C-PC all initialize from before any RL step. The Slurm launcher
`training/slurm/sft_launch.sh` runs single-node, 4-GPU DeepSpeed via
`train_sft.py`. **No GPU job in this runbook is auto-submitted; each
sbatch line below is an explicit, opt-in step.**

## Prerequisites

Before any SFT step:

- `container/chem_rlvr.sif` rebuilt with transformers >= 4.51 (current
  build is 4.44 — see the `chem_rlvr.def` comment around the openrlhf
  install for the relax-the-pin pattern; the smoke test now exercises a
  Qwen3 tokenizer load to catch this).
- `training/tokenizer/qwen3_atomInSmiles/` exists (committed; produced by
  `training/tokenizer/extend_vocab.py`). Stage 0a reads it via the
  `tokenizer_extended` config field.
- Receptor + prompts + reward sanity check pass (or are documented as
  deviations). SFT itself doesn't dock, but Stage 0b's panel build does.

## Stage 0a — chemistry LM warm-start (~5 M drug-like SMILES)

Plan Fix 8.1.

Data sources:

| Source | Where | Approx. count | Notes |
|---|---|---|---|
| ChEMBL 35 SMI flatfile | `https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35.smi.gz` | ~2.4 M | ~1 GB gzipped |
| ZINC20 in-stock drug-like | `https://files.docking.org/2D/` (multi-tranche) | ~10 M+ | Several GB across tranches |

Download (CPU node, ~1 h on a typical link):

```bash
REPO=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/molecule-rlvr
mkdir -p $REPO/data/sft/raw/zinc20_druglike

# ChEMBL 35
curl -L -o $REPO/data/sft/raw/chembl_35.smi.gz \
    'https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35.smi.gz'
gunzip -k $REPO/data/sft/raw/chembl_35.smi.gz
# (prepare_stage0a.py handles .gz directly via smiles_utils.iter_smi_file —
#  unzip only if you want to inspect manually.)

# ZINC20 — adjust tranche list to taste. BDAA is what we already used for
# the ZINC reference seed file; the larger SFT corpus needs many tranches.
# This is illustrative; pick tranches that match your storage budget.
for t in BD BE BF; do
  for sub in AA AB BA BB; do
    url="https://files.docking.org/2D/${t}/${t}${sub}.smi"
    code=$(curl -sI "$url" -o /dev/null -w "%{http_code}")
    if [ "$code" = "200" ]; then
      curl -sL "$url" > "$REPO/data/sft/raw/zinc20_druglike/${t}${sub}.smi"
    fi
  done
done
ls -la $REPO/data/sft/raw/
```

Build the corpus:

```bash
singularity exec --nv \
    --bind "$REPO:/workspace" \
    --env  CHEM_VERIFIER_CONFIG=/workspace/training/configs/verifier.yaml \
    $REPO/container/chem_rlvr.sif \
    python /workspace/training/sft/prepare_stage0a.py \
        --chembl-smi  /workspace/data/sft/raw/chembl_35.smi.gz \
        --zinc-dir    /workspace/data/sft/raw/zinc20_druglike \
        --out-dir     /workspace/data/sft/stage0a \
        --target-size 5000000
```

Outputs: `data/sft/stage0a/train.jsonl`, `val.jsonl`, `corpus_stats.json`.
Wall time: 2–4 h on one CPU node.

Submit the SFT job (4× H100, ~1 epoch, ~10–15 h):

```bash
sbatch --export=CONFIG=training/configs/sft_stage0a.yaml \
    $REPO/training/slurm/sft_launch.sh
```

Acceptance: validity ratio > 0.8 on a held-out eval set (the same 60 eval
prompts), checkpoint at `data/sft/stage0a` / `checkpoints/sft_stage0a/`.

## Stage 0b — R1 CoT traces about Mpro docking (~5 k traces)

Plan Fix 8.1, residue-aware reasoning prior. Two-step:

1. Build the (context, smiles, vina_score, label) panel. Expensive — it
   actually docks ~5 k molecules. Run on a CPU node.

   ```bash
   singularity exec --nv \
       --bind "$REPO:/workspace" \
       --env  CHEM_VERIFIER_CONFIG=/workspace/training/configs/verifier.yaml \
       $REPO/container/chem_rlvr.sif \
       python /workspace/training/sft/build_stage0b_panel.py \
           --known        /workspace/data/reference/known_mpro_inhibitors.csv \
           --zinc-smi     /workspace/data/reference/zinc_random.smi \
           --chembl       /workspace/data/sft/stage0a/train.jsonl \
           --receptor-dir /workspace/data/receptors/7L13 \
           --out          /workspace/data/sft/stage0b/panel.jsonl
   ```

   Wall time: ~4–8 h on 64 CPU cores at exhaustiveness=16.

2. Generate R1-CoT traces. Needs `DEEPSEEK_API_KEY` (DeepSeek API) or a
   local R1 vLLM endpoint.

   ```bash
   export DEEPSEEK_API_KEY=...
   singularity exec --nv \
       --bind "$REPO:/workspace" \
       $REPO/container/chem_rlvr.sif \
       python /workspace/training/sft/generate_r1_cot.py \
           --panel    /workspace/data/sft/stage0b/panel.jsonl \
           --out      /workspace/data/sft/stage0b/r1_cot.jsonl \
           --endpoint https://api.deepseek.com \
           --model    deepseek-reasoner \
           --n-per-item 2
   ```

Then SFT:

```bash
sbatch --export=CONFIG=training/configs/sft_stage0b.yaml \
    $REPO/training/slurm/sft_launch.sh
```

## Stage 0c — Mpro analogs + BRICS expansion (~5 k molecules)

```bash
singularity exec --nv \
    --bind "$REPO:/workspace" \
    $REPO/container/chem_rlvr.sif \
    python /workspace/training/sft/prepare_stage0c.py \
        --known            /workspace/data/reference/known_mpro_inhibitors.csv \
        --out              /workspace/data/sft/stage0c/train.jsonl \
        --target-size      5000 \
        --analogs-per-seed 200

sbatch --export=CONFIG=training/configs/sft_stage0c.yaml \
    $REPO/training/slurm/sft_launch.sh
```

The Stage-0c checkpoint at `checkpoints/sft_stage0c/` is what all four RL
arms (A, B, C, C-PC) initialize from.

## What to verify after each stage

- Stage 0a: loss curve descending, validity ratio on a 1 k-prompt held-out
  set > 0.8.
- Stage 0b: trace yield > 50 % (i.e., > 5 k panel items → > 2.5 k kept
  after `trace_is_good()` filter in `generate_r1_cot.py`).
- Stage 0c: parse rate on the eval prompts ≥ 0.9; we expect the
  target-specific tune to slightly *raise* validity vs. Stage 0a.

All three stages can run on `kempner_h100` per `sft_launch.sh` (single
node, 4× H100, 24 h limit). Override `--partition` if you queue elsewhere.
