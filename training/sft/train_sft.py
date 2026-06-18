"""Generic SFT trainer for Stage 0a (chemistry LM) and Stage 0c (Mpro analogs).

Stage 0b uses a different loader (conversations, not raw SMILES), so it has
its own trainer entry point.

Reads a YAML config with the schema defined in training/configs/sft_stage0a.yaml.
Uses HF Trainer with DeepSpeed for multi-GPU; relies on the extended tokenizer
produced by training/tokenizer/extend_vocab.py.

Usage (inside container, single node):
    torchrun --nproc_per_node=4 training/sft/train_sft.py \
        --config training/configs/sft_stage0a.yaml

Multi-node launch is via training/slurm/sft_launch.sh (to be written when the
Stage-0a data corpus is ready on disk).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class SFTConfig:
    pretrain: str
    save_path: Path
    dataset: Path
    input_key: str
    max_len: int
    train_batch_size: int
    micro_train_batch_size: int
    learning_rate: float
    max_epochs: int
    num_warmup_steps: int
    weight_decay: float
    tokenizer_extended: Path | None
    resize_token_embeddings: bool
    embed_init: str
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str
    logging_steps: int
    save_steps: int
    seed: int = 42
    gradient_checkpointing: bool = True
    bf16: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        d: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls(
            pretrain=d["pretrain"],
            save_path=Path(d["save_path"]),
            dataset=Path(d["dataset"]),
            input_key=d.get("input_key", "text"),
            max_len=int(d.get("max_len", 512)),
            train_batch_size=int(d["train_batch_size"]),
            micro_train_batch_size=int(d["micro_train_batch_size"]),
            learning_rate=float(d["learning_rate"]),
            max_epochs=int(d.get("max_epochs", 1)),
            num_warmup_steps=int(d.get("num_warmup_steps", 200)),
            weight_decay=float(d.get("weight_decay", 0.01)),
            tokenizer_extended=Path(d["tokenizer_extended"]) if d.get("tokenizer_extended") else None,
            resize_token_embeddings=bool(d.get("resize_token_embeddings", False)),
            embed_init=str(d.get("embed_init", "mean")),
            use_wandb=bool(d.get("use_wandb", False)),
            wandb_project=d.get("wandb_project", "molecule-rlvr"),
            wandb_run_name=d.get("wandb_run_name", "sft"),
            logging_steps=int(d.get("logging_steps", 10)),
            save_steps=int(d.get("save_steps", 1000)),
            seed=int(d.get("seed", 42)),
            gradient_checkpointing=bool(d.get("gradient_checkpointing", True)),
            bf16=bool(d.get("bf16", True)),
        )


def load_and_extend_tokenizer(cfg: SFTConfig):
    """Load the extended tokenizer if provided, else fall back to the base.
    Returns (tokenizer, added_ids) — added_ids is [] when no extension."""
    if cfg.tokenizer_extended and cfg.tokenizer_extended.exists():
        tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer_extended), trust_remote_code=True)
        added_meta_path = cfg.tokenizer_extended / "added_tokens.json"
        if added_meta_path.exists():
            meta = json.loads(added_meta_path.read_text())
            return tok, meta.get("ids", [])
        return tok, []
    tok = AutoTokenizer.from_pretrained(cfg.pretrain, trust_remote_code=True)
    return tok, []


def mean_init_new_embeddings(model, added_ids: list[int]) -> None:
    """Initialize new token embeddings as the mean of the existing ones.
    `resize_token_embeddings` defaults to random init, which wastes compute."""
    if not added_ids:
        return
    with torch.no_grad():
        in_emb = model.get_input_embeddings().weight
        base_end = min(added_ids)  # new ids start at base_end
        base_mean = in_emb[:base_end].mean(dim=0)
        in_emb[added_ids] = base_mean.to(in_emb.dtype)

        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight is not in_emb:
            base_out_mean = out_emb.weight[:base_end].mean(dim=0)
            out_emb.weight[added_ids] = base_out_mean.to(out_emb.weight.dtype)


def build_dataset(cfg: SFTConfig, tokenizer):
    """Stream-load the dataset; tokenize on the fly.

    Two formats supported:
      * Raw text (Stage 0a, 0c): cfg.input_key = "text", each row has a SMILES
        or short string to learn.
      * Chat (Stage 0b): cfg.input_key = "messages", each row has a list of
        {"role", "content"} dicts. Rendered with the tokenizer's chat template.
    """
    suffix = cfg.dataset.suffix.lower()
    if suffix == ".jsonl":
        ds = load_dataset("json", data_files=str(cfg.dataset), split="train", streaming=False)
    elif suffix in {".smi", ".txt"}:
        def _gen():
            for ln in cfg.dataset.read_text().splitlines():
                s = ln.split()[0].strip() if ln.strip() else ""
                if s and not s.startswith("#"):
                    yield {"text": s}
        from datasets import Dataset
        ds = Dataset.from_generator(_gen)
    else:
        raise ValueError(f"Unsupported dataset extension: {suffix}")

    is_chat = cfg.input_key == "messages"

    def _tokenize_text(examples):
        return tokenizer(examples[cfg.input_key],
                         max_length=cfg.max_len, truncation=True, padding=False)

    def _tokenize_chat(examples):
        rendered = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples[cfg.input_key]
        ]
        return tokenizer(rendered, max_length=cfg.max_len, truncation=True, padding=False)

    tokenize_fn = _tokenize_chat if is_chat else _tokenize_text

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names,
                desc=f"Tokenizing {cfg.dataset.name}")
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = ap.parse_args()

    cfg = SFTConfig.from_yaml(args.config)
    set_seed(cfg.seed)

    # Tokenizer (extended if provided) ---------------------------------------
    tokenizer, added_ids = load_and_extend_tokenizer(cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg.pretrain,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )
    if cfg.resize_token_embeddings and len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        if cfg.embed_init == "mean":
            mean_init_new_embeddings(model, added_ids)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Dataset ----------------------------------------------------------------
    train_ds = build_dataset(cfg, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer ----------------------------------------------------------------
    world_size = max(1, int(os.environ.get("WORLD_SIZE", 1)))
    grad_accum = max(1, cfg.train_batch_size // (cfg.micro_train_batch_size * world_size))

    args_hf = TrainingArguments(
        output_dir=str(cfg.save_path),
        overwrite_output_dir=True,
        num_train_epochs=cfg.max_epochs,
        per_device_train_batch_size=cfg.micro_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.num_warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=["wandb"] if cfg.use_wandb else [],
        run_name=cfg.wandb_run_name,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(cfg.save_path))
    tokenizer.save_pretrained(str(cfg.save_path))

    # Small report -----------------------------------------------------------
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Saved {n_params:.2f}B-param model + tokenizer to {cfg.save_path}")


if __name__ == "__main__":
    main()
