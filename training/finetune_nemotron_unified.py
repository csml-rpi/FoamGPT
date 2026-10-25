#!/usr/bin/env python3
"""
finetune_nemotron_unified.py
============================
Fine-tune nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 with a SINGLE LoRA
adapter trained on ALL substeps together, then merge the adapter back
into the base model.

Usage:
    python finetune_nemotron_unified.py
    python finetune_nemotron_unified.py --train data/train.jsonl
    python finetune_nemotron_unified.py --no-merge       # train only
    python finetune_nemotron_unified.py --only-merge     # merge existing adapter

Differences from finetune_nemotron.py:
    - All 9 substeps concatenated into one dataset (shuffled by seed)
    - One LoRA adapter trained over the combined dataset
    - Adapter merged into the base model after training, producing a single
      merged BF16 checkpoint (no --enable-lora needed at serve time)
    - Per-substep length filter still applied with one global max_length so
      SFTTrainer never sees a truncated assistant response.

Output structure:
    nemotron-3-nano-30b-unified/
    ├── adapter/         # LoRA-only weights (kept even after merge)
    ├── merged/          # base + LoRA, ready for vLLM serving
    ├── metrics.json     # full trainer log_history
    └── serve.sh         # vLLM launch script for the merged model
"""

import argparse
import gc
import glob
import json
import os
import shutil
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get(
    "MODEL_NAME", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
)
SEED = int(os.environ.get("SEED", 3407))
NUM_PROC = int(os.environ.get("NUM_PROC", min(8, os.cpu_count() or 1)))

SUBSTEP_ORDER: List[str] = [
    "parse_case_info",
    "build_advice",
    "decompose_subtasks",
    "generate_file",
    "allrun_commands",
    "allrun_script",
    "error_analysis",
    "rewrite_plan",
    "rewrite_files",
]

# ──────────────────────────────────────────────────────────────────────────────
# Unified training configuration
# ──────────────────────────────────────────────────────────────────────────────
# One LoRA learns every substep, so we pick a single global max_length big
# enough to keep most rows. Anything longer is filtered out (we never want
# SFTTrainer to chop assistant tokens — see filter_by_length).
#
# At max_length=16384 with eager attention the scores tensor is ~48 GB per
# GPU, which is the practical ceiling on 4× 80 GB. ~16 k captures:
#   - 100% of plan / generate / allrun substeps
#   - ~95% of rewrite_plan / rewrite_files
#   - ~66% of error_analysis (which has a 55 k tail)
#
# Effective batch = batch × accum. We keep accum high so the gradient stays
# smooth across heterogeneous samples (sequential samples can be different
# substeps).
#
# Epochs: 3 is a compromise. Per-substep training used 6–8 epochs for the
# small plan substeps (~111 rows) and 2 epochs for long-context substeps.
# With substeps mixed each row gets 3 forward/backward passes — a reasonable
# middle ground; bump NUM_EPOCHS=4-5 if plan substeps look under-fit.
MAX_LENGTH       = int(os.environ.get("MAX_LENGTH", 16384))
PER_DEVICE_BATCH = int(os.environ.get("PER_DEVICE_BATCH", 1))
GRAD_ACCUM       = int(os.environ.get("GRAD_ACCUM", 16))
NUM_EPOCHS       = float(os.environ.get("NUM_EPOCHS", 3))
WARMUP_RATIO     = float(os.environ.get("WARMUP_RATIO", 0.05))
MIN_LR_RATE      = float(os.environ.get("MIN_LR_RATE", 0.1))
LEARNING_RATE    = float(os.environ.get("LEARNING_RATE", 2e-4))
WEIGHT_DECAY     = float(os.environ.get("WEIGHT_DECAY", 0.01))

# ──────────────────────────────────────────────────────────────────────────────
# LoRA configuration — same target modules as the per-substep variant.
# ──────────────────────────────────────────────────────────────────────────────
# Rank/alpha defaults are kept at 16/32 for memory parity with the per-substep
# script. Since one LoRA now has to cover all 9 task types, you can usually
# afford a higher rank: try LORA_R=32 LORA_ALPHA=64 if VRAM allows.
LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))

PEFT_CONFIG = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        # GQA Attention (6 layers)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MoE Expert MLP projections (23 MoE layers)
        "gate_proj", "up_proj", "down_proj",
        # Mamba-2 projections (23 Mamba layers)
        "in_proj", "out_proj",
    ],
    # MoE router/gate is intentionally NOT targeted — fine-tuning it
    # destabilises expert selection (Unsloth + NVIDIA guidance).
    task_type="CAUSAL_LM",
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_substep(rows: List[Dict], substep: str) -> List[Dict]:
    return [r for r in rows if r.get("substep") == substep]


def filter_by_length(
    rows: List[Dict],
    tokenizer,
    max_length: int,
    split_name: str,
    substep_label: str = "",
) -> List[Dict]:
    """
    Drop rows whose chat-templated (system+user+assistant) token length
    exceeds max_length.

    SFTTrainer truncates over-length sequences, which for prompt-completion
    data can chop most of the assistant response and destroy the loss
    signal. Better to drop than train on a mangled row.
    """
    kept: List[Dict] = []
    dropped: List[int] = []
    for row in rows:
        system_prompt = (row.get("system_prompt") or "").strip()
        user_prompt = (row.get("user_prompt") or "").strip()
        assistant_text = (row.get("response") or "").strip()
        if not user_prompt or not assistant_text:
            continue
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        msgs.append({"role": "assistant", "content": assistant_text})
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False
        )
        if len(ids) <= max_length:
            kept.append(row)
        else:
            dropped.append(len(ids))
    if dropped:
        dropped_sorted = sorted(dropped, reverse=True)
        label = f"{split_name}/{substep_label}" if substep_label else split_name
        print(
            f"  [length filter] {label}: dropped {len(dropped)}/"
            f"{len(rows)} rows over max_length={max_length}  "
            f"(longest dropped: {dropped_sorted[0]}, shortest dropped: "
            f"{dropped_sorted[-1]})"
        )
    return kept


def to_prompt_completion(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a JSONL record to TRL's conversational prompt-completion format
    so SFTTrainer's _collate_prompt_completion masks the prompt tokens to
    -100 and only computes loss on the assistant response.
    """
    system_prompt = (example.get("system_prompt") or "").strip()
    user_prompt = (example.get("user_prompt") or "").strip()
    assistant_text = (example.get("response") or "").strip()
    if not user_prompt:
        raise ValueError("Row has empty user_prompt")
    if not assistant_text:
        raise ValueError("Row has empty response")
    prompt_messages: List[Dict[str, Any]] = []
    if system_prompt:
        prompt_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.append({"role": "user", "content": user_prompt})
    completion_messages: List[Dict[str, Any]] = [
        {"role": "assistant", "content": assistant_text}
    ]
    return {"prompt": prompt_messages, "completion": completion_messages}


def build_combined_dataset(
    rows: List[Dict],
    tokenizer,
    max_length: int,
    split_name: str,
) -> Optional[Dataset]:
    """Filter every substep by length, concatenate, then convert to prompt-completion."""
    counts_before = Counter(
        r.get("substep") for r in rows if r.get("substep") in SUBSTEP_ORDER
    )

    kept_rows: List[Dict] = []
    for substep in SUBSTEP_ORDER:
        sub_rows = filter_substep(rows, substep)
        sub_rows = filter_by_length(
            sub_rows, tokenizer, max_length, split_name, substep
        )
        kept_rows.extend(sub_rows)

    counts_after = Counter(r.get("substep") for r in kept_rows)

    print(f"\n  [{split_name}] substep distribution after length filtering:")
    print(f"    {'substep':<22s} {'kept':>6s} / {'total':>6s}")
    for substep in SUBSTEP_ORDER:
        kept = counts_after.get(substep, 0)
        total = counts_before.get(substep, 0)
        print(f"    {substep:<22s} {kept:>6d} / {total:>6d}")
    print(f"    {'TOTAL':<22s} "
          f"{sum(counts_after.values()):>6d} / "
          f"{sum(counts_before.values()):>6d}")

    if not kept_rows:
        return None

    ds = Dataset.from_list(kept_rows)
    ds = ds.map(
        to_prompt_completion,
        remove_columns=ds.column_names,
        num_proc=NUM_PROC,
    )
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def load_base_model():
    """
    Load Nemotron-3-Nano-30B-A3B in BF16 with eager attention and
    device_map="balanced" for multi-GPU sharding. Same kwargs as the
    per-substep variant — see finetune_nemotron.py for the rationale.
    """
    print(f"\n{'='*60}")
    print(f"Loading base model: {MODEL_NAME}")
    print(f"{'='*60}\n")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        use_cache=False,
        device_map="balanced",
    )
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train_unified(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    tokenizer,
    output_root: str,
):
    """
    Train one LoRA adapter on the combined dataset. Returns the wrapped
    PEFT model so the caller can immediately merge_and_unload(), or None
    if training was skipped because the adapter already exists.
    """
    adapter_dir = os.path.join(output_root, "adapter")

    if os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        print(f"\n✓ unified adapter already exists at {adapter_dir} — skipping training\n")
        return None

    base_model = load_base_model()
    peft_model = get_peft_model(base_model, PEFT_CONFIG)
    peft_model.print_trainable_parameters()

    total_train = len(train_dataset)
    steps_per_epoch = max(1, total_train // (PER_DEVICE_BATCH * GRAD_ACCUM))

    save_eval_strategy = "steps"
    save_eval_steps = int(os.environ.get("SAVE_STEPS", 50))

    print(f"\n{'='*60}")
    print(f"Unified training")
    print(f"  Train examples : {total_train}")
    print(f"  Eval examples  : {len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"  Max length     : {MAX_LENGTH}")
    print(f"  Batch size     : {PER_DEVICE_BATCH} × {GRAD_ACCUM} accum")
    print(f"  Epochs         : {NUM_EPOCHS}")
    print(f"  Steps / epoch  : {steps_per_epoch}")
    print(f"  Save / eval    : every {save_eval_strategy}"
          + (f" ({save_eval_steps} steps)" if save_eval_steps else ""))
    print(f"{'='*60}\n")

    training_args_kwargs = dict(
        output_dir=output_root,
        learning_rate=LEARNING_RATE,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=NUM_EPOCHS,
        logging_steps=1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_length=MAX_LENGTH,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": MIN_LR_RATE},
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        report_to="none",
        seed=SEED,
        dataset_num_proc=NUM_PROC,
        save_strategy=save_eval_strategy,
        save_total_limit=3,
        completion_only_loss=True,
    )
    if save_eval_steps is not None:
        training_args_kwargs["save_steps"] = save_eval_steps
    if eval_dataset is not None:
        # Eval is expensive (~30 min on this dataset), so run it only once per
        # epoch — not on every save. load_best_model_at_end requires
        # save_strategy == eval_strategy, so we leave it off here; resume from
        # the latest checkpoint instead.
        training_args_kwargs.update(eval_strategy="epoch")

    training_args = SFTConfig(**training_args_kwargs)

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    checkpoints = sorted(
        glob.glob(os.path.join(output_root, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    resume_from = checkpoints[-1] if checkpoints else None
    if resume_from:
        print(f"  ↪ Resuming from checkpoint: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics_path = os.path.join(output_root, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\n✓ adapter saved to {adapter_dir}")
    print(f"  metrics saved to {metrics_path}\n")

    for ckpt in glob.glob(os.path.join(output_root, "checkpoint-*")):
        shutil.rmtree(ckpt, ignore_errors=True)

    # Drop trainer/optimizer state but keep peft_model so the caller can merge
    # without reloading the base model from scratch.
    peft_model.zero_grad(set_to_none=True)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return peft_model


# ──────────────────────────────────────────────────────────────────────────────
# Merging
# ──────────────────────────────────────────────────────────────────────────────
def merge_adapter(
    output_root: str,
    tokenizer,
    peft_model=None,
):
    """
    Merge the trained LoRA into the base model and save the result as a
    normal HuggingFace BF16 checkpoint that vLLM can serve without
    --enable-lora.
    """
    adapter_dir = os.path.join(output_root, "adapter")
    merged_dir = os.path.join(output_root, "merged")

    if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        raise FileNotFoundError(
            f"adapter not found at {adapter_dir}; run training first"
        )

    if os.path.isfile(os.path.join(merged_dir, "config.json")):
        print(f"✓ merged model already exists at {merged_dir} — skipping merge\n")
        return

    print(f"\n{'='*60}")
    print(f"Merging LoRA adapter into base model")
    print(f"  Adapter : {adapter_dir}")
    print(f"  Output  : {merged_dir}")
    print(f"{'='*60}\n")

    if peft_model is None:
        # Cold path (e.g. --only-merge or training was skipped because the
        # adapter already existed).
        base_model = load_base_model()
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    merged_model = peft_model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(
        merged_dir,
        max_shard_size="5GB",
        safe_serialization=True,
    )
    tokenizer.save_pretrained(merged_dir)
    print(f"\n✓ merged model saved to {merged_dir}\n")

    del merged_model
    del peft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# vLLM serving script (single merged model — no --enable-lora needed)
# ──────────────────────────────────────────────────────────────────────────────
def write_serve_script(output_root: str):
    serve_path = os.path.join(output_root, "serve.sh")
    serve_script = f"""#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# serve.sh — vLLM serving for the unified merged Nemotron model.
# ──────────────────────────────────────────────────────────────────────────────
# The merged BF16 checkpoint already contains the LoRA weights, so vLLM is
# launched without --enable-lora. Clients call it with model="<merged_dir>".
#
# Usage:
#   bash serve.sh
#   PORT=8080 bash serve.sh
#   MAX_MODEL_LEN=32768 TP=4 bash serve.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PORT="${{PORT:-8000}}"
MAX_MODEL_LEN="${{MAX_MODEL_LEN:-16384}}"
GPU_MEMORY_UTILIZATION="${{GPU_MEMORY_UTILIZATION:-0.90}}"
TP="${{TP:-1}}"
MERGED_DIR="${{MERGED_DIR:-$SCRIPT_DIR/merged}}"

echo "================================================================="
echo "  Nemotron-3-Nano-30B-A3B — Unified merged model"
echo "================================================================="
echo "  Merged model     : $MERGED_DIR"
echo "  Port             : $PORT"
echo "  Max context len  : $MAX_MODEL_LEN"
echo "  Tensor parallel  : $TP"
echo "================================================================="

vllm serve "$MERGED_DIR" \\
    --port "$PORT" \\
    --tensor-parallel-size "$TP" \\
    --max-model-len "$MAX_MODEL_LEN" \\
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \\
    --trust-remote-code \\
    "$@"
"""
    with open(serve_path, "w") as f:
        f.write(serve_script)
    os.chmod(serve_path, 0o755)
    print(f"✓ Serving script written to {serve_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Nemotron-3-Nano-30B-A3B with one LoRA across "
                    "all substeps, then merge the adapter into the base model."
    )
    parser.add_argument("--train", default="train.jsonl",
                        help="Path to training JSONL (default: train.jsonl)")
    parser.add_argument("--test", default="test.jsonl",
                        help="Path to evaluation JSONL (default: test.jsonl)")
    parser.add_argument("--output", default="nemotron-3-nano-30b-unified",
                        help="Root output directory (default: nemotron-3-nano-30b-unified)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Train only; don't merge the adapter into the base model")
    parser.add_argument("--only-merge", action="store_true",
                        help="Skip training; just merge an existing adapter")
    args = parser.parse_args()

    output_root = args.output
    os.makedirs(output_root, exist_ok=True)
    tokenizer = load_tokenizer()

    if args.only_merge:
        merge_adapter(output_root, tokenizer)
        write_serve_script(output_root)
        return

    print("Loading datasets...")
    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.test)
    print(f"  Train: {len(train_rows)} rows")
    print(f"  Eval:  {len(eval_rows)} rows")

    train_dataset = build_combined_dataset(
        train_rows, tokenizer, MAX_LENGTH, "train"
    )
    eval_dataset = build_combined_dataset(
        eval_rows, tokenizer, MAX_LENGTH, "eval"
    )

    if train_dataset is None:
        print("ERROR: no training data after filtering")
        sys.exit(1)

    # Shuffle so substeps interleave within each epoch — otherwise the trainer
    # walks all rows of one substep before moving to the next, biasing the
    # optimizer and risking catastrophic forgetting of earlier substeps.
    train_dataset = train_dataset.shuffle(seed=SEED)

    peft_model = train_unified(
        train_dataset, eval_dataset, tokenizer, output_root
    )

    if not args.no_merge:
        merge_adapter(output_root, tokenizer, peft_model=peft_model)
    else:
        print("\n[--no-merge] Skipping merge step.")

    write_serve_script(output_root)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Output directory: {output_root}/")
    print(f"  adapter/      — LoRA-only weights")
    if not args.no_merge:
        print(f"  merged/       — merged model (base + LoRA, BF16)")
    print(f"  metrics.json  — trainer log_history")
    print(f"  serve.sh      — vLLM launch script")
    print(f"\nTo start serving:")
    print(f"  bash {output_root}/serve.sh")
    print()


if __name__ == "__main__":
    main()
