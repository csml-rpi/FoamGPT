#!/usr/bin/env python3
"""
finetune_gemma4_26b_a4b_unified.py
===================================
Fine-tune Gemma 4 26B A4B with a SINGLE LoRA adapter trained on ALL substeps
together, then optionally merge the adapter back into the base model.

This is the Gemma counterpart to finetune_nemotron_unified.py. It keeps the
Gemma-specific Unsloth training path from finetune_gemma4_26b_a4b.py:
    - Load with FastModel.
    - Apply Unsloth's "gemma-4" chat template.
    - Pre-format rows into a single text column.
    - Mask prompt tokens with train_on_responses_only using Gemma 4 markers.

Output structure:
    gemma4-26b-a4b-unified/
    |-- adapter/       # LoRA-only weights
    |-- merged/        # base + LoRA bf16 checkpoint, if merge is enabled
    |-- base/          # tokenizer/config reference
    |-- metrics.json   # trainer log_history
    `-- serve.sh       # vLLM launch script for the merged model

Usage:
    python finetune_gemma4_26b_a4b_unified.py
    python finetune_gemma4_26b_a4b_unified.py --train data/train.jsonl --test data/test.jsonl
    python finetune_gemma4_26b_a4b_unified.py --no-merge
    python finetune_gemma4_26b_a4b_unified.py --only-merge

Common environment overrides:
    MAX_LENGTH=32768 python finetune_gemma4_26b_a4b_unified.py
    LORA_R=32 LORA_ALPHA=32 python finetune_gemma4_26b_a4b_unified.py
    MERGE_DTYPE=fp32 python finetune_gemma4_26b_a4b_unified.py --only-merge
"""

import argparse
import gc
import glob
import inspect
import json
import os
import shutil
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# Import Unsloth before torch/transformers so its patches are applied early.
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, Gemma4ForConditionalGeneration
from trl import SFTConfig, SFTTrainer


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/gemma-4-26B-A4B-it")
MERGE_BASE_MODEL = os.environ.get("BASE_MODEL", MODEL_NAME)
SEED = int(os.environ.get("SEED", 3407))
NUM_PROC = int(os.environ.get("NUM_PROC", min(8, os.cpu_count() or 1)))

LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0").lower() in {"1", "true", "yes"}
LOAD_IN_16BIT = os.environ.get("LOAD_IN_16BIT", "1").lower() in {"1", "true", "yes"}
FULL_FINETUNING = os.environ.get("FULL_FINETUNING", "0").lower() in {"1", "true", "yes"}

CHAT_TEMPLATE_NAME = "gemma-4"
GEMMA4_INSTRUCTION_PART = "<|turn>user\n"
GEMMA4_RESPONSE_PART = "<|turn>model\n"

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


# -----------------------------------------------------------------------------
# Unified training configuration
# -----------------------------------------------------------------------------
# One LoRA learns every substep, so use one global max_length. Rows longer than
# this are filtered out per substep; we do not truncate because that can remove
# assistant tokens and break the response-only loss mask.
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 16384))
PER_DEVICE_BATCH = int(os.environ.get("PER_DEVICE_BATCH", 1))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 16))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", 3))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.05))
MIN_LR_RATE = float(os.environ.get("MIN_LR_RATE", 0.10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))


# -----------------------------------------------------------------------------
# LoRA configuration for Gemma 4 26B A4B
# -----------------------------------------------------------------------------
LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 16))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.0))

GEMMA4_LORA_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# -----------------------------------------------------------------------------
# Merge configuration
# -----------------------------------------------------------------------------
MAX_SHARD = os.environ.get("MAX_SHARD", "5GB")
SAFE_MERGE = os.environ.get("SAFE_MERGE", "1").lower() not in {"0", "false", "no"}
MERGE_DTYPE_NAME = os.environ.get("MERGE_DTYPE", "bf16").lower()
MERGE_DTYPE = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}[MERGE_DTYPE_NAME]
SAVE_DTYPE = torch.bfloat16

TOKENIZER_FILES = (
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
)


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def filter_substep(rows: List[Dict[str, Any]], substep: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("substep") == substep]


def row_to_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    system_prompt = (row.get("system_prompt") or "").strip()
    user_prompt = (row.get("user_prompt") or "").strip()
    assistant_text = (row.get("response") or "").strip()
    if not user_prompt or not assistant_text:
        return None

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": assistant_text})
    return messages


def filter_by_length(
    rows: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    split_name: str,
    substep_label: str = "",
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    dropped: List[int] = []

    for row in rows:
        messages = row_to_messages(row)
        if messages is None:
            continue
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if len(ids) <= max_length:
            kept.append(row)
        else:
            dropped.append(len(ids))

    if dropped:
        dropped_sorted = sorted(dropped, reverse=True)
        label = f"{split_name}/{substep_label}" if substep_label else split_name
        print(
            f"  [length filter] {label}: dropped {len(dropped)}/{len(rows)} "
            f"rows over max_length={max_length} "
            f"(longest={dropped_sorted[0]}, shortest={dropped_sorted[-1]})"
        )
    return kept


def build_text_dataset(rows: List[Dict[str, Any]], tokenizer) -> Dataset:
    """
    Render each row to a single text column using the gemma-4 chat template.
    """
    add_bos = bool(getattr(tokenizer, "add_bos_token", False))
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and not add_bos:
        add_bos = bool(getattr(inner, "add_bos_token", False))

    def _format(examples):
        n = len(examples["user_prompt"])
        sysprompts = examples.get("system_prompt", [""] * n)
        responses = examples["response"]

        texts: List[str] = []
        for i in range(n):
            system_prompt = (
                (sysprompts[i] or "").strip() if sysprompts[i] is not None else ""
            )
            user_prompt = (examples["user_prompt"][i] or "").strip()
            assistant_text = (responses[i] or "").strip()

            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": assistant_text})

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if add_bos:
                text = text.removeprefix("<bos>")
            texts.append(text)
        return {"text": texts}

    raw = Dataset.from_list(rows)
    return raw.map(
        _format,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=NUM_PROC,
    )


def build_combined_rows(
    rows: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    split_name: str,
) -> List[Dict[str, Any]]:
    counts_before = Counter(
        row.get("substep") for row in rows if row.get("substep") in SUBSTEP_ORDER
    )

    kept_rows: List[Dict[str, Any]] = []
    for substep in SUBSTEP_ORDER:
        sub_rows = filter_substep(rows, substep)
        sub_rows = filter_by_length(
            sub_rows,
            tokenizer,
            max_length,
            split_name,
            substep,
        )
        kept_rows.extend(sub_rows)

    counts_after = Counter(row.get("substep") for row in kept_rows)

    print(f"\n  [{split_name}] substep distribution after length filtering:")
    print(f"    {'substep':<22s} {'kept':>6s} / {'total':>6s}")
    for substep in SUBSTEP_ORDER:
        kept = counts_after.get(substep, 0)
        total = counts_before.get(substep, 0)
        print(f"    {substep:<22s} {kept:>6d} / {total:>6d}")
    print(
        f"    {'TOTAL':<22s} "
        f"{sum(counts_after.values()):>6d} / "
        f"{sum(counts_before.values()):>6d}"
    )

    return kept_rows


# -----------------------------------------------------------------------------
# Model/tokenizer loading
# -----------------------------------------------------------------------------
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE_NAME)
    return tokenizer


def load_base_model(max_seq_length: int) -> Tuple[torch.nn.Module, Any]:
    print("\n" + "=" * 72)
    print(f"Loading base model: {MODEL_NAME}")
    print(
        f"max_seq_length={max_seq_length}, "
        f"load_in_4bit={LOAD_IN_4BIT}, load_in_16bit={LOAD_IN_16BIT}"
    )
    print("=" * 72 + "\n")

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_16bit=LOAD_IN_16BIT,
        full_finetuning=FULL_FINETUNING,
        device_map="balanced",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE_NAME)

    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        inner.chat_template = getattr(tokenizer, "chat_template", None)

    return model, tokenizer


def add_lora_adapters(model: torch.nn.Module, max_seq_length: int) -> torch.nn.Module:
    if FULL_FINETUNING:
        print("FULL_FINETUNING=1: skipping LoRA adapter creation.")
        return model

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=max_seq_length,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


# -----------------------------------------------------------------------------
# TRL compatibility helpers
# -----------------------------------------------------------------------------
def make_sft_config(kwargs: Dict[str, Any]) -> SFTConfig:
    candidates: List[Dict[str, Any]] = []

    base = dict(kwargs)
    candidates.append(base)

    if "max_length" in base:
        alt = dict(base)
        alt["max_seq_length"] = alt.pop("max_length")
        candidates.append(alt)

    more_candidates: List[Dict[str, Any]] = []
    for item in candidates:
        more_candidates.append(item)
        if "eval_strategy" in item:
            alt = dict(item)
            alt["evaluation_strategy"] = alt.pop("eval_strategy")
            more_candidates.append(alt)
    candidates = more_candidates

    stripped_candidates: List[Dict[str, Any]] = []
    for item in candidates:
        stripped_candidates.append(item)
        alt = dict(item)
        alt.pop("lr_scheduler_kwargs", None)
        stripped_candidates.append(alt)
    candidates = stripped_candidates

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return SFTConfig(**candidate)
        except TypeError as exc:
            last_error = exc
            continue
    raise TypeError(f"Could not construct SFTConfig. Last error: {last_error}")


def make_trainer(model, args, train_dataset, eval_dataset, tokenizer) -> SFTTrainer:
    sft_processing = tokenizer
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        inner.chat_template = getattr(tokenizer, "chat_template", None)
        sft_processing = inner

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = sft_processing
    else:
        trainer_kwargs["tokenizer"] = sft_processing
    return SFTTrainer(**trainer_kwargs)


def checkpoint_sort_key(path: str) -> int:
    try:
        return int(path.rsplit("-", 1)[-1])
    except ValueError:
        return -1


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_unified(
    train_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    output_root: str,
) -> None:
    adapter_dir = os.path.join(output_root, "adapter")

    if os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        print(f"\nUnified adapter already exists at {adapter_dir}; skipping training.\n")
        return

    total_train = len(train_rows)
    total_eval = len(eval_rows)
    steps_per_epoch = max(1, total_train // (PER_DEVICE_BATCH * GRAD_ACCUM))
    save_eval_strategy = "steps"
    save_eval_steps = int(os.environ.get("SAVE_STEPS", 50))

    print("\n" + "=" * 72)
    print("Unified Gemma training")
    print(f"  Train examples : {total_train}")
    print(f"  Eval examples  : {total_eval}")
    print(f"  Max length     : {MAX_LENGTH}")
    print(f"  Batch size     : {PER_DEVICE_BATCH} with {GRAD_ACCUM} grad accum")
    print(f"  Epochs         : {NUM_EPOCHS}")
    print(f"  Steps / epoch  : {steps_per_epoch}")
    print(f"  LoRA           : r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print("=" * 72 + "\n")

    base_model, train_tokenizer = load_base_model(MAX_LENGTH)
    model = add_lora_adapters(base_model, MAX_LENGTH)

    train_dataset = build_text_dataset(train_rows, train_tokenizer)
    train_dataset = train_dataset.shuffle(seed=SEED)
    eval_dataset = build_text_dataset(eval_rows, train_tokenizer) if eval_rows else None

    training_args_kwargs: Dict[str, Any] = dict(
        output_dir=output_root,
        learning_rate=LEARNING_RATE,
        optim="adamw_8bit",
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
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        fp16=False,
        report_to="none",
        seed=SEED,
        dataset_num_proc=NUM_PROC,
        save_strategy=save_eval_strategy,
        save_steps=save_eval_steps,
        save_total_limit=3,
        dataset_text_field="text",
        packing=False,
    )

    if eval_dataset is not None:
        # Eval is expensive for the long-context rows. Run it once per epoch,
        # while checkpointing by steps for robust resume.
        training_args_kwargs.update(eval_strategy="epoch")

    training_args = make_sft_config(training_args_kwargs)
    trainer = make_trainer(model, training_args, train_dataset, eval_dataset, train_tokenizer)

    trainer = train_on_responses_only(
        trainer,
        instruction_part=GEMMA4_INSTRUCTION_PART,
        response_part=GEMMA4_RESPONSE_PART,
    )

    checkpoints = sorted(
        glob.glob(os.path.join(output_root, "checkpoint-*")),
        key=checkpoint_sort_key,
    )
    resume_from = checkpoints[-1] if checkpoints else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    train_tokenizer.save_pretrained(adapter_dir)

    metrics_path = os.path.join(output_root, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\nAdapter saved to {adapter_dir}")
    print(f"Metrics saved to {metrics_path}\n")

    for ckpt in glob.glob(os.path.join(output_root, "checkpoint-*")):
        shutil.rmtree(ckpt, ignore_errors=True)

    del trainer
    del model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------
def already_merged(merged_dir: str) -> bool:
    return (
        os.path.isfile(os.path.join(merged_dir, "config.json"))
        and len(glob.glob(os.path.join(merged_dir, "*.safetensors"))) > 0
    )


def merge_adapter(output_root: str) -> None:
    adapter_dir = os.path.join(output_root, "adapter")
    merged_dir = os.path.join(output_root, "merged")

    if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        raise FileNotFoundError(f"adapter not found at {adapter_dir}; run training first")

    if already_merged(merged_dir):
        print(f"Merged model already exists at {merged_dir}; skipping merge.\n")
        return

    print("\n" + "=" * 72)
    print("Merging Gemma LoRA adapter into base model")
    print(f"  Base    : {MERGE_BASE_MODEL}")
    print(f"  Adapter : {adapter_dir}")
    print(f"  Output  : {merged_dir}")
    print(f"  Dtype   : {MERGE_DTYPE_NAME} (save dtype: bf16)")
    print("=" * 72 + "\n")

    adapter_cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    adapter_base = adapter_cfg.get("base_model_name_or_path")
    if adapter_base and adapter_base.lower() != MERGE_BASE_MODEL.lower():
        print(f"[WARN] adapter points to {adapter_base}, BASE_MODEL is {MERGE_BASE_MODEL}")

    base_model = Gemma4ForConditionalGeneration.from_pretrained(
        MERGE_BASE_MODEL,
        dtype=MERGE_DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    model = model.merge_and_unload(safe_merge=SAFE_MERGE)
    if MERGE_DTYPE != SAVE_DTYPE:
        print(f"Casting merged weights {MERGE_DTYPE_NAME} -> bf16 for save...")
        model = model.to(SAVE_DTYPE)

    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size=MAX_SHARD,
    )

    copied_from_adapter: List[str] = []
    for filename in TOKENIZER_FILES:
        src = os.path.join(adapter_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(merged_dir, filename))
            copied_from_adapter.append(filename)

    if copied_from_adapter:
        print(f"Copied tokenizer/processor files from adapter: {copied_from_adapter}")
    else:
        print("Adapter has no tokenizer files; saving tokenizer/processor from base.")
        try:
            processor = AutoProcessor.from_pretrained(MERGE_BASE_MODEL, trust_remote_code=True)
            processor.save_pretrained(merged_dir)
        except Exception as exc:
            print(f"AutoProcessor failed ({exc}); falling back to AutoTokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(MERGE_BASE_MODEL, trust_remote_code=True)
            tokenizer.save_pretrained(merged_dir)

    print(f"\nMerged model saved to {merged_dir}\n")

    del model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def save_base_reference(output_root: str, tokenizer) -> None:
    base_dir = os.path.join(output_root, "base")
    os.makedirs(base_dir, exist_ok=True)
    tokenizer.save_pretrained(base_dir)

    config_path = os.path.join(base_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_model": MODEL_NAME,
                "merge_base_model": MERGE_BASE_MODEL,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "lora_target_modules": GEMMA4_LORA_TARGET_MODULES,
                "load_in_4bit": LOAD_IN_4BIT,
                "load_in_16bit": LOAD_IN_16BIT,
                "chat_template": CHAT_TEMPLATE_NAME,
                "max_length": MAX_LENGTH,
            },
            f,
            indent=2,
        )


def write_serve_script(output_root: str) -> None:
    serve_path = os.path.join(output_root, "serve.sh")
    serve_script = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP="${TP:-1}"
MERGED_DIR="${MERGED_DIR:-$SCRIPT_DIR/merged}"

cat <<EOF
=================================================================
  Gemma 4 26B A4B - Unified merged model
=================================================================
  Merged model     : $MERGED_DIR
  Port             : $PORT
  Max context len  : $MAX_MODEL_LEN
  Tensor parallel  : $TP
=================================================================
EOF

vllm serve "$MERGED_DIR" \\
    --port "$PORT" \\
    --tensor-parallel-size "$TP" \\
    --max-model-len "$MAX_MODEL_LEN" \\
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \\
    --trust-remote-code \\
    "$@"
"""
    with open(serve_path, "w", encoding="utf-8") as f:
        f.write(serve_script)
    os.chmod(serve_path, 0o755)
    print(f"Serving script written to {serve_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Gemma 4 26B A4B with one LoRA across all substeps, "
            "then optionally merge the adapter into the base model."
        )
    )
    parser.add_argument("--train", default="train.jsonl", help="Path to training JSONL")
    parser.add_argument("--test", default="test.jsonl", help="Path to evaluation JSONL")
    parser.add_argument(
        "--output",
        default="gemma4-26b-a4b-unified",
        help="Root output directory",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Train only; do not merge the adapter into the base model",
    )
    parser.add_argument(
        "--only-merge",
        action="store_true",
        help="Skip training; just merge an existing adapter",
    )
    args = parser.parse_args()

    output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    print("Gemma 4 26B A4B unified LoRA configuration")
    print(f"  Training model : {MODEL_NAME}")
    print(f"  Merge base     : {MERGE_BASE_MODEL}")
    print(f"  Chat template  : {CHAT_TEMPLATE_NAME}")
    print(f"  load_in_4bit   : {LOAD_IN_4BIT}")
    print(f"  load_in_16bit  : {LOAD_IN_16BIT}")
    print(f"  full finetune  : {FULL_FINETUNING}")
    print(f"  Max length     : {MAX_LENGTH}")
    print(f"  LoRA rank      : {LORA_R}")
    print(f"  LoRA alpha     : {LORA_ALPHA}")
    print(f"  LR             : {LEARNING_RATE}")

    if args.only_merge:
        merge_adapter(output_root)
        write_serve_script(output_root)
        return

    print("\nLoading tokenizer for length filtering...")
    length_tokenizer = load_tokenizer()
    save_base_reference(output_root, length_tokenizer)

    print("\nLoading datasets...")
    train_rows_raw = load_jsonl(args.train)
    eval_rows_raw = load_jsonl(args.test)
    print(f"  Train: {len(train_rows_raw)} rows")
    print(f"  Eval : {len(eval_rows_raw)} rows")

    train_rows = build_combined_rows(train_rows_raw, length_tokenizer, MAX_LENGTH, "train")
    eval_rows = build_combined_rows(eval_rows_raw, length_tokenizer, MAX_LENGTH, "eval")

    if not train_rows:
        print("ERROR: no training data after filtering")
        sys.exit(1)

    train_unified(train_rows, eval_rows, output_root)

    if not args.no_merge:
        merge_adapter(output_root)
    else:
        print("\n[--no-merge] Skipping merge step.")

    write_serve_script(output_root)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Output directory: {output_root}/")
    print("  adapter/      - LoRA-only weights")
    if not args.no_merge:
        print("  merged/       - merged model (base + LoRA, bf16)")
    print("  base/         - tokenizer/config reference")
    print("  metrics.json  - trainer log_history")
    print("  serve.sh      - vLLM launch script for merged model")
    print("\nTo start serving:")
    print(f"  bash {output_root}/serve.sh")


if __name__ == "__main__":
    main()
