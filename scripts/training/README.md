# Training

| File | Use |
|---|---|
| `finetune_nemotron_unified.py` | Shared Nemotron adapter; also the split adapters (run on the records of one group's call sites) and the scaling adapters (run on one level's records) |
| `finetune_gemma4_26b_a4b_unified.py` | Shared Gemma4 adapter (Unsloth `FastModel`, response-only loss) |
| `build_manifest.py` | Applies the 16,384-token length filter and writes the manifests in `data/manifests/` |
| `make_subsets.py` | Builds the nested scaling subsets in `data/scaling/` |
| `requirements-nemotron.txt`, `requirements-gemma4.txt` | Training environments |

Settings (defaults of both scripts): learning rate 2e-4, cosine schedule with a 0.1 minimum-rate floor, warmup ratio 0.05, 3 epochs, per-device batch 1 with gradient accumulation 16, weight decay 0.01, seed 3407, maximum length 16,384 tokens (longer records are dropped), loss on completion tokens only, thinking disabled. Nemotron: LoRA r = 16, alpha = 32, dropout 0, on `q/k/v/o_proj`, `gate/up/down_proj`, and the Mamba `in/out_proj`; 470 optimizer steps. Gemma4: LoRA r = 16, alpha = 16, dropout 0, on the language-tower attention and feed-forward linears; 471 optimizer steps.

Run from the repository root. Use a separate environment for each model, installing the corresponding requirements file in `scripts/training/`.

```bash
gunzip -k data/train.jsonl.gz data/test.jsonl.gz
python scripts/training/finetune_nemotron_unified.py --train data/train.jsonl --test data/test.jsonl --output nemotron-shared
python scripts/training/finetune_gemma4_26b_a4b_unified.py --train data/train.jsonl --test data/test.jsonl
# settings can be overridden through environment variables (LEARNING_RATE, NUM_EPOCHS, LORA_R, ...; see the top of each script)
```

Split adapters: filter `train.jsonl` to the call sites of one group (the maps are in `configs/split/`) and train one adapter per group with the same script and settings. Scaling adapters: `python scripts/training/make_subsets.py` writes each level's `train.jsonl` to `data/scaling/subsets/`; train each with the same script and settings.
