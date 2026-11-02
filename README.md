<div align="center">

# FoamGPT: Fine-Tuning Large Language Models for Agentic Automation of CFD Simulations with OpenFOAM

[![Workshop paper](https://img.shields.io/badge/NeurIPS%202025-ML4PS%20workshop-4b44ce.svg)](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_271.pdf) [![Adapters](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-adapters-yellow)](https://huggingface.co/ghzlmc/foamgpt) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

FoamGPT fine-tunes open-weight language models to automate CFD simulations with OpenFOAM and [Foam-Agent](https://github.com/csml-rpi/Foam-Agent). The [NeurIPS 2025 ML4PS workshop paper](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_271.pdf) introduces the approach: curate training data from OpenFOAM tutorials, fine-tune a language model, and use it to generate simulation files within an agent workflow.

This repository contains the current implementation. It trains **one shared LoRA adapter** on teacher traces, including error diagnosis and repair, to serve **all nine LLM call sites** with a local 26B–30B model. The results below describe this implementation; the linked workshop paper documents the earlier study.

<p align="center">
  <img src="assets/overview.png" width="95%" alt="FoamGPT overview: teacher rollouts, one shared adapter for nine call sites, local execution">
</p>

## Results

Success rate on 33 held-out FoamBench cases: new parameter values of the 11 training case families. A case succeeds when the solver finishes with finite fields within ten attempts.

| Model | Size | Open weights | Success (%) |
|---|---:|:---:|---:|
| GPT-6 Sol | – | | 66.7 |
| **FoamGPT-Nemotron** (ours) | 30B | ✓ | **47.5** |
| DeepSeek-V4.1, untuned | 552B | ✓ | 30.3 |
| **FoamGPT-Gemma4** (ours) | 26B | ✓ | 29.3 |
| GPT-6 Luna | – | | 21.2 |
| Qwen3.8, untuned | 125B | ✓ | 21.2 |
| Nemotron-Nano, untuned | 30B | ✓ | 0.0 |

Splitting the same training data across three adapters, by pipeline stage or at random, lowers the success rate to 34.3–43.4%. [Per-case results](data/results/README.md) include field fidelity, the 16 out-of-distribution Advanced cases, and call-site ablations. [Data documentation](data/README.md) describes the teacher traces, evaluation cases, and retrieval indexes.

## Installation

Requirements: 4 GPUs with at least 80 GB each, [OpenFOAM 10](https://openfoam.org/version/10/), and Python 3.12.

```bash
git clone https://github.com/csml-rpi/FoamGPT.git && cd FoamGPT

# vLLM 0.20.1 serves Nemotron; vLLM 0.31.0 serves the Qwen3.8 and DeepSeek baselines
python3.12 -m venv ~/venvs/vllm020 && ~/venvs/vllm020/bin/pip install -r scripts/requirements-vllm0.20.1.txt
python3.12 -m venv ~/venvs/vllm031 && ~/venvs/vllm031/bin/pip install -r scripts/requirements-vllm0.31.0.txt
source ~/venvs/vllm020/bin/activate                    # provides the hf command used below

scripts/fetch_models.sh ~/models nano embedder         # base model at the pinned revision, retrieval embedder
hf download ghzlmc/foamgpt --exclude "references/*" --local-dir ~/adapters   # fine-tuned adapters
scripts/download_references.sh                         # reference solutions, needed only for field fidelity

# write the machine-specific paths into the configs
python3 scripts/localize.py --models-dir ~/models --openfoam-bashrc /opt/openfoam10/etc/bashrc \
    --venv020 ~/venvs/vllm020 --venv031 ~/venvs/vllm031 --adapters-dir ~/adapters
```

## Quick start

Evaluate FoamGPT-Nemotron on the 33 held-out cases, then score the run:

```bash
scripts/run_benchmark.sh E32_UNIFIED 1
```

One run takes 18–26 hours on a node with four A100 GPUs. Running the same command again resumes an interrupted run. Transcripts, case directories and scores are written to `code/experiment-finetuned/E32_UNIFIED_holdout_run1/`.

## Training and evaluation

### Training

```bash
# Run from the repository root, in the corresponding training environment
gunzip -k data/train.jsonl.gz data/test.jsonl.gz
python scripts/training/finetune_nemotron_unified.py --train data/train.jsonl --test data/test.jsonl --output nemotron-shared
python scripts/training/finetune_gemma4_26b_a4b_unified.py --train data/train.jsonl --test data/test.jsonl
```

Both scripts train one rank-16 LoRA adapter for three epochs on all nine call sites. The split and scaling adapters use the same script on subsets of the data. Settings and environments are in [`scripts/training/README.md`](scripts/training/README.md).

### Evaluation

Each system is a config that maps every call site to a model or adapter; `scripts/localize.py` writes them for your machine.

| System | Run |
|---|---|
| FoamGPT-Nemotron | `scripts/run_benchmark.sh E32_UNIFIED <run>` |
| Untuned Nemotron-Nano, Nemotron-Super, Qwen3.8, DeepSeek-V4, DeepSeek-V4.1 | The same command with `BASE_NANO`, `E33_SUPER_BASE` and `E37_SUPER_NOSTRIP`, `E46_QWEN38_NOSTRIP`, `E47_DSV4F_NOSTRIP`, `E48_DSV41F_NOSTRIP` |
| FoamGPT-Gemma4 | Merge the adapter into `unsloth/gemma-4-26b-a4b-it` with PEFT and serve the merged model |
| Shared vs. split adapters | `SPLIT_STAGE_SPECIFIC`, `SPLIT_RANDOM_R1`, `SPLIT_RANDOM_R2`, `SPLIT_RANDOM_R3` |
| Base model at one call site | The shared-adapter config with one call site's adapter removed |
| Advanced cases | Add `--advanced` to the command |

Run `python3 scripts/score_run.py <ID>_holdout_run<k>` to score a finished run again.

<details>
<summary>Evaluation protocol</summary>

Keep these settings to compare with the included results: temperature 0.6, a 100,000-token context, a 16,384-token output cap per call, at most ten solver attempts, one sequence at a time, eager mode, and tensor parallelism 4. A case succeeds when the solver log ends in `End`, the final time step is written, and all fields are finite. Crashes and malformed output count as failures. The configs of runs that cleaned each answer at its last `final` marker (the shared and split adapters, `E33_SUPER_BASE`) set `FOAMGPT_STRIP_FINAL=1`.

</details>

## Repository structure

```
code/          Agent harness, benchmark runners, and scoring code
configs/       Model maps, split-adapter maps, and chat templates
data/          Teacher traces, splits, manifests, and scaling subsets
  dataset/     Evaluation prompts, reference-set audit, and downloaded reference solutions
  retrieval/   OpenFOAM tutorial data and FAISS indexes
  results/     Per-case evaluation results
scripts/       Setup, downloads, benchmark execution, and scoring
  training/    Fine-tuning scripts and training environments
assets/        README figure
```

## Citation

```bibtex
@inproceedings{yue2025foamgpt,
  title     = {FoamGPT: Fine-tuning Large Language Model for Agentic Automation of CFD Simulations with OpenFOAM},
  author    = {Yue, Ling and Xu, Peijing and Zhang, Tingwen and Somasekharan, Nithin and Pan, Shaowu},
  booktitle = {Machine Learning and the Physical Sciences Workshop, NeurIPS},
  year      = {2025},
  url       = {https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_271.pdf}
}
```

## License and acknowledgments

The code is released under the [MIT License](LICENSE). `code/` is adapted from [Foam-Agent](https://github.com/csml-rpi/Foam-Agent) (MIT), and the evaluation cases come from FoamBench in [CFDLLMBench](https://github.com/NLR-Theseus/cfdllmbench) (BSD 3-Clause, [`data/dataset/LICENSE`](data/dataset/LICENSE)). The adapters follow the licenses of their base models. The teacher traces are outputs of Claude Opus 4.6 obtained through Amazon Bedrock and are subject to Anthropic's terms of service; they are released for research use.
