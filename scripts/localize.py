#!/usr/bin/env python3
"""Localize the kit to this machine.

Writes, from configs/experiments.json:
  configs/<ID>.json  per-substep model map for code/benchmark_finetuned*.py (paths filled in)
  configs/<ID>.env   VENV_ACTIVATE + the experiment's extra environment (sourced by scripts/run_benchmark.sh)
  configs/SPLIT_<NAME>.json/.env  the same for each split-adapter map in configs/split/
and env.sh at the kit root (OpenFOAM bashrc, venvs, HF cache).

Usage:
  python3 scripts/localize.py --models-dir /data/models \
      --openfoam-bashrc /opt/openfoam10/etc/bashrc \
      --venv020 /data/venvs/vllm020 --venv031 /data/venvs/vllm031 \
      [--adapters-dir /data/foamgpt_adapters] [--hf-home /data/hf]

Model folders expected under --models-dir (see scripts/fetch_models.sh): nemotron-3-nano-30b-a3b-bf16,
nemotron-3-super-120b-a12b-bf16, qwen3.8-flash-next-bf16, deepseek-v4-flash-0731, deepseek-v4.1-flash.
Experiments whose model folder or adapter is missing are still written, with a warning.
"""
import argparse
import glob
import json
import os
import shlex

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSTEPS = ["parse_case_info", "build_advice", "decompose_subtasks", "generate_file", "allrun_commands",
            "allrun_script", "error_analysis", "rewrite_plan", "rewrite_files"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--openfoam-bashrc", required=True)
    ap.add_argument("--venv020", required=True, help="venv with vLLM 0.20.1 (Nemotron runs)")
    ap.add_argument("--venv031", required=True, help="venv with vLLM 0.31.0 (Qwen3.8 / DeepSeek runs)")
    ap.add_argument("--adapters-dir", default=None, help="only for adapter experiments (E32)")
    ap.add_argument("--hf-home", default=None, help="HF cache holding Qwen/Qwen3-Embedding-0.6B")
    a = ap.parse_args()
    venvs = {"vllm020": os.path.abspath(a.venv020), "vllm031": os.path.abspath(a.venv031)}
    specs = json.load(open(os.path.join(KIT, "configs", "experiments.json")))
    for exp, s in specs.items():
        if exp.startswith("_"):
            continue
        model = os.path.join(os.path.abspath(a.models_dir), s["model"])
        entry = {"model_provider": "huggingface", "model_version": model, "temperature": 0.6}
        if "adapter" in s:
            if not a.adapters_dir:
                print(f"skip {exp}: needs --adapters-dir ({s['adapter']})")
                continue
            entry["lora_adapter_path"] = os.path.join(os.path.abspath(a.adapters_dir), s["adapter"])
        if "chat_template" in s:
            t = s["chat_template"]
            if t.startswith("@ADAPTERS_DIR@"):
                t = t.replace("@ADAPTERS_DIR@", os.path.abspath(a.adapters_dir or ""))
            else:
                t = os.path.join(KIT, "configs", "templates", t)
            entry["chat_template_path"] = t
        for p in [model] + [entry.get("lora_adapter_path", model), entry.get("chat_template_path", model)]:
            if not os.path.exists(p):
                print(f"  warning {exp}: {p} does not exist yet")
        json.dump({k: dict(entry) for k in SUBSTEPS}, open(os.path.join(KIT, "configs", f"{exp}.json"), "w"), indent=2)
        with open(os.path.join(KIT, "configs", f"{exp}.env"), "w") as f:
            f.write(f"VENV_ACTIVATE={shlex.quote(os.path.join(venvs[s['venv']], 'bin', 'activate'))}\n")
            for k, v in s.get("env", {}).items():
                f.write(f"export {k}={shlex.quote(v)}\n")
        print(f"wrote configs/{exp}.json, configs/{exp}.env")
    # Split-adapter systems: configs/split/<name>.json -> configs/SPLIT_<NAME>.json
    nano = os.path.join(os.path.abspath(a.models_dir), "nemotron-3-nano-30b-a3b-bf16")
    for path in sorted(glob.glob(os.path.join(KIT, "configs", "split", "*.json"))):
        exp = "SPLIT_" + os.path.splitext(os.path.basename(path))[0].upper()
        if not a.adapters_dir:
            print(f"skip {exp}: needs --adapters-dir")
            continue
        text = open(path).read().replace("@BASE_MODEL_DIR@", nano)
        text = text.replace("@ADAPTERS_DIR@", os.path.abspath(a.adapters_dir))
        with open(os.path.join(KIT, "configs", f"{exp}.json"), "w") as f:
            f.write(text)
        with open(os.path.join(KIT, "configs", f"{exp}.env"), "w") as f:
            f.write(f"VENV_ACTIVATE={shlex.quote(os.path.join(venvs['vllm020'], 'bin', 'activate'))}\n")
            f.write("export FOAMGPT_STRIP_FINAL=1\n")
        print(f"wrote configs/{exp}.json, configs/{exp}.env")
    with open(os.path.join(KIT, "env.sh"), "w") as f:
        f.write(f"export FOAMGPT_KIT={shlex.quote(KIT)}\n")
        f.write(f"export FOAMGPT_OPENFOAM_BASHRC={shlex.quote(os.path.abspath(a.openfoam_bashrc))}\n")
        if a.hf_home:
            f.write(f"export HF_HOME={shlex.quote(os.path.abspath(a.hf_home))}\n")
    print("wrote env.sh")


if __name__ == "__main__":
    main()
