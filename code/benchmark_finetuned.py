import os
import json
import shutil
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmark_utils import DATASETS_BASIC, CASES_BASIC, DIR_BASIC


# Markers printed when the agent loop terminates normally. See src/main.py:
#   - "Workflow completed successfully!"                                 (normal exit)
#   - "Workflow finished after reaching the maximum review loop limit." (max review loop)
# Their presence means the agent did everything it was going to do for this case.
# A failure INSIDE the solver (OpenFOAM rejects a file) is not a rerun trigger —
# the agent's work is deterministic, so rerunning would produce the same files.
_AGENT_COMPLETE_MARKERS = (
    "Workflow completed successfully!",
    "Workflow finished after reaching the maximum review loop limit.",
)


def case_status(exp_root: str, dataset: str, case) -> str:
    """Classify a prior attempt as one of:
      - 'never_started'  : no output.txt exists
      - 'complete'       : agent loop terminated normally (OpenFOAM errors OK)
      - 'aborted'        : agent bailed mid-loop (JSON parse error, exception,
                           SLURM kill) — no completion marker in output.txt
    """
    out_txt = os.path.join(exp_root, "results", dataset, str(case), "output.txt")
    if not os.path.isfile(out_txt):
        return "never_started"
    try:
        with open(out_txt, "r", errors="ignore") as f:
            text = f.read()
    except Exception:
        return "aborted"
    if any(m in text for m in _AGENT_COMPLETE_MARKERS):
        return "complete"
    return "aborted"


def case_has_output_marker(exp_root: str, dataset: str, case, marker: str) -> bool:
    out_txt = os.path.join(exp_root, "results", dataset, str(case), "output.txt")
    try:
        with open(out_txt, "r", errors="ignore") as handle:
            return marker.lower() in handle.read().lower()
    except OSError:
        return False

WM_PROJECT_DIR = os.environ.get('WM_PROJECT_DIR')
if not WM_PROJECT_DIR:
    print("Error: WM_PROJECT_DIR is not set in the environment.")
    exit(1)

EXP_FINETUNED = "./experiment-finetuned"

SUBSTEPS = [
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

STRONG_MODEL_CFG = {
    "model_provider": "openai-codex",
    "model_version": "gpt-5.3-codex",
    "temperature": 0.0,
}


def load_finetuned_config(path: str) -> dict:
    """Load per-substep fine-tuned model configs from a JSON file."""
    with open(path) as f:
        return json.load(f)


def build_finetuned_config(substep: str, ft_cfg: dict) -> dict:
    """All 3 services use STRONG; one substep is overridden with its fine-tuned model."""
    return {
        "models": {
            "plan":   STRONG_MODEL_CFG,
            "write":  STRONG_MODEL_CFG,
            "review": STRONG_MODEL_CFG,
        },
        "substep_model_overrides": {
            substep: ft_cfg[substep],
        },
    }


def build_all_finetuned_config(ft_cfg: dict) -> dict:
    """All substeps use their fine-tuned model. Service-level models use the
    same config as a fallback (they are never actually called when every
    substep has an override, but LLMService still instantiates them)."""
    fallback = next(iter(ft_cfg.values()))
    return {
        "models": {
            "plan":   dict(fallback),
            "write":  dict(fallback),
            "review": dict(fallback),
        },
        "substep_model_overrides": dict(ft_cfg),
    }


def run_benchmark(dataset, case, substep_label, model_config_path, clean_run_dir: bool = False):
    """Run foambench_main.py for one (dataset, case) with a pre-written config JSON."""
    folder_path = os.path.abspath(os.path.join(DIR_BASIC, dataset, str(case)))
    requirement_txt_path = os.path.abspath(os.path.join(folder_path, "usr_requirement.txt"))

    output_folder = os.path.abspath(os.path.join(
        EXP_FINETUNED, substep_label, "runs", dataset, str(case)
    ))
    if clean_run_dir and os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    case_id = f"Basic/{dataset}/{case}"
    dataset_log_path = os.path.abspath(os.path.join(
        EXP_FINETUNED, substep_label, "results", dataset, str(case), "dataset.jsonl"
    ))

    output_text_path = os.path.abspath(os.path.join(
        EXP_FINETUNED, substep_label, "results", dataset, str(case), "output.txt"
    ))
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)

    command = (
        f"python foambench_main.py"
        f" --openfoam_path {WM_PROJECT_DIR}"
        f" --output {output_folder}"
        f" --prompt_path {requirement_txt_path}"
        f" --dataset_log_path {dataset_log_path}"
        f" --case_id '{case_id}'"
        f" --model_config_path {model_config_path}"
    )
    print(f"Running: {command}")

    with open(output_text_path, 'w') as f:
        try:
            subprocess.run(command, shell=True, check=True, stdout=f, stderr=f)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for {dataset}/{case}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fine-tuned model benchmark: one substep at a time uses its fine-tuned model."
    )
    parser.add_argument(
        "--finetuned_config", type=str, default="./nemotron_models.json",
        help="Path to JSON file mapping substep names to model configs (default: ./nemotron_models.json — Nemotron Nano 30B base baseline; pass ./finetuned_models.json for the gpt-oss-20b LoRA adapters)."
    )
    parser.add_argument(
        "--substep", type=str, default=None,
        help=f"Run only a single named substep. Choices: {SUBSTEPS}"
    )
    parser.add_argument(
        "--all_finetuned", action="store_true",
        help="Run one experiment with ALL substeps using their fine-tuned models simultaneously."
    )
    parser.add_argument(
        "--run_label", type=str, default=None,
        help="Output subdirectory name for --all_finetuned mode (default: derived from the config's model_version)."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1; vLLM loads model per process — increase only if VRAM allows)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Run only cases that have no output.txt yet (never started). Skip any case with a prior "
             "attempt, regardless of outcome. Use --retry-agent-errors to ALSO rerun cases whose agent "
             "loop aborted mid-way (JSON parse errors, exceptions, SLURM kills)."
    )
    parser.add_argument(
        "--retry-agent-errors", action="store_true",
        help="Only meaningful with --resume. Also rerun cases where the previous attempt aborted mid-loop "
             "without reaching a completion marker (e.g. JSON parse errors, Python exceptions, SLURM kills). "
             "Their run/ dir is cleaned first. Cases where the agent loop ran to completion are still "
             "skipped — OpenFOAM-side solver errors are not re-attempted."
    )
    parser.add_argument(
        "--retry-agent-errors-only", action="store_true",
        help="Rerun only cases with an existing output.txt that lacks an agent completion marker. "
             "Skips both completed cases and never-started cases."
    )
    parser.add_argument(
        "--retry-context-errors-only", action="store_true",
        help="With --resume, retry aborted cases only when output.txt contains a maximum-context-length "
             "error. Never-started cases are still run. This avoids resampling model-level JSON or "
             "application failures while recovering an infrastructure context cap."
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.retry_agent_errors_only:
        args.resume = True
        args.retry_agent_errors = True
        if not args.all_finetuned and not args.substep:
            args.all_finetuned = True
            print("[retry] --retry-agent-errors-only defaults to --all_finetuned. "
                  "Pass --substep to retry a single-substep ablation instead.")
    if args.retry_context_errors_only:
        args.resume = True
        args.retry_agent_errors = True

    ft_cfg = load_finetuned_config(args.finetuned_config)

    if args.all_finetuned:
        if args.run_label:
            label = args.run_label
        else:
            first_cfg = next(iter(ft_cfg.values()))
            label = first_cfg["model_version"].split("/")[-1]
        runs = [(label, build_all_finetuned_config(ft_cfg))]
    elif args.substep:
        if args.substep not in SUBSTEPS:
            raise ValueError(f"Unknown substep '{args.substep}'. Choices: {SUBSTEPS}")
        if args.substep not in ft_cfg:
            raise ValueError(f"Substep '{args.substep}' not found in {args.finetuned_config}.")
        runs = [(args.substep, build_finetuned_config(args.substep, ft_cfg))]
    else:
        runs = []
        for s in SUBSTEPS:
            if s not in ft_cfg:
                print(f"Warning: substep '{s}' not in {args.finetuned_config} — skipping.")
                continue
            runs.append((s, build_finetuned_config(s, ft_cfg)))

    for substep_label, model_config in runs:
        config_dir = os.path.abspath(os.path.join(EXP_FINETUNED, substep_label))
        os.makedirs(config_dir, exist_ok=True)
        model_config_path = os.path.join(config_dir, "model_config.json")
        with open(model_config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        print(f"\n=== Fine-tuned benchmark: substep = '{substep_label}' ===")
        print(f"Config written to: {model_config_path}")

        exp_root = os.path.abspath(os.path.join(EXP_FINETUNED, substep_label))
        tasks = []
        skipped_complete = skipped_aborted = skipped_never_started = 0
        skipped_non_context = new_cases = retry_cases = 0
        for dataset in DATASETS_BASIC:
            for case in CASES_BASIC:
                if args.resume:
                    status = case_status(exp_root, dataset, case)
                    if status == "never_started" and args.retry_agent_errors_only:
                        print(f"[resume] skip {dataset}/{case} (never started)")
                        skipped_never_started += 1
                        continue
                    if status == "complete":
                        print(f"[resume] skip {dataset}/{case} (agent loop complete)")
                        skipped_complete += 1
                        continue
                    if status == "aborted" and not args.retry_agent_errors:
                        print(f"[resume] skip {dataset}/{case} (agent aborted; pass --retry-agent-errors to include)")
                        skipped_aborted += 1
                        continue
                    if (
                        status == "aborted"
                        and args.retry_context_errors_only
                        and not case_has_output_marker(
                            exp_root, dataset, case, "maximum context length"
                        )
                    ):
                        print(f"[resume] skip {dataset}/{case} (aborted without context-limit error)")
                        skipped_non_context += 1
                        continue
                    clean = (status == "aborted")
                    if status == "never_started":
                        new_cases += 1
                    else:
                        retry_cases += 1
                    tasks.append((dataset, case, substep_label, model_config_path, clean))
                else:
                    tasks.append((dataset, case, substep_label, model_config_path, False))

        if args.resume:
            print(f"[resume] total {len(tasks)} to run: {new_cases} never-started + {retry_cases} retried "
                  f"aborted; skipped {skipped_complete} complete, {skipped_aborted} aborted, "
                  f"{skipped_never_started} never-started, {skipped_non_context} non-context aborted.")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_benchmark, *t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Benchmark {task} failed: {e}")

# nohup python benchmark_finetuned.py > finetuned.log 2>&1 &
# nohup python benchmark_finetuned.py --all_finetuned > finetuned_all.log 2>&1 &
# python benchmark_finetuned.py --substep generate_file
# nohup python benchmark_finetuned.py --all_finetuned --resume >> nemotron_basic.log 2>&1 &                        # only never-started cases
# nohup python benchmark_finetuned.py --all_finetuned --resume --retry-agent-errors >> nemotron_basic.log 2>&1 &   # + retry agent aborts
# nohup python benchmark_finetuned.py --all_finetuned --retry-agent-errors-only >> nemotron_basic.log 2>&1 &       # only prior agent aborts
