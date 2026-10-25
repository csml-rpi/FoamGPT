"""Summarize Foam-Agent benchmark success and token usage.

Walks <exp_root>/results/<dataset>/<case>/output.txt for token counts and
<exp_root>/runs/<dataset>/<case>/ for the actual solver outcome. Success
matches the rule used by ablation-notebooks/utils_finetuned.py so this
script agrees with the notebook analysis:

  - success:         at least one log.*Foam in the run dir has "End" as
                     its second-to-last line (OpenFOAM's clean-finish
                     marker emitted by Time::run()).
  - early-exit:      failed AND agent did not exhaust its retry loop
                     (no "Maximum loop count (N) reached." marker AND
                     last observed "Loop N" < max_loop). This means the
                     workflow crashed — e.g. JSON parse error — before
                     giving the review loop a chance. These are the
                     cases worth re-running.
  - max-loop-reached: failed AND agent exhausted its retry loop. The
                     agent already tried max_loop times; re-running is
                     unlikely to help.

Token counts are parsed from "Total prompt tokens: / Total completion
tokens: / Total tokens:" lines (last occurrence wins).

Usage
-----
  python summarize_benchmark.py --exp_root experiment-finetuned/<model>
  python summarize_benchmark.py --save_json report.json
  python summarize_benchmark.py --rerun_agent_failures       # re-run early-exit cases only
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from benchmark_utils import DIR_BASIC


TOKEN_PATTERNS = {
    "prompt": re.compile(r"Total prompt tokens:\s*(\d+)", re.IGNORECASE),
    "completion": re.compile(r"Total completion tokens:\s*(\d+)", re.IGNORECASE),
    "total": re.compile(r"Total tokens:\s*(\d+)", re.IGNORECASE),
    # Engine-reported vLLM counts (populated when model_provider=huggingface).
    # cached_prompt is the OpenAI-API equivalent of
    # prompt_tokens_details.cached_tokens — prompt tokens served from vLLM's
    # prefix cache. engine_prompt / engine_completion are vLLM's own view of
    # the rendered (chat-template) prompt and generated output.
    "cached_prompt": re.compile(r"Total cached prompt tokens:\s*(\d+)", re.IGNORECASE),
    "engine_prompt": re.compile(r"Total engine prompt tokens:\s*(\d+)", re.IGNORECASE),
    "engine_completion": re.compile(r"Total engine completion tokens:\s*(\d+)", re.IGNORECASE),
}

LOOP_PATTERN = re.compile(r"Loop\s+(\d+)", re.IGNORECASE)
MAX_LOOP_REACHED_PATTERN = re.compile(
    r"Maximum loop count\s*\((\d+)\)\s*reached", re.IGNORECASE
)
DEFAULT_MAX_LOOP = 10  # mirrors src/config.py Config.max_loop


def parse_tokens(text: str) -> Dict[str, int]:
    out = {k: 0 for k in TOKEN_PATTERNS}
    for k, pat in TOKEN_PATTERNS.items():
        m = pat.findall(text)
        if m:
            out[k] = int(m[-1])
    if out["total"] == 0 and (out["prompt"] or out["completion"]):
        out["total"] = out["prompt"] + out["completion"]
    return out


def solver_logs(run_case_dir: Path) -> List[Path]:
    """Return all files named log.*Foam in the run dir (solver logs)."""
    if not run_case_dir.is_dir():
        return []
    return [
        p for p in run_case_dir.iterdir()
        if p.is_file() and p.name.startswith("log.") and p.name.endswith("Foam")
    ]


def solver_ended_cleanly(run_case_dir: Path) -> bool:
    """True iff any log.*Foam ends with 'End' as its second-to-last line.

    Mirrors ablation-notebooks/utils_finetuned.py:success(). Requires the
    run dir to exist and to contain at least one subdirectory (so that a
    run that only wrote input files, with no time step output, does not
    count as a success).
    """
    if not run_case_dir.is_dir():
        return False
    if not any(p.is_dir() for p in run_case_dir.iterdir()):
        return False
    for log in solver_logs(run_case_dir):
        try:
            lines = log.read_text(errors="ignore").splitlines()
        except Exception:
            return False
        if len(lines) > 1 and lines[-2].strip() == "End":
            return True
    return False


def case_generated(run_case_dir: Path) -> bool:
    """True if OpenFOAM wrote any numeric time directory other than '0'."""
    if not run_case_dir.is_dir():
        return False
    for p in run_case_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.strip()
        if name == "0":
            continue
        try:
            float(name)
            return True
        except ValueError:
            continue
    return False


def parse_last_loop(text: str) -> int:
    """Return the highest 'Loop N' number seen in output.txt (0 if none)."""
    nums = [int(m) for m in LOOP_PATTERN.findall(text)]
    return max(nums) if nums else 0


def max_loop_reached(text: str) -> bool:
    """True if output.txt contains the router's 'Maximum loop count reached' marker."""
    return bool(MAX_LOOP_REACHED_PATTERN.search(text))


def failure_reason(text: str, last_loop: int, max_loop: int) -> str:
    """Categorize a failed case.

    Returns:
      - "max-loop-reached": router hit the retry cap, or last loop >= max_loop.
      - "early-exit":       workflow crashed before exhausting retries
                            (e.g. JSON parse error) — worth re-running.
    """
    if max_loop_reached(text) or last_loop >= max_loop:
        return "max-loop-reached"
    return "early-exit"


def summarize(exp_root: Path, max_loop: int = DEFAULT_MAX_LOOP) -> Tuple[Dict, List[Dict]]:
    runs_root = exp_root / "runs"
    results_root = exp_root / "results"

    if not runs_root.exists() or not results_root.exists():
        raise FileNotFoundError(
            f"Expected both runs/ and results/ under {exp_root}. "
            f"Got: runs={runs_root.exists()}, results={results_root.exists()}"
        )

    case_records: List[Dict] = []

    for output_txt in sorted(results_root.rglob("output.txt")):
        rel = output_txt.relative_to(results_root)
        if len(rel.parts) < 3:
            continue
        dataset = rel.parts[0]
        case = rel.parts[1]
        run_case_dir = runs_root / dataset / case

        text = output_txt.read_text(encoding="utf-8", errors="ignore")
        success = solver_ended_cleanly(run_case_dir)
        generated = case_generated(run_case_dir)
        tokens = parse_tokens(text)
        last_loop = parse_last_loop(text)

        case_records.append({
            "dataset": dataset,
            "case": case,
            "success": success,
            "generated": generated,
            "has_solver_log": bool(solver_logs(run_case_dir)),
            "last_loop": last_loop,
            "max_loop_reached": max_loop_reached(text),
            "failure_reason": None if success else failure_reason(text, last_loop, max_loop),
            "prompt_tokens": tokens["prompt"],
            "completion_tokens": tokens["completion"],
            "total_tokens": tokens["total"],
            "cached_prompt_tokens": tokens["cached_prompt"],
            "engine_prompt_tokens": tokens["engine_prompt"],
            "engine_completion_tokens": tokens["engine_completion"],
            "output_path": str(output_txt),
            "run_case_dir": str(run_case_dir),
        })

    total_cases = len(case_records)
    successes = sum(1 for r in case_records if r["success"])
    total_tokens = sum(r["total_tokens"] for r in case_records)
    total_cached = sum(r["cached_prompt_tokens"] for r in case_records)
    total_engine_prompt = sum(r["engine_prompt_tokens"] for r in case_records)
    total_engine_completion = sum(r["engine_completion_tokens"] for r in case_records)

    summary = {
        "exp_root": str(exp_root),
        "total_cases": total_cases,
        "successes": successes,
        "success_rate": (successes / total_cases) if total_cases else 0.0,
        "total_tokens": total_tokens,
        "average_tokens": (total_tokens / total_cases) if total_cases else 0.0,
        "total_cached_prompt_tokens": total_cached,
        "total_engine_prompt_tokens": total_engine_prompt,
        "total_engine_completion_tokens": total_engine_completion,
        "cache_hit_rate": (total_cached / total_engine_prompt) if total_engine_prompt else 0.0,
    }
    return summary, case_records


def agent_failed_cases(cases: List[Dict]) -> List[Dict]:
    """Cases that crashed before exhausting the retry loop — worth re-running."""
    return [c for c in cases if c["failure_reason"] == "early-exit"]


def exhausted_cases(cases: List[Dict]) -> List[Dict]:
    """Cases that ran the retry loop to its cap without succeeding. Not rerun candidates."""
    return [c for c in cases if c["failure_reason"] == "max-loop-reached"]


def rerun_case(case: Dict, exp_root: Path, repo_root: Path, dataset_root: str) -> int:
    """Re-invoke foambench_main.py for a single (dataset, case). Returns exit code."""
    wm = os.environ.get("WM_PROJECT_DIR")
    if not wm:
        print("ERROR: WM_PROJECT_DIR not set — source OpenFOAM before rerunning.", file=sys.stderr)
        return 2

    model_config_path = (exp_root / "model_config.json").resolve()
    if not model_config_path.exists():
        print(f"ERROR: {model_config_path} missing — cannot rerun.", file=sys.stderr)
        return 2

    dataset, case_id_n = case["dataset"], case["case"]
    prompt_path = (repo_root / dataset_root / dataset / case_id_n / "usr_requirement.txt").resolve()
    run_dir = Path(case["run_case_dir"]).resolve()
    out_txt = Path(case["output_path"]).resolve()
    dataset_log = (out_txt.parent / "dataset.jsonl").resolve()

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    case_id = f"Basic/{dataset}/{case_id_n}"
    cmd = [
        "python", "foambench_main.py",
        "--openfoam_path", wm,
        "--output", str(run_dir),
        "--prompt_path", str(prompt_path),
        "--dataset_log_path", str(dataset_log),
        "--case_id", case_id,
        "--model_config_path", str(model_config_path),
    ]
    print(f"  retry [{dataset}/{case_id_n}] -> {out_txt}")
    with open(out_txt, "w") as f:
        try:
            return subprocess.run(cmd, cwd=str(repo_root), stdout=f, stderr=f, check=False).returncode
        except Exception as e:
            print(f"  exception: {e}", file=sys.stderr)
            return 1


def format_rerun_command(repo_root: Path, exp_root: Path) -> str:
    rel_exp = os.path.relpath(exp_root, repo_root)
    return (
        f"python {Path(__file__).name} "
        f"--repo_root {shlex.quote(str(repo_root))} "
        f"--exp_root {shlex.quote(rel_exp)} "
        f"--rerun_agent_failures"
    )


def per_dataset_breakdown(cases: List[Dict]) -> Dict[str, Dict]:
    by_ds: Dict[str, Dict] = {}
    for c in cases:
        d = by_ds.setdefault(c["dataset"], {
            "total": 0, "success": 0, "tokens": 0,
            "cached_prompt_tokens": 0, "engine_prompt_tokens": 0,
        })
        d["total"] += 1
        d["success"] += int(c["success"])
        d["tokens"] += c["total_tokens"]
        d["cached_prompt_tokens"] += c["cached_prompt_tokens"]
        d["engine_prompt_tokens"] += c["engine_prompt_tokens"]
    return by_ds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parent,
                        help="Foam-Agent repo root (default: directory of this script)")
    parser.add_argument("--exp_root", type=str, default="experiment-finetuned/gpt-oss-20b-finetune",
                        help="Experiment dir (relative to repo_root or absolute) with runs/ and results/")
    parser.add_argument("--save_json", type=Path, default=None,
                        help="Optional path to write detailed JSON report")
    parser.add_argument("--show_failed", action="store_true", default=True,
                        help="List failed cases in the output (default: on)")
    parser.add_argument("--no_show_failed", dest="show_failed", action="store_false")
    parser.add_argument("--rerun_agent_failures", action="store_true",
                        help="Re-run early-exit cases (crashed before hitting max_loop). "
                             "Cases that exhausted max_loop are NOT rerun.")
    parser.add_argument("--max_loop", type=int, default=DEFAULT_MAX_LOOP,
                        help=f"Router retry cap used during the run (default: {DEFAULT_MAX_LOOP}, "
                             "mirrors src/config.py Config.max_loop). A failed case is "
                             "classified 'max-loop-reached' if it hit this cap, else 'early-exit'.")
    parser.add_argument("--dataset_root", default=DIR_BASIC,
                        help="Directory (relative to repo_root) holding <dataset>/<case>/usr_requirement.txt")
    args = parser.parse_args()

    exp_path = Path(args.exp_root)
    if not exp_path.is_absolute():
        exp_path = args.repo_root / exp_path
    exp_path = exp_path.resolve()

    summary, cases = summarize(exp_path, max_loop=args.max_loop)
    by_ds = per_dataset_breakdown(cases)

    print("=" * 60)
    print(f"Benchmark Summary: {exp_path}")
    print("=" * 60)
    print(f"  Total cases:    {summary['total_cases']}")
    print(f"  Successes:      {summary['successes']}")
    print(f"  Success rate:   {summary['success_rate']:.2%}")
    print(f"  Total tokens:   {summary['total_tokens']:,}")
    print(f"  Avg tokens:     {summary['average_tokens']:,.0f}")

    # vLLM engine / prefix-cache accounting. Skip the block if no case
    # reported it (e.g. a pure Bedrock/Anthropic/OpenAI run).
    if summary["total_engine_prompt_tokens"] > 0:
        print(f"  Engine prompt tokens:     {summary['total_engine_prompt_tokens']:,}")
        print(f"  Engine completion tokens: {summary['total_engine_completion_tokens']:,}")
        print(f"  Cached prompt tokens:     {summary['total_cached_prompt_tokens']:,}")
        print(f"  Cache hit rate:           {summary['cache_hit_rate']:.2%}")

    if by_ds:
        print("\nPer-dataset:")
        name_w = max(len(d) for d in by_ds)
        for ds in sorted(by_ds):
            v = by_ds[ds]
            rate = v["success"] / v["total"] if v["total"] else 0.0
            cache_str = ""
            if v["engine_prompt_tokens"] > 0:
                hit = v["cached_prompt_tokens"] / v["engine_prompt_tokens"]
                cache_str = (
                    f"  cached={v['cached_prompt_tokens']:,}/"
                    f"{v['engine_prompt_tokens']:,} ({hit:.0%})"
                )
            print(f"  {ds:<{name_w}}  {v['success']}/{v['total']}  ({rate:.0%})  "
                  f"tokens={v['tokens']:,}{cache_str}")

    if args.show_failed:
        failed = [c for c in cases if not c["success"]]
        if failed:
            print(f"\nFailed cases ({len(failed)}):")
            for c in failed:
                log_flag = "log" if c["has_solver_log"] else "no-log"
                print(f"  - {c['dataset']}/{c['case']:<3s}  [{c['failure_reason']}]  "
                      f"last_loop={c['last_loop']}  {log_flag}  {c['output_path']}")

    if args.save_json:
        payload = {"summary": summary, "per_dataset": by_ds, "cases": cases}
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {args.save_json}")

    agent_failed = agent_failed_cases(cases)
    exhausted = exhausted_cases(cases)

    if agent_failed or exhausted:
        print(f"\nFailure breakdown (max_loop={args.max_loop}):")
        print(f"  Early-exit       (crashed before max_loop, rerun OK): {len(agent_failed)}")
        print(f"  Max-loop-reached (agent exhausted retries, skip):     {len(exhausted)}")

    if agent_failed and not args.rerun_agent_failures:
        print(f"\nTo re-run just the {len(agent_failed)} early-exit case(s), copy-paste:\n")
        print(f"  {format_rerun_command(args.repo_root, exp_path)}\n")

    if args.rerun_agent_failures:
        if not agent_failed:
            print("\nNo early-exit cases to re-run.")
            return
        print(f"\nRe-running {len(agent_failed)} early-exit case(s)...")
        failures = []
        for c in agent_failed:
            rc = rerun_case(c, exp_path, args.repo_root, args.dataset_root)
            if rc != 0:
                failures.append((c["dataset"], c["case"], rc))
        print(f"\nDone. {len(agent_failed) - len(failures)}/{len(agent_failed)} rerun processes exited 0.")
        if failures:
            print("Non-zero exits:")
            for ds, cs, rc in failures:
                print(f"  - {ds}/{cs}  exit={rc}")


if __name__ == "__main__":
    main()
