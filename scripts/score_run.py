#!/usr/bin/env python3
"""Score one run the way the paper's results were scored.

  success      End rule: some log.*Foam in the case's run dir has "End" as its second-to-last line
               (summarize_benchmark.solver_ended_cleanly)
  failure      max-loop-reached (used all 10 review loops) or early-exit (the agent stopped earlier: JSON failure,
               exception, or a call stopped by the per-call wall guard)
  clean        success without nan/inf in the solver logs or final fields (only differs from success on platforms
               where OpenFOAM cannot trap floating-point exceptions, e.g. aarch64)
  NMSE         nmse_basic / nmse_advanced against GT_Files, 9999 when not a success;
               NMSE score per case = 1 (< 0.1), 0.5 (< 0.3) or 0; the run's NMSE score is the mean over cases
Writes <exp_root>/<label>/score.json and score.csv; prints a summary.
Usage: scripts/score_run.py <RUN_LABEL> [--advanced]
"""
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
os.chdir(CODE)
sys.path.insert(0, CODE)
os.environ.setdefault("FOAMAGENT_CASES_BASIC", "8 9 10")
import benchmark_utils as bu  # noqa: E402
import summarize_benchmark as sb  # noqa: E402

NANINF = re.compile(r"(?<![A-Za-z])-?(nan|inf)(?![A-Za-z])", re.I)


def nan_inf(run_dir):
    for lg in glob.glob(os.path.join(run_dir, "log.*Foam")):
        if NANINF.search(open(lg, errors="ignore").read()):
            return True
    if not os.path.isdir(run_dir):
        return False
    times = [t for t in os.listdir(run_dir)
             if re.fullmatch(r"[0-9.eE+-]+", t) and os.path.isdir(os.path.join(run_dir, t))]
    if times:
        tmax = max(times, key=float)
        if tmax != "0":
            for f in glob.glob(os.path.join(run_dir, tmax, "*")):
                if os.path.isfile(f) and os.path.getsize(f) < 50_000_000 and NANINF.search(open(f, errors="ignore").read()):
                    return True
    return False


def main():
    label = sys.argv[1]
    advanced = "--advanced" in sys.argv
    if advanced:
        import nmse_advanced as nm
        root = os.path.join("experiment-finetuned-advanced", label)
        cases = [(d, None) for d in bu.DATASETS_ADVANCED]
    else:
        import nmse_basic as nm
        root = os.path.join("experiment-finetuned", label)
        cases = [(d, str(c)) for d in bu.DATASETS_BASIC for c in bu.CASES_BASIC]
    rows = []
    for fam, var in cases:
        parts = [fam] + ([var] if var else [])
        run = os.path.join(root, "runs", *parts)
        out = os.path.join(root, "results", *parts, "output.txt")
        gt = os.path.join(bu.DIR_ADVANCED if advanced else bu.DIR_BASIC, *parts, "GT_Files")
        text = open(out, errors="ignore").read() if os.path.exists(out) else ""
        success = sb.solver_ended_cleanly(Path(run))
        status = "not started" if not text else ("complete" if any(m in text for m in (
            "Workflow completed successfully!", "Workflow finished after reaching the maximum review loop limit."))
            else "aborted")
        failure = None if success else (sb.failure_reason(text, sb.parse_last_loop(text), sb.DEFAULT_MAX_LOOP)
                                        if text else "not started")
        if success and nm.success(run) and os.path.exists(gt):
            try:
                val = float(nm.evaluate_nmse(gt, run))
            except Exception:
                val = 9999.0
        else:
            val = 9999.0
        rows.append({"case": "/".join(parts), "success": int(success), "clean": int(success and not nan_inf(run)),
                     "status": status, "failure": failure or "", "last_loop": sb.parse_last_loop(text),
                     "wall_guard": int("[wall-guard]" in text), "nmse": val, "nmse_score": nm.score_nmse(val)})
    n = len(rows)
    summary = {
        "label": label, "cases": n, "successes": sum(r["success"] for r in rows),
        "clean": sum(r["clean"] for r in rows),
        "max_loop": sum(r["failure"] == "max-loop-reached" for r in rows),
        "early_exit": sum(r["failure"] == "early-exit" for r in rows),
        "wall_guard": sum(r["wall_guard"] for r in rows),
        "not_started": sum(r["status"] == "not started" for r in rows),
        "nmse_score": round(sum(r["nmse_score"] for r in rows) / n, 3) if n else None,
    }
    json.dump({"summary": summary, "cases": rows}, open(os.path.join(root, "score.json"), "w"), indent=1)
    with open(os.path.join(root, "score.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=1))
    print(f"per case: {os.path.join(CODE, root, 'score.csv')}")


if __name__ == "__main__":
    main()
