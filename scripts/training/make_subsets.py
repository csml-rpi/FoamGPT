#!/usr/bin/env python3
"""Build the nested teacher-data subsets for the scaling experiment.

Unit of sampling = one teacher rollout = one case_id (all its rows go together).
Algorithm (seed recorded in subsets/selection.json):
  1. rng = random.Random(SEED)
  2. for each family (sorted by name): shuffle its 7 variants  -> per-family order
  3. for round r = 1..7: shuffle the family order (fresh shuffle every round) and
     append each family's r-th variant                          -> 77-long sequence
  4. level L takes the first N_L rollouts: 25%=19, 50%=39, 75%=58, 100%=77.
Levels are nested by construction, and round 1 alone contains every family, so every
family appears at every level.

Each level's train.jsonl keeps the original rows (byte-identical lines) in their
original file order. The 100% file must be byte-identical to the source train.jsonl.
Row counts here are BEFORE the length filter; after-filter counts are produced by the
training script's own build_combined_dataset() in the training env (bin/filter_check.py).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

SEED = 20260928
LEVELS = {"L025": 19, "L050": 39, "L075": 58, "L100": 77}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBSTEPS = ["parse_case_info", "build_advice", "decompose_subtasks", "generate_file",
            "allrun_commands", "allrun_script", "error_analysis", "rewrite_plan",
            "rewrite_files"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build the nested scaling subsets.")
    parser.add_argument("--train", default=os.path.join(ROOT, "data", "train.jsonl"),
                        help="Training JSONL to sample from (unpacked data/train.jsonl.gz)")
    parser.add_argument("--output", default=os.path.join(ROOT, "data", "scaling", "subsets"),
                        help="Output folder; must not exist yet")
    args = parser.parse_args()
    src, out = args.train, args.output
    if os.path.exists(out):
        sys.exit(f"{out} already exists — refusing to overwrite")
    if not os.path.exists(src):
        sys.exit(f"{src} not found; unpack it first: gunzip -k data/train.jsonl.gz")
    with open(src, "rb") as f:
        raw_lines = [l for l in f.read().split(b"\n") if l.strip()]
    rows = [json.loads(l) for l in raw_lines]
    assert len(rows) == 2647, len(rows)

    fams = sorted({r["case_id"].split("/")[1] for r in rows})
    variants = {f: sorted({int(r["case_id"].split("/")[2]) for r in rows
                           if r["case_id"].split("/")[1] == f}) for f in fams}
    assert len(fams) == 11 and all(v == list(range(1, 8)) for v in variants.values())

    rng = random.Random(SEED)
    var_order = {}
    for f in fams:
        v = list(range(1, 8))
        rng.shuffle(v)
        var_order[f] = v
    sequence, rounds = [], []
    for r in range(7):
        order = list(fams)
        rng.shuffle(order)
        rounds.append(order)
        sequence += [f"Basic/{f}/{var_order[f][r]}" for f in order]
    assert len(sequence) == 77 and len(set(sequence)) == 77

    os.makedirs(out)
    summary = {"seed": SEED, "source": os.path.relpath(src, ROOT), "source_sha256": sha256(src),
               "algorithm": __doc__.strip().splitlines()[2:10],
               "per_family_variant_order": var_order,
               "round_family_order": rounds, "sequence": sequence, "levels": {}}
    for name, n in LEVELS.items():
        ids = sequence[:n]
        idset = set(ids)
        d = os.path.join(out, name)
        os.makedirs(d)
        keep = [i for i, r in enumerate(rows) if r["case_id"] in idset]
        with open(os.path.join(d, "case_ids.txt"), "w") as f:
            f.write("\n".join(ids) + "\n")
        with open(os.path.join(d, "train.jsonl"), "wb") as f:
            f.write(b"".join(raw_lines[i] + b"\n" for i in keep))
        sub = collections.Counter(rows[i]["substep"] for i in keep)
        fam = collections.Counter(rows[i]["case_id"].split("/")[1] for i in keep)
        summary["levels"][name] = {
            "rollouts": n,
            "case_ids": ids,
            "rows_before_filter": len(keep),
            "rows_before_filter_by_substep": {s: sub.get(s, 0) for s in SUBSTEPS},
            "rollouts_by_family": dict(sorted(collections.Counter(
                c.split("/")[1] for c in ids).items())),
            "rows_by_family": dict(sorted(fam.items())),
            "train_jsonl_sha256": sha256(os.path.join(d, "train.jsonl")),
        }
        assert set(summary["levels"][name]["rollouts_by_family"]) == set(fams), name
    lv = summary["levels"]
    for a, b in (("L025", "L050"), ("L050", "L075"), ("L075", "L100")):
        assert set(lv[a]["case_ids"]) < set(lv[b]["case_ids"])
    assert lv["L100"]["train_jsonl_sha256"] == summary["source_sha256"], "100% file not identical"
    with open(os.path.join(out, "selection.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for name in LEVELS:
        print(name, lv[name]["rollouts"], "rollouts,", lv[name]["rows_before_filter"],
              "rows (before filter)", lv[name]["rows_before_filter_by_substep"])
    print("100% file byte-identical to source:", lv["L100"]["train_jsonl_sha256"] == summary["source_sha256"])


if __name__ == "__main__":
    main()
