"""Rebuild the trained-record manifest: apply the training scripts' length filter
(chat-templated system+user+assistant length <= 16,384 tokens, rows with an empty
prompt or response skipped) with the tokenizer saved next to an adapter.

usage: python build_manifest.py <tokenizer_dir> <train.jsonl> <out.jsonl>
"""
import hashlib
import json
import sys
from collections import Counter

from transformers import AutoTokenizer

tok_dir, train_path, out_path = sys.argv[1:4]
MAX_LENGTH = 16384
tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

kept = dropped = skipped = 0
by_site = Counter()
drop_site = Counter()
types = Counter()
with open(train_path) as f, open(out_path, "w") as out:
    for i, line in enumerate(f):
        row = json.loads(line)
        sp = (row.get("system_prompt") or "").strip()
        up = (row.get("user_prompt") or "").strip()
        at = (row.get("response") or "").strip()
        site = row.get("substep") or row.get("step")
        rec_hash = hashlib.sha256(line.rstrip("\n").encode("utf-8")).hexdigest()
        if not up or not at:
            skipped += 1
            status, n = "skipped_empty", None
        else:
            msgs = ([{"role": "system", "content": sp}] if sp else []) + [
                {"role": "user", "content": up},
                {"role": "assistant", "content": at},
            ]
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            types[type(ids).__name__] += 1
            if hasattr(ids, "keys"):  # newer transformers return a BatchEncoding
                ids = ids["input_ids"]
            n = len(ids)
            if n <= MAX_LENGTH:
                kept += 1
                by_site[site] += 1
                status = "kept"
            else:
                dropped += 1
                drop_site[site] += 1
                status = "dropped_over_length"
        out.write(json.dumps({"line": i, "sha256": rec_hash, "case_id": row.get("case_id"),
                              "call_site": site, "loop_iteration": row.get("loop_iteration"),
                              "tokens": n, "status": status}) + "\n")

print(json.dumps({"tokenizer": tok_dir, "rows": i + 1, "kept": kept, "dropped": dropped,
                  "skipped_empty": skipped, "dropped_by_site": drop_site,
                  "apply_chat_template_return_types": types}, default=dict))
