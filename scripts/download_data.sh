#!/bin/bash
# Download what is not stored in git, from the FoamGPT dataset on Hugging Face:
#   traces      teacher traces and the training / held-out splits   -> data/
#   references  reference solutions of the 49 evaluation cases      -> dataset/**/GT_Files (scoring only)
#   faiss       retrieval indexes the agent searches                 -> code/database/faiss
# Usage: scripts/download_data.sh [traces] [references] [faiss]      (no argument: all three)
#        FOAMGPT_HF_DATA=<owner>/<name> scripts/download_data.sh ...   (another copy of the dataset)
#        scripts/download_data.sh --from <dir> [...]                   (install from local copies of the files)
# Needs the `hf` CLI, which both requirements files in scripts/ install.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${FOAMGPT_HF_DATA:-ghzlmc/foamgpt}"
SRC=""
if [ "${1:-}" = "--from" ]; then SRC="$(cd "${2:?--from needs a directory}" && pwd)"; shift 2; fi
[ $# -gt 0 ] || set -- traces references faiss
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

fetch() {  # print the local path of one file of the dataset
  if [ -n "$SRC" ]; then
    echo "$SRC/$1"
  else
    hf download "$REPO" "$1" --repo-type dataset --local-dir "$TMP" >/dev/null
    echo "$TMP/$1"
  fi
}

for item in "$@"; do
  case $item in
    traces)
      for f in teacher_traces.jsonl.gz train.jsonl.gz test.jsonl.gz; do
        cp "$(fetch "data/$f")" "$ROOT/data/$f"
      done
      echo "traces: data/teacher_traces.jsonl.gz, data/train.jsonl.gz, data/test.jsonl.gz" ;;
    references)
      tar -xzf "$(fetch references/foambench_references.tar.gz)" -C "$ROOT"
      echo "references: $(find "$ROOT/dataset" -type d -name GT_Files | wc -l | tr -d ' ') GT_Files folders in dataset/" ;;
    faiss)
      tar -xzf "$(fetch faiss/faiss_qwen3-embedding-0.6b.tar.gz)" -C "$ROOT"
      echo "faiss: code/database/faiss/Qwen_Qwen3-Embedding-0.6B" ;;
    *)
      echo "unknown item: $item (use traces, references, faiss)" >&2; exit 1 ;;
  esac
done
