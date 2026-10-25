#!/bin/bash
# Download the reference solutions of the 49 evaluation cases into dataset/**/GT_Files.
# Only the field-fidelity (NMSE) scorer reads them; execution success does not need them.
# They sit in the adapter repository on Hugging Face, next to the adapters.
# Usage: scripts/download_references.sh                  (from Hugging Face)
#        scripts/download_references.sh --from <archive>  (from a local copy of foambench_references.tar.gz)
#        FOAMGPT_HF_REPO=<owner>/<name> scripts/download_references.sh
# Needs the `hf` CLI, which both requirements files in scripts/ install.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${FOAMGPT_HF_REPO:-ghzlmc/foamgpt}"
FILE="references/foambench_references.tar.gz"
if [ "${1:-}" = "--from" ]; then
  ARCHIVE="${2:?--from needs the path of foambench_references.tar.gz}"
else
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  hf download "$REPO" "$FILE" --local-dir "$TMP" >/dev/null
  ARCHIVE="$TMP/$FILE"
fi
tar -xzf "$ARCHIVE" -C "$ROOT"
echo "references: $(find "$ROOT/dataset" -type d -name GT_Files | wc -l | tr -d ' ') GT_Files folders in dataset/"
