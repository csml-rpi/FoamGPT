#!/bin/bash
# Download the base models at the exact revisions used on the second cluster, plus the retrieval embedder.
# Usage: scripts/fetch_models.sh <MODELS_DIR> <name> [<name> ...]
#   names: nano super qwen38 dsv4 dsv41 embedder   (or: all)
# Needs huggingface_hub (the `hf` CLI; installed with either requirements file).
# The embedder goes to the HF cache ($HF_HOME); the harness loads it by name, so run the benchmark with the same HF_HOME.
set -euo pipefail
DEST="${1:?usage: fetch_models.sh <MODELS_DIR> <name> [...]}"; shift
[ $# -gt 0 ] || { echo "names: nano super qwen38 dsv4 dsv41 embedder all" >&2; exit 1; }
[ "$1" = all ] && set -- nano super qwen38 dsv4 dsv41 embedder
get() {  # repo revision folder
  echo "== $1@$2 -> $DEST/$3"
  hf download "$1" --revision "$2" --local-dir "$DEST/$3"
}
for n in "$@"; do
  case $n in
    nano)  get nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 cbd3fa9f933d55ef16a84236559f4ee2a0526848 nemotron-3-nano-30b-a3b-bf16 ;;   # ~60 GB
    super) get nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 2dc98e2afe4face0e4ce40972a915c45368bd34a nemotron-3-super-120b-a12b-bf16 ;;
    qwen38) get Qwen/Qwen3.8-Flash-Next de4b8e4d43b917e7706784d8bb445c9af86a3540 qwen3.8-flash-next-bf16 ;;
    dsv4)  get deepseek-ai/DeepSeek-V4-Flash-0731 7872f01b1d1fe23eabc4c98b48bffcef5a386062 deepseek-v4-flash-0731 ;;
    dsv41) get deepseek-ai/DeepSeek-V4.1-Flash 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 deepseek-v4.1-flash ;;
    embedder) echo "== Qwen/Qwen3-Embedding-0.6B -> HF cache (${HF_HOME:-~/.cache/huggingface})"
              hf download Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 ;;
    *) echo "unknown name $n" >&2; exit 1 ;;
  esac
done
