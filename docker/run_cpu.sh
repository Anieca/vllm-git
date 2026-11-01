#!/usr/bin/env bash
# Run GIT on the vLLM CPU image built by docker/build_cpu.sh.
#
#   docker/run_cpu.sh compare [args]  # HF vs vLLM greedy check (scripts/compare_hf_vllm.py)
#   docker/run_cpu.sh serve [args]    # OpenAI-compatible server on localhost:8000
#
# The host HF cache is mounted so weights are downloaded only once.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-compare}"
shift || true

common=(
  --rm
  --cap-add SYS_NICE
  --security-opt seccomp=unconfined
  --shm-size 2g
  -e VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-1}"
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
)

case "$mode" in
  compare)
    docker run "${common[@]}" \
      -v "$PWD/scripts:/scripts:ro" \
      --entrypoint python3 \
      vllm-git:cpu /scripts/compare_hf_vllm.py "$@"
    ;;
  serve)
    docker run "${common[@]}" \
      -p 8000:8000 \
      -v "$PWD/serving:/serving:ro" \
      vllm-git:cpu microsoft/git-base-coco \
      --dtype float32 \
      --max-model-len 512 \
      --chat-template /serving/git_chat_template.jinja \
      "$@"
    ;;
  *)
    echo "usage: $0 {compare|serve} [args...]" >&2
    exit 2
    ;;
esac
