#!/usr/bin/env bash
# Build the vLLM CPU image from the fork checkout in ./vllm (branch add-git-model),
# using upstream's docker/Dockerfile.cpu. Native arch: linux/arm64 on Apple
# silicon / Graviton, linux/amd64 elsewhere.
#
#   MAX_JOBS=4 docker/build_cpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

case "$(uname -m)" in
  arm64 | aarch64) platform=linux/arm64 ;;
  *) platform=linux/amd64 ;;
esac

docker buildx build \
  --platform "$platform" \
  -f vllm/docker/Dockerfile.cpu \
  --target vllm-openai \
  --build-arg max_jobs="${MAX_JOBS:-4}" \
  --tag vllm-git:cpu \
  --load \
  vllm
