set -x

MODEL_PATH=${1:-Qwen/Qwen3-Embedding-8B}
PORT=${PORT:-18081}

vllm serve $MODEL_PATH \
    --served-model-name skill-embedder \
    --port $PORT \
    --runner pooling
