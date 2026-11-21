set -x

MODEL_PATH=${1:-Qwen/Qwen3-4B-Instruct-2507}
PORT=${PORT:-18080}
NUM_GPUS=${NUM_GPUS:-4}

vllm serve $MODEL_PATH \
    --served-model-name skill-writer \
    --port $PORT \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --data-parallel-size $NUM_GPUS \
    --max-model-len 32768 \
    --enable-chunked-prefill
