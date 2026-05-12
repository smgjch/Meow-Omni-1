#!/bin/bash

MODEL_PATH="smgjch/Meow-Omni-1"
DATA_PATH="./data/benchmark_omni_synthesis.json"
EVAL_SCRIPT="./src/evaluation/eval_meow.py"
OUTPUT_DIR="./results/eval_results/"
MODALS="video audio ts"
DEVICE="cuda"

mkdir -p "$OUTPUT_DIR"

# Default output filename
OUTPUT_FILE="${OUTPUT_DIR}/evaluation_results.json"

echo "Starting evaluation of $MODEL_PATH..."
CUDA_VISIBLE_DEVICES=0 python "$EVAL_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_file "$OUTPUT_FILE" \
    --device "$DEVICE" \
    --modals $MODALS

echo "✅ Evaluation completed. Results saved to $OUTPUT_FILE"