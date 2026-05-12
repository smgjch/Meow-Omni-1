#!/usr/bin/env bash
# ==============================================================================
# run_pretrain.sh — Stage 1: Projector-only pretraining
#
# Only model.tspm.projector is trainable.
# Everything else (LLM, vision, audio, TS encoder) is frozen.
# ==============================================================================

set -e

MODEL_PATH="/path/to/meow-omni-1-base"   # <-- edit
DATA_PATH="/path/to/pretrain_data.json"  # <-- edit
OUTPUT_DIR="./checkpoints/stage1_projector_pretrain"

NUM_GPUS=${NUM_GPUS:-8}

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=29500 \
  finetune_meow_omni.py \
    --stage pretrain \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --init_vision True \
    --init_audio True \
    --init_tts False \
    --init_time_series True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --model_max_length 2048 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --deepspeed ds_config_pretrain.json

echo "Stage 1 complete. Checkpoint saved to $OUTPUT_DIR"
