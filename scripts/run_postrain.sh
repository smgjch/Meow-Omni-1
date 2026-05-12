#!/bin/bash
# ==============================================================================
# run_posttrain.sh — Stage 2: Multi-Modal Superivsed Finetuning
#
# Only LLM is trainable.
# ==============================================================================

torchrun --nproc_per_node=8 ./src/training/train.py \
    --model_name_or_path "PATH/TO/YOUR/PROJECTOR/PRETRAINED/CHECKPOINT" \
    --data_path "PATH/TO/YOUR/DATA" \
    --output_dir "output/" \
    --remove_unused_columns False \
    --model_max_length 4500 \
    --stage "2" \
    --dataloader_num_workers 4 \
    --use_lora False \
    --tune_ts False \
    --tune_vision False \
    --tune_audio False \
    --tune_llm True \
    --bf16 True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10

echo "Stage 2 complete."
