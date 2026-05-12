import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple
from types import MethodType

import torch
import transformers
from accelerate.utils import DistributedType

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 导入你自定义的模型和数据组件
from src.modeling_meow_omni_1 import MeowOmni1ForCausalLM
from src.processing_meow_omni_1 import MeowOmni1Processor
from dataset import MeowOmni1Dataset, meow_data_collator
from meow_trainer import MeowTrainer

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="openbmb/MiniCPM-o-4_5")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)
    
    # 核心训练阶段控制
    stage: str = field(default="2", metadata={"help": "Stage '1' for Projector Pretraining, '2' for SFT."})
    
    tune_vision: Optional[bool] = field(default=False)
    tune_audio: Optional[bool] = field(default=False)
    tune_ts: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    
    use_lora: Optional[bool] = field(default=False)

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_bias: str = "none"

local_rank = 0

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return {'Total': all_param, 'Trainable': trainable_params}

def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None): 
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    local_rank = training_args.local_rank

    # 1. 加载 Processor
    processor = MeowOmni1Processor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)


# ==========================================
    # 🚨 终极修复：双段式权重加载逻辑
    # ==========================================
    


# ==========================================
# 🚨 修正：移除 ignore_mismatched_sizes，直接加载
# ==========================================

# 确保 model_args.model_name_or_path 指向你测试成功的那个 checkpoint 路径
# 例如: .../stage1_projector_pretrain/checkpoint-5998

    print(f"Loading model from {model_args.model_name_or_path}...")

    model = MeowOmni1ForCausalLM.from_pretrained(
        model_args.model_name_or_path,  # ✅ 直接加载 checkpoint
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        init_vision=True,
        init_audio=True,
        init_time_series=True,
    )

    print("Model loaded successfully.")

    # 3. 冻结策略控制 (极度重要!)
    if training_args.stage == "1":
        if local_rank == 0: print("--- Entering Stage 1: Projector Pre-training ---")
        # Stage 1: 冻结全身上下，只开放 Time-Series Projector
        for name, param in model.named_parameters():
            if "tspm" in name:
                # 可以选择只开放 projector，或者整个 ts encoder。通常只开放 projector
                if "projector" in name or "ts_projection_layer" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = getattr(training_args, "tune_ts_encoder", False)
            else:
                param.requires_grad = False
                
    elif training_args.stage == "2":
        if local_rank == 0: print("--- Entering Stage 2: Supervised Fine-Tuning ---")
        # 冻结视觉和音频编码器
        if not training_args.tune_vision and hasattr(model, "vpm"):
            model.vpm.requires_grad_(False)
            model.resampler.requires_grad_(False)
        if not training_args.tune_audio and hasattr(model, "apm"):
            model.apm.requires_grad_(False)
            
        # 根据配置决定是否冻结时序模块 (通常在 Stage 2，特征编码器被冻结)
        if not training_args.tune_ts and hasattr(model, "tspm"):
            model.tspm.requires_grad_(False)
            
        if not training_args.tune_llm:
            model.llm.requires_grad_(False)

    # 4. LoRA 注入逻辑 (仅在 Stage 2 生效)
    if training_args.use_lora and training_args.stage == "2":
        if training_args.tune_llm:
            raise ValueError("Cannot perform Full LLM tuning and LoRA simultaneously.")
            
        if local_rank == 0: print("Applying LoRA to the LLM backend...")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
            
        # 保证我们需要训练的额外模态层的 embedding 是活跃的
        modules_to_save = ['embed_tokens']
        
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            modules_to_save=modules_to_save,
        )
        
        # 适配 MiniCPM 获取 embedding 的接口
        if not hasattr(model, 'get_input_embeddings'):
            model.get_input_embeddings = MethodType(lambda self: self.llm.get_input_embeddings(), model)
            
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    if local_rank == 0: 
        print(get_parameter_number(model))

    # 5. 构建数据集
    train_dataset = MeowOmni1Dataset(
        data_path=data_args.data_path,
        processor=processor,
        max_length=training_args.model_max_length,
    )
    
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = MeowOmni1Dataset(
            data_path=data_args.eval_data_path,
            processor=processor,
            max_length=training_args.model_max_length,
        )

    # 6. 初始化 Trainer 并开始训练
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = MeowTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(meow_data_collator, max_length=training_args.model_max_length)
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()