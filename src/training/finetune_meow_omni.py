#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finetune_meow_omni.py — Two-Stage Fine-Tuning for MeowOmni1

Usage
-----
# Stage 1 — Projector-only pretraining (freeze everything except tspm.projector)
python finetune_meow_omni.py \
    --stage pretrain \
    --model_name_or_path /path/to/meow-omni-1-base \
    --data_path /path/to/train.json \
    --output_dir ./checkpoints/stage1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --bf16 True

# Stage 2 — Full visual SFT (vision + resampler + LLM backbone unfrozen)
python finetune_meow_omni.py \
    --stage sft \
    --model_name_or_path ./checkpoints/stage1 \
    --data_path /path/to/sft_train.json \
    --output_dir ./checkpoints/stage2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --bf16 True \
    --use_lora True

Architecture note
-----------------
  model.llm         → Qwen3ForCausalLM backbone
  model.vpm         → SiglipVisionTransformer
  model.resampler   → Perceiver resampler (vision → LLM dim)
  model.apm         → MiniCPMWhisperEncoder  (audio encoder)
  model.audio_*     → audio pooler + projector
  model.tts         → MiniCPMTTS
  model.tspm        → InternS1ProTimeSeriesModel
                        ├── encoder  (frozen in stage 1)
                        └── projector (trained in stage 1)
"""

import logging
import os
from dataclasses import dataclass, field
from functools import partial
from types import MethodType
from typing import List, Optional, Tuple

import torch
import transformers
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer
from transformers.integrations import deepspeed as hf_deepspeed

from dataset import MeowOmni1Dataset, meow_data_collator
from trainer import MeowTrainer

# ── local model imports (adjust to your package layout) ─────────────────────
from modeling_meow_omni_1 import MeowOmni1ForCausalLM
from processing_meow_omni_1 import MeowOmni1Processor

logger = logging.getLogger(__name__)


# ============================================================================
# Argument dataclasses
# ============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to a pretrained MeowOmni1 checkpoint."}
    )
    # Modality init flags — defaults match the test suite
    init_vision: bool = field(default=True)
    init_audio: bool = field(default=True)
    init_tts: bool = field(default=False)
    init_time_series: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to training JSON."})
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to evaluation JSON (optional)."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # ── stage selection ──────────────────────────────────────────────────────
    stage: str = field(
        default="sft",
        metadata={
            "help": (
                "Training stage. Choose from:\n"
                "  pretrain — freeze everything except tspm.projector\n"
                "  sft      — full visual SFT (vision + LLM backbone unfrozen)"
            )
        },
    )
    # ── per-module tune flags (used in sft stage) ────────────────────────────
    tune_llm: bool = field(default=True, metadata={"help": "Unfreeze LLM backbone."})
    tune_vision: bool = field(default=True, metadata={"help": "Unfreeze vpm + resampler."})
    tune_audio: bool = field(default=False, metadata={"help": "Unfreeze audio encoder."})
    tune_time_series_encoder: bool = field(
        default=False,
        metadata={"help": "Unfreeze TS encoder (always True in pretrain stage)."},
    )
    # ── LoRA ─────────────────────────────────────────────────────────────────
    use_lora: bool = field(default=False)
    # ── misc ─────────────────────────────────────────────────────────────────
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=2048)
    optim: str = field(default="adamw_torch")


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # Targets every self-attention projection inside the Qwen3 backbone
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layers_to_transform: Optional[List[int]] = None


# ============================================================================
# Parameter-freezing logic
# ============================================================================

def _freeze_all(model: MeowOmni1ForCausalLM) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def apply_stage1_pretrain_freeze(model: MeowOmni1ForCausalLM) -> None:
    """
    Stage 1 — Projector-only pretraining.

    Freeze everything; then unfreeze only the TS projector.
    The TS encoder backbone stays frozen so that the pretrained
    time-series representations are not corrupted.

    Target: model.tspm.projector  (the MLP inside InternS1ProTimeSeriesModel)
    """
    _freeze_all(model)

    # InternS1ProTimeSeriesModel exposes its projector as `.projector`.
    # If the attribute name is different in your version, adjust here.
    ts_proj = None
    if hasattr(model, "tspm"):
        if hasattr(model.tspm, "projector"):
            ts_proj = model.tspm.projector
        elif hasattr(model.tspm, "ts_projector"):
            ts_proj = model.tspm.ts_projector

    if ts_proj is None:
        raise AttributeError(
            "Cannot locate the TS projector inside model.tspm. "
            "Expected model.tspm.projector or model.tspm.ts_projector. "
            "Please update apply_stage1_pretrain_freeze()."
        )

    for p in ts_proj.parameters():
        p.requires_grad_(True)

    _log_trainable(model)


def apply_stage2_sft_freeze(
    model: MeowOmni1ForCausalLM,
    tune_llm: bool,
    tune_vision: bool,
    tune_audio: bool,
    tune_time_series_encoder: bool,
) -> None:
    """
    Stage 2 — Full visual SFT.

    Selectively unfreeze modules according to tune_* flags.
    The TS projector is always trainable in stage 2.
    """
    _freeze_all(model)

    # LLM backbone
    if tune_llm and hasattr(model, "llm"):
        model.llm.requires_grad_(True)

    # Vision encoder + resampler
    if tune_vision:
        if hasattr(model, "vpm"):
            model.vpm.requires_grad_(True)
        if hasattr(model, "resampler"):
            model.resampler.requires_grad_(True)

    # Audio encoder
    if tune_audio:
        if hasattr(model, "apm"):
            model.apm.requires_grad_(True)
        if hasattr(model, "audio_avg_pooler"):
            model.audio_avg_pooler.requires_grad_(True)
        if hasattr(model, "audio_projection_layer"):
            model.audio_projection_layer.requires_grad_(True)

    # TS encoder (full)
    if tune_time_series_encoder and hasattr(model, "tspm"):
        model.tspm.requires_grad_(True)
    elif hasattr(model, "tspm"):
        # Always keep the projector trainable
        proj = getattr(model.tspm, "projector", None) or getattr(
            model.tspm, "ts_projector", None
        )
        if proj is not None:
            for p in proj.parameters():
                p.requires_grad_(True)

    _log_trainable(model)


def _log_trainable(model: MeowOmni1ForCausalLM) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )


# ============================================================================
# LoRA setup (stage 2 only)
# ============================================================================

def apply_lora(
    model: MeowOmni1ForCausalLM,
    lora_args: LoraArguments,
    tune_vision: bool,
) -> MeowOmni1ForCausalLM:
    """
    Wrap the model with PEFT LoRA adapters on the LLM attention projections.
    Embedding and resampler are always saved as full modules.
    """
    # Freeze LLM params first — LoRA will add its own trainable deltas
    if hasattr(model, "llm"):
        for p in model.llm.parameters():
            p.requires_grad_(False)

    modules_to_save = ["embed_tokens", "resampler"]
    if tune_vision:
        modules_to_save.append("vpm")

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_args.lora_layers_to_transform,
        modules_to_save=modules_to_save,
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(model)

    # Inject a get_input_embeddings shim if missing (required by PEFT)
    if not hasattr(model, "get_input_embeddings"):
        def _get_input_embeddings(self):
            return self.llm.get_input_embeddings()
        model.get_input_embeddings = MethodType(_get_input_embeddings, model)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ============================================================================
# Main training function
# ============================================================================

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = (
        parser.parse_args_into_dataclasses()
    )

    # ── DeepSpeed distributed type patch ─────────────────────────────────────
    if getattr(training_args, "deepspeed", None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ── Load model ───────────────────────────────────────────────────────────
    logger.info(f"Loading model from {model_args.model_name_or_path} ...")
    model = MeowOmni1ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        init_vision=model_args.init_vision,
        init_audio=model_args.init_audio,
        init_tts=model_args.init_tts,
        init_time_series=model_args.init_time_series,
        trust_remote_code=True,
    )

    # ── Load processor + tokenizer ───────────────────────────────────────────
    processor = MeowOmni1Processor.from_pretrained(
        model_args.model_name_or_path,
        fix_mistral_regex=True,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer

    # ── Apply freeze / unfreeze strategy ────────────────────────────────────
    if training_args.stage == "pretrain":
        logger.info("=== Stage 1: Projector-only pretraining ===")
        apply_stage1_pretrain_freeze(model)

    elif training_args.stage == "sft":
        logger.info("=== Stage 2: Full visual SFT ===")

        if training_args.use_lora:
            # In LoRA mode we first apply stage-2 freezes for non-LLM modules,
            # then wrap the LLM with LoRA adapters.
            apply_stage2_sft_freeze(
                model,
                tune_llm=False,          # LoRA handles LLM trainability
                tune_vision=training_args.tune_vision,
                tune_audio=training_args.tune_audio,
                tune_time_series_encoder=training_args.tune_time_series_encoder,
            )
            model = apply_lora(model, lora_args, training_args.tune_vision)
            if training_args.gradient_checkpointing:
                model.enable_input_require_grads()
        else:
            apply_stage2_sft_freeze(
                model,
                tune_llm=training_args.tune_llm,
                tune_vision=training_args.tune_vision,
                tune_audio=training_args.tune_audio,
                tune_time_series_encoder=training_args.tune_time_series_encoder,
            )
    else:
        raise ValueError(
            f"Unknown stage '{training_args.stage}'. Choose 'pretrain' or 'sft'."
        )

    # ── Build datasets ───────────────────────────────────────────────────────
    logger.info("Loading datasets ...")
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

    collator = partial(
        meow_data_collator,
        max_length=training_args.model_max_length,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    # ── Gradient-checkpointing kwargs ────────────────────────────────────────
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = MeowTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    train()
