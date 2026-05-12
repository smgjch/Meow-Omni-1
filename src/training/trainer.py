#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trainer.py — MeowOmni1 Custom Trainer

Extends HuggingFace Trainer to:
  1. Support DeepSpeed ZeRO-3 parameter gathering for safe state-dict saving.
  2. Override `compute_loss` to pull `labels` from the batch (since
     MeowOmni1ForCausalLM routes through Qwen3's standard CausalLM head
     which accepts `labels` natively).
"""

import os
from typing import Dict, Optional, Union

import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

try:
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    import deepspeed
    _DEEPSPEED_AVAILABLE = True
except ImportError:
    _DEEPSPEED_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility: collect ZeRO-3-sharded parameters
# ---------------------------------------------------------------------------

def _maybe_zero_3(param, ignore_status=False, name=None):
    if _DEEPSPEED_AVAILABLE and hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(f"[Trainer] Param {name} not available under ZeRO-3")
        with deepspeed.zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_model_state_dict(model, state_dict=None, dtype=None):
    """
    Collect state dict, handling both ZeRO-3 sharded params and plain params.
    """
    if state_dict is None:
        state_dict = model.state_dict()

    to_return = {}
    for k, v in state_dict.items():
        to_return[k] = _maybe_zero_3(v, name=k)
        if dtype is not None:
            to_return[k] = to_return[k].to(dtype)
    return to_return


# ---------------------------------------------------------------------------
# MeowTrainer
# ---------------------------------------------------------------------------

class MeowTrainer(Trainer):
    """
    Thin wrapper around HuggingFace Trainer tailored for MeowOmni1.

    Key differences from the base Trainer:
      - `compute_loss` correctly passes `labels` through to the model.
      - `save_model` handles ZeRO-3 parameter gathering.
    """

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Forward pass.  MeowOmni1ForCausalLM delegates to Qwen3ForCausalLM,
        which computes cross-entropy when `labels` is present in kwargs.
        """
        labels = inputs.pop("labels", None)

        outputs = model(**inputs, labels=labels)

        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Model saving (ZeRO-3 safe)
    # ------------------------------------------------------------------
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            # Gather and save weights
            state_dict = get_model_state_dict(self.model)
            if self.args.local_rank in (-1, 0):
                self.model.config.save_pretrained(output_dir)
                # Save the processor / tokenizer if available
                if hasattr(self, "processing_class") and self.processing_class is not None:
                    self.processing_class.save_pretrained(output_dir)
                elif hasattr(self, "tokenizer") and self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)

                torch.save(
                    state_dict,
                    os.path.join(output_dir, "pytorch_model.bin"),
                )
                print(f"[MeowTrainer] Model saved to {output_dir}")
