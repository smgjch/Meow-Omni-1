import torch
import torch.nn as nn
import os
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled


class MeowTrainer(Trainer):
    # 修复：加上 **kwargs 以接收 num_items_in_batch 等未来HF可能新增的参数
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 提取 Labels
        labels = inputs.pop("labels", None)
        
        # 2. 提取原生多模态数据字典 (包含 input_ids, image_bound, audio_bounds 等)
        data = inputs.pop("data", None)
        
        if data is None:
            raise ValueError("The 'data' dictionary is missing from inputs. Check the DataCollator.")

        # 3. 前向传播
        if not self.args.use_lora:
            # 原生模型调用
            outputs = model(data=data, **inputs, use_cache=False)
        else:
            # LoRA 模式下，通常使用 base_model 调用
            with model._enable_peft_forward_hooks(**inputs):
                outputs = model.base_model(data=data, **inputs, use_cache=False)
                
        # 4. 计算 Loss

        if labels is not None:
            # ==========================================
            # 🚨 核心逻辑：Causal LM 错位预测 (Shift)
            # 让第 t 个 token 的预测输出，对齐第 t+1 个真实的 label
            # ==========================================
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 彻底展平为 (N*SeqLen, VocabSize) 和 (N*SeqLen)
            logits = shift_logits.view(-1, shift_logits.shape[-1])
            labels = shift_labels.view(-1).long()
            
            loss_fct = nn.CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss