# coding=utf-8
# Copyright 2025 Jucheng Hu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this code are adapted from modeling_interns1_pro.py,
# which is Copyright 2025 HuggingFace Inc. team, also licensed under
# the Apache License, Version 2.0.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from typing import Optional


# 1. Import MiniCPM-o Multi-modal Modules
from src.minicpm_utils.modeling_minicpmo import (
    MiniCPMOPreTrainedModel,
    MiniCPMO
)

# 2. Import InternS1-Pro Time Series Modules
from src.interns1_utils.modeling_interns1_pro import InternS1ProTimeSeriesModel

# 3. Import your unified configuration
from .configuration_meow_omni_1 import MeowOmni1Config


class MeowOmni1PreTrainedModel(MiniCPMOPreTrainedModel):
    config_class = MeowOmni1Config


class MeowOmni1ForCausalLM(MiniCPMO):
    def __init__(self, config):
        super().__init__(config)

        # ==========================================
        # 4. Time-Series Module (InternS1Pro Transplant)
        # ==========================================
        if getattr(self.config, "init_time_series", False):
            self.tspm = self.init_time_series_module()

    def init_time_series_module(self):
        if isinstance(self.config.ts_config, dict):
            from transformers import PretrainedConfig
            self.config.ts_config = PretrainedConfig(**self.config.ts_config)

        # 1. Inject the MiniCPM LLM's dimension into the TS config
        self.config.ts_config.out_hidden_size = self.embed_dim
        
        # 2. Initialize the InternS1 model 
        model = InternS1ProTimeSeriesModel(self.config.ts_config)
        return model

    def get_time_series_embeddings(self, time_series_values, ts_lens=None, sr=None):
        ts_outputs = self.tspm(time_series_values, ts_lens=ts_lens, sr=sr)
        
        if isinstance(ts_outputs, tuple):
            ts_embeddings = ts_outputs[0]
        else:
            ts_embeddings = ts_outputs.last_hidden_state if hasattr(ts_outputs, "last_hidden_state") else ts_outputs
            
        return ts_embeddings
    
    
    def forward(
        self,
        data: dict = None,
        time_series_values: Optional[torch.Tensor] = None,
        ts_lens: Optional[torch.Tensor] = None,
        sr: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        vllm_embedding = self.get_omni_embedding(
            data,
            input_embeddings=vllm_embedding,
            chunk_length=self.config.audio_chunk_length,
        )
        if "position_ids" in data:
            position_ids = data["position_ids"]
        else:
            attention_mask = data.get("attention_mask")
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
            else:
                input_ids = data["input_ids"]
                position_ids = torch.arange(
                    input_ids.shape[1], 
                    dtype=torch.long, 
                    device=input_ids.device
                ).unsqueeze(0).expand_as(input_ids)

        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()


        if time_series_values is not None and getattr(self.config, "init_time_series", False):

            ts_embeds = self.get_time_series_embeddings(time_series_values, ts_lens=ts_lens, sr=sr)
            
            if "input_ids" not in data:
                raise KeyError("The 'data' dictionary must contain 'input_ids' for time-series masking.")
                
            ts_mask = (data["input_ids"] == self.config.ts_unit_id)
            
            valid_embeds = []
            ts_idx = 0 

            for i in range(data["input_ids"].shape[0]):
                num_tokens = ts_mask[i].sum().item()
                if num_tokens > 0:

                    valid_embeds.append(ts_embeds[ts_idx, :num_tokens, :])
                    ts_idx += 1
            
            if valid_embeds:
                flat_ts_embeds = torch.cat(valid_embeds, dim=0)
                num_placeholders = ts_mask.sum().item()
                
                if num_placeholders != flat_ts_embeds.shape[0]:
                    raise ValueError(f"Mismatch: {num_placeholders} TS tokens in text != {flat_ts_embeds.shape[0]} valid TS features.")
                
                vllm_embedding[ts_mask] = flat_ts_embeds.to(vllm_embedding.dtype)
                
        return self.llm(
            input_ids= None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )