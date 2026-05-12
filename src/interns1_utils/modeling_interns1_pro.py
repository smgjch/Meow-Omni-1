# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from .modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.utils.generic import OutputRecorder, check_model_inputs
from .configuration_interns1_pro import InternS1ProConfig, InternS1ProTextConfig, InternS1ProVisionConfig, InternS1ProTimeSeriesConfig 
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers import WhisperPreTrainedModel
import math

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3VLMoeTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3VLMoeTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class InternS1ProMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        # self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        if hasattr(config, "router_n_groups") and config.router_n_groups > 1:
            self.gate = InternS1ProMoeTextGroupedRouter(config)
        else:
            self.gate = Qwen3VLMoeTextTopKRouter(config)
        self.experts = nn.ModuleList(
            [Qwen3VLMoeTextMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        # since all the models use norm_topk_prob, we don't need to have a extra check for it
        # self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        router_logits, routing_weights, router_indices = self.gate(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        final_hidden_states = torch.zeros(
            (hidden_states.shape[0], self.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            expert_layer = self.experts[expert_idx]
            current_state = hidden_states[top_x]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, -1, self.hidden_size)
        return final_hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_sep(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors with separate head handling for FoPE.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    num_groups = int(q.shape[unsqueeze_dim] // cos.shape[unsqueeze_dim])
    cos_rep = repeat_kv(cos, num_groups)
    sin_rep = repeat_kv(sin, num_groups)
    q_embed = (q * cos_rep) + (rotate_half(q) * sin_rep)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InternS1ProMoeTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternS1ProTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3VLMoeTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLMoeTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

        # Check if FoPE is enabled
        self.use_fope = False
        self.fope_sep_head = False
        if config.rope_scaling is not None:
            self.use_fope = (
                config.rope_scaling.get("fope_init_factor", None) is not None
                or config.rope_scaling.get("fope_sep_head", None) is not None
                or config.rope_scaling.get("num_inv_freq", None) is not None
            )
            self.fope_sep_head = config.rope_scaling.get("fope_sep_head", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.use_fope and self.fope_sep_head:
            query_states, key_states = apply_rotary_pos_emb_sep(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3VLMoeTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3VLMoeTextRouter(nn.Module, ABC):
    """
    Abstract base class for MoE routers.

    Why this class exists:
      - We have multiple router implementations selected by config (e.g. TopK vs Grouped routing).
      - HuggingFace `PreTrainedModel._can_record_outputs` / `OutputRecorder` expects a *single* target module class
        to match when recording intermediate outputs (it does not reliably support passing a tuple/union of classes).
      - Defining a shared base class lets us set:
            OutputRecorder(Qwen3VLMoeTextRouter, layer_name="mlp.gate", index=0)
        so both `Qwen3VLMoeTextTopKRouter` and `Qwen3VLMoeTextGroupedRouter` are recordable through the same hook,
        while still keeping the implementation-specific routing logic in subclasses.
    """
    def __init__(self, config):
        super().__init__()
        
    @abstractmethod
    def forward(self, hidden_states):
        pass


class Qwen3VLMoeTextTopKRouter(Qwen3VLMoeTextRouter):
    def __init__(self, config):
        super().__init__(config)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        routing_weights = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class InternS1ProMoeTextGroupedRouter(Qwen3VLMoeTextRouter):
    def __init__(self, config):
        super().__init__(config)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

        self.router_n_groups = config.router_n_groups

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        routing_weights = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)

        # group-based selection
        assert self.num_experts % self.router_n_groups == 0, f"n_routed_experts must be divisible by {self.router_n_groups}"
        group_size = max(1, self.num_experts // self.router_n_groups)
        seq_len = hidden_states.shape[0]
        routing_weights = routing_weights.view(seq_len, self.router_n_groups, group_size)

        # [seq, n_groups, top_k_per_group]
        group_local_max_idx = torch.topk(routing_weights, k=self.top_k // self.router_n_groups, dim=2)[1]
        # [1, n_groups, 1]
        group_offsets = (torch.arange(self.router_n_groups, device=routing_weights.device) * group_size).view(1, -1, 1)

        router_indices = (group_local_max_idx + group_offsets).to(torch.long)  # [seq, n_groups, top_k_per_group]
        routing_weights = routing_weights.view(seq_len, self.num_experts)
        router_indices = router_indices.view(seq_len, self.top_k)  # [seq, top_k]
        router_top_value = routing_weights.gather(1, router_indices)  # [seq, n_groups]

        if self.norm_topk_prob:
            denominator = router_top_value.sum(dim=-1, keepdim=True) + 1e-20
            router_top_value = router_top_value / denominator

        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class InternS1ProMoeTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: InternS1ProTextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = InternS1ProMoeTextAttention(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = InternS1ProMoeTextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3VLMoeTextMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class InternS1ProPreTrainedModel(PreTrainedModel):
    config: InternS1ProConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternS1ProMoeTextDecoderLayer", "Qwen3VLMoeVisionBlock",'WhisperEncoderLayer']
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3VLMoeTextRouter, layer_name="mlp.gate", index=0),
        "hidden_states": InternS1ProMoeTextDecoderLayer,
        "attentions": InternS1ProMoeTextAttention,
    }

    def _init_weights(self, module):
        """Initialize the weights."""
        super()._init_weights(module)
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)
        if isinstance(module, InternS1ProMoeTextSparseMoeBlock):
            for expert in module.experts:
                expert.gate_proj.weight.data.normal_(mean=0.0, std=std)
                expert.up_proj.weight.data.normal_(mean=0.0, std=std)
                expert.down_proj.weight.data.normal_(mean=0.0, std=std)


class Qwen3VLMoeVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLMoeVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLMoeVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLMoeVisionPatchMerger(nn.Module):
    def __init__(self, config: InternS1ProVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3VLMoeVisionAttention(nn.Module):
    def __init__(self, config: InternS1ProVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLMoeVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLMoeVisionAttention(config=config)
        self.mlp = Qwen3VLMoeVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class InternS1ProVisionModel(InternS1ProPreTrainedModel):
    config: InternS1ProVisionConfig
    _no_split_modules = ["Qwen3VLMoeVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLMoeVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLMoeVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLMoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLMoeVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = grid_thw.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)

        return hidden_states


class Qwen3VLMoeTextRotaryEmbedding(nn.Module):
    def __init__(self, config: InternS1ProConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class InternS1ProMoeTextFourierEmbedding(Qwen3VLMoeTextRotaryEmbedding):
    def __init__(self, config: InternS1ProConfig, device=None):
        super().__init__(config=config)
        self.num_inv_freq = config.rope_scaling.get("num_inv_freq", None)
        self.fope_sep_head = config.rope_scaling.get("fope_sep_head", None)
        self.fope_init_factor = config.rope_scaling.get("fope_init_factor", None)
        if self.num_inv_freq is not None:
            assert (self.inv_freq > (2.0 * torch.pi / config.max_position_embeddings)).all() or (self.inv_freq.shape[-1] == self.num_inv_freq), "FoPE is wrongly initialized."

        self.head_dim = getattr(self.config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.input_dim = self.inv_freq.shape[-1]
        self.output_dim = self.inv_freq.shape[-1]

        if self.fope_sep_head:
            sin_coef = torch.randn(self.config.num_key_value_heads, self.input_dim, self.output_dim).to(self.inv_freq.device)
            cos_coef = torch.randn(self.config.num_key_value_heads, self.input_dim, self.output_dim).to(self.inv_freq.device)
        else:
            sin_coef = torch.randn(self.input_dim, self.output_dim).to(self.inv_freq.device)
            cos_coef = torch.randn(self.input_dim, self.output_dim).to(self.inv_freq.device)

        torch.nn.init.xavier_normal_(sin_coef, gain=self.fope_init_factor)
        torch.nn.init.xavier_normal_(cos_coef, gain=self.fope_init_factor)

        if self.input_dim == self.output_dim:
            sin_coef += torch.eye(self.input_dim, device=sin_coef.device)
            cos_coef += torch.eye(self.input_dim, device=cos_coef.device)
        else:
            sin_coef += self.get_step_eye(sin_coef)
            cos_coef += self.get_step_eye(cos_coef)

        self.register_buffer("sin_coef", sin_coef, persistent=True)
        self.register_buffer("cos_coef", cos_coef, persistent=True)

    def get_step_eye(self, _param):
        import math
        
        _step_eye = torch.zeros_like(_param)
        
        step = math.ceil(self.input_dim / self.output_dim)
        for i in range(self.output_dim):
            if i*step < self.input_dim:
                _step_eye[..., i*step, i] = 1.0
        
        return _step_eye

    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            raise NotImplementedError
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        batch_size, seq_len, hidden_size = x.shape
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            if self.fope_sep_head:
                pos_cos = freqs.cos().unsqueeze(1).expand(batch_size, self.config.num_key_value_heads, seq_len, -1)
                pos_sin = freqs.sin().unsqueeze(1).expand(batch_size, self.config.num_key_value_heads, seq_len, -1)
            else:
                pos_cos = freqs.cos()
                pos_sin = freqs.sin()

            if self.fope_sep_head:
                sin = torch.einsum("bhtD, hDd -> bhtd", pos_sin, self.sin_coef.float())
                cos = torch.einsum("bhtD, hDd -> bhtd", pos_cos, self.cos_coef.float())
            else:
                sin = torch.einsum("btD, Dd -> btd", pos_sin, self.sin_coef.float())
                cos = torch.einsum("btD, Dd -> btd", pos_cos, self.cos_coef.float())

            sin = F.pad(input=sin, pad=(0, self.head_dim // 2 - sin.size(-1)), mode="constant", value=1)
            cos = F.pad(input=cos, pad=(0, self.head_dim // 2 - cos.size(-1)), mode="constant", value=1)
            
            sin = torch.cat((sin, sin), dim=-1)
            cos = torch.cat((cos, cos), dim=-1)

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring(
    custom_intro=(
        "Text part of Qwen3VLMoe with 1D FoPE (Fourier Position Embedding)."
    )
)
class InternS1ProTextModel(InternS1ProPreTrainedModel):
    config: InternS1ProTextConfig
    _no_split_modules = ["InternS1ProMoeTextDecoderLayer"]

    def __init__(self, config: InternS1ProTextConfig):
        super().__init__(config)
        # Check if FoPE is enabled and use appropriate rotary embedding
        self.use_fope = False
        if config.rope_scaling is not None:
            self.use_fope = (
                config.rope_scaling.get("fope_init_factor", None) is not None
                or config.rope_scaling.get("fope_sep_head", None) is not None
                or config.rope_scaling.get("num_inv_freq", None) is not None
            )

        if self.use_fope:
            with torch.device("cpu"):
                self.rotary_emb = InternS1ProMoeTextFourierEmbedding(config=config)
        else:
            self.rotary_emb = Qwen3VLMoeTextRotaryEmbedding(config=config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [InternS1ProMoeTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        Args documentation for InternS1ProTextModel forward method.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            batch_size = inputs_embeds.shape[0]
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class InternS1ProTimeSeriesEncoder(WhisperPreTrainedModel):
    def __init__(self, config: InternS1ProTimeSeriesConfig):
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, self.embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, self.embed_dim)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        self.post_init()

        self.mask_type = None
        self.chunk_length = None

        self.adapt_in = nn.Linear(config.ts_adapt_in_dim, 80)
        self.adapt_out = nn.Linear(self.embed_dim, config.ts_adapt_out_dim)
        
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value
        
    def define_masktype(self, masktype, chunk_length=None):
        self.mask_type = masktype
        self.chunk_length = chunk_length

    def _make_causal_mask(self,
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        # print(mask.size())
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask
    
    def prepare_chunk_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        
        block_size = round(self.chunk_length / 4 * 2)
        matrix_size = input_shape[1]
        
        matrix = torch.ones(matrix_size, matrix_size)
        
        num_full_blocks = round(matrix_size // block_size)
        remainder = matrix_size % block_size  
        for i in range(num_full_blocks):
            row_start = i * block_size
            col_start = i * block_size
            matrix[row_start:row_start + block_size, col_start:col_start + block_size] = torch.zeros(block_size, block_size)
        
        if remainder > 0:
            last_row_start = num_full_blocks * block_size
            last_col_start = num_full_blocks * block_size
            matrix[last_row_start:last_row_start + remainder, last_col_start:last_col_start + remainder] = torch.zeros(remainder, remainder)
        
        matrix = matrix * -65504
        matrix = matrix.unsqueeze(0).unsqueeze(0).repeat(input_shape[0], 1, 1, 1)
        attention_mask = matrix.to(inputs_embeds.device)
        return attention_mask
    
    def forward(
            self,
            input_features,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # (N, T, C) -> (T, N, C) -> (N, C, T)
        input_features = input_features.permute(1, 0, 2)
        input_features = self.adapt_in(input_features)
        input_features = input_features.permute(1, 2, 0)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # (N, C, T) -> (N, C, T//2)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        # (N, C, T) -> (N, T, C)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # torch.Size([1, 100, 768])
        embed_pos = self.embed_positions.weight         # torch.Size([1500, 768])
        
        if inputs_embeds.shape[1] > embed_pos.shape[0]:
            target_len = inputs_embeds.shape[1]
            padding = [0, 0, 0, target_len-embed_pos.shape[0]]

            embed_pos = nn.functional.pad(embed_pos, pad=padding, mode='constant', value=0)
            hidden_states = inputs_embeds[:, :embed_pos.shape[0], :] + embed_pos
        else:
            hidden_states = inputs_embeds + embed_pos[:inputs_embeds.shape[1], :]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        input_shape = inputs_embeds.size()[:-1]
        past_key_values_length = 0
        attention_mask = None
        if self.mask_type == 'chunk':
            attention_mask = self.prepare_chunk_attention_mask(attention_mask, input_shape, inputs_embeds)
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (self.layer_norm(hidden_states),)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # (N, T, C) -> (T, N, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.adapt_out(hidden_states)

        # (T, N, C) -> (N, T, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return ModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class InternS1ProTimeSeriesConcatSubsampling(nn.Module):
    def __init__(self, in_channels: int, concat_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * concat_size
    
    def forward(self, ts_signals: torch.Tensor, ts_lens: torch.Tensor):
        if ts_signals.shape[1] % 2 != 0:
            ts_signals = ts_signals[:, :-1, :]
        even_frames = ts_signals[:, ::2, :]  
        odd_frames = ts_signals[:, 1::2, :]
        ts_signals = torch.cat((even_frames, odd_frames), dim=2)
        ts_lens = ts_lens // 2
        return ts_signals, ts_lens


class InternS1ProTimeSeriesFixPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model,dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).to(torch.float32) # (max_len, 1, d_model)
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x.clone()


class InternS1ProTimeSeriesMultiChannelAdaptiveSubsampling(nn.Module):
    def __init__(self, hidden_dim=128, nhead=8,num_encoder_layers = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.pos_encoder = InternS1ProTimeSeriesFixPositionalEncoding(d_model=hidden_dim)
        self.subsampling = InternS1ProTimeSeriesConcatSubsampling(128, 2)


    def forward(self, inputs, input_lens, sr):
        features, feature_lens = self.forward_patch(inputs, input_lens, sr)
        outputs = features
        output_lens = feature_lens
        return outputs, output_lens
    
    def forward_patch(self, inputs, input_lens, sr):
        sr = sr.float()
        strides = torch.floor(160/((1+torch.exp(-sr/100))**6))
        patch_sizes = strides * 2
        patched_outputs = []
        output_lens = []

        for i in range(len(inputs)):
            seq = inputs[i] # [seq_len, num_channel]
            ps = patch_sizes[i].item()
            st = strides[i].item()
            le = input_lens[i]
            
            output_len = torch.ceil((le - ps) / st) + 1
            pad_len = ((output_len - 1) * st + ps - le).long().item()
            if seq.ndim == 1:
                seq = seq.unsqueeze(-1)
            seq = nn.functional.pad(seq, (0, 0, 0, pad_len), "constant", 0)
            assert output_len > 0,(seq.shape, ps,st,le,output_len)
            output_lens.append(output_len)
            indices = (torch.arange(0, output_len * st, st).unsqueeze(1) + torch.arange(ps)).long()
            patched = seq[indices]

            output = self.forward_encoder(patched)    #[num_patch, D]
            patched_outputs.append(output)

        outputs = nn.utils.rnn.pad_sequence(patched_outputs, batch_first=True)
        output_lens = torch.tensor(output_lens).squeeze().to(outputs.device).long()
        if output_lens.ndim == 0:
            output_lens = output_lens.unsqueeze(0)

        outputs, output_lens = self.subsampling(outputs.clone(), output_lens.clone())
        return outputs, output_lens
    
    def forward_encoder(self, x):
        num_patch, patch_len, C = x.shape
        # conv1
        x = x.reshape(num_patch*C, 1, patch_len)    # 每个 channel 当作独立样本送入 conv1
        x = nn.functional.relu((self.conv(x))) # [B*C, D1, L]
        x = x.permute(2,0,1)    # [L, B*C, D1]

        x = self.pos_encoder(x) # [L, B*C, D1]
        x = self.transformer_encoder(x.to(torch.bfloat16))
        x = x.mean(0)

        x = x.reshape(num_patch,C,-1)

        return x.mean(1)

class InternS1ProTimeSeriesProjector(nn.Module):
    def __init__(self, config: InternS1ProTimeSeriesConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.ts_hidden_dim)
        self.linear_1 = nn.Linear(config.ts_hidden_dim, config.out_hidden_size)
        self.act = ACT2FN[config.activation_function]
        self.linear_2 = nn.Linear(config.out_hidden_size, config.out_hidden_size)

    def forward(self, ts_features):
        hidden_states = self.layer_norm(ts_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class InternS1ProTimeSeriesModel(InternS1ProPreTrainedModel):
    main_input_name = 'time_series_signals'
    _supports_flash_attn_2 = False
    config_class = InternS1ProTimeSeriesConfig
    _no_split_modules = ['WhisperEncoderLayer']

    def __init__(self, config: InternS1ProTimeSeriesConfig):
        super().__init__(config)
        self.config = config
        self.encoder_embed = InternS1ProTimeSeriesMultiChannelAdaptiveSubsampling()
        self.encoder = InternS1ProTimeSeriesEncoder(config)
        self.projector = InternS1ProTimeSeriesProjector(config)

    def get_input_embeddings(self):
        return self.encoder_embed
    
    def make_pad_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
        lengths:
            A 1-D tensor containing sentence lengths.
        max_len:
            The length of masks.
        Returns:
        Return a 2-D bool tensor, where masked positions
        are filled with `True` and non-masked positions are
        filled with `False`.

        >>> lengths = torch.tensor([1, 3, 2, 5])
        >>> make_pad_mask(lengths)
        tensor([[False,  True,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False, False, False]])
        """
        assert lengths.ndim == 1, lengths.ndim
        max_len = lengths.max()
        n = lengths.size(0)
        seq_range = torch.arange(0, max_len, device=lengths.device)
        expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
        return expaned_lengths >= lengths.unsqueeze(-1)

    def forward(
            self,
            time_series_signals: Optional[torch.FloatTensor] = None,
            ts_lens: Optional[torch.Tensor] = None,
            sr: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            time_series_embeds: Optional[torch.FloatTensor] = None,
    ):
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if time_series_signals is None and time_series_embeds is None:
            raise ValueError('You have to specify time_series_signals or time_series_embeds')

        if time_series_embeds is not None and len(time_series_embeds.shape) == 3 and time_series_embeds.shape[-1] == self.config.ts_adapt_in_dim:
            time_series_embeds = time_series_embeds
        else:
            if (isinstance(time_series_signals,list) and len(time_series_signals[0].shape) == 2) \
                or (isinstance(time_series_signals, torch.Tensor) and  len(time_series_signals.shape) == 3):
                time_series_embeds, ts_lens = self.encoder_embed(time_series_signals, ts_lens, sr)
            else:
                raise ValueError(f'wrong time_series_signals size: {time_series_signals[0].shape}')
            
        # [B, 64000, 1] -> [B, 200, 256] -> [B, 100, 1024]
        encoder_outputs = self.encoder(
            input_features=time_series_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # ts_lens after encoder
        ts_lens = (ts_lens+1) // 2
        assert torch.all(ts_lens > 0), f"The length of time_series_embeds is so small. ts_lens: {ts_lens}"
        
        src_key_padding_mask = self.make_pad_mask(ts_lens)
        last_hidden_state = encoder_outputs.last_hidden_state

        ts_pad_mask = src_key_padding_mask
        ts_embeds = self.projector(last_hidden_state)

        return ts_embeds,ts_pad_mask
    

@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen3VLMoe causal language model (or autoregressive) outputs.
    """
)
class Qwen3VLMoeCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    aux_loss: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class Qwen3VLMoeModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

        Raw router logits (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
        loss for Mixture of Experts models.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None


@auto_docstring
class InternS1ProModel(InternS1ProPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: InternS1ProConfig
    _no_split_modules = ["InternS1ProMoeTextDecoderLayer", "Qwen3VLMoeVisionBlock",'WhisperEncoderLayer']

    def __init__(self, config):
        super().__init__(config)
        self.visual = InternS1ProVisionModel._from_config(config.vision_config)
        self.language_model = InternS1ProTextModel._from_config(config.text_config)
        self.time_series = InternS1ProTimeSeriesModel._from_config(config.ts_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds
    
    def get_ts_feature(self, ts_values, ts_lens, sr):
        ts_embeds, ts_pad_mask = self.time_series(
            time_series_signals=ts_values,
            ts_lens=ts_lens,
            sr=sr,
            output_hidden_states=False,
            return_dict=True)
        return ts_embeds, ts_pad_mask

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ts_values: Union[torch.FloatTensor, list[torch.FloatTensor]] = None,
        ts_lens: Union[torch.Tensor, list[torch.Tensor]] = None,
        ts_sr: Union[torch.FloatTensor, list[torch.FloatTensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLMoeModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        ts_values (`torch.FloatTensor` of shape `(batch_size, seq_len, num_channels)`, *optional*):
            The tensors corresponding to the input time series signals.
        ts_lens (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            The valid lengths of each time series signal in the batch.
        ts_sr (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            The sampling rates of each time series signal in the batch.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if pixel_values is None and pixel_values_videos is None and ts_values is not None:
            ts_features, ts_pad_mask = self.get_ts_feature(ts_values, ts_lens, ts_sr)  # [B, T, C], [B, T]
            ts_features = ts_features[~ts_pad_mask].to(inputs_embeds.device, inputs_embeds.dtype)   # [num_valid_ts_tokens, C]
            B, N, C = inputs_embeds.shape
            input_ids = input_ids.reshape(B * N)
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            # replace ts_token in inputs_embeds and attention_mask
            ts_placeholder = (input_ids == self.config.ts_token_id)
            n_ts_placeholders = ts_placeholder.sum().item()
            n_ts_tokens = ts_features.size(0)
            assert n_ts_placeholders == n_ts_tokens, f"[ERROR]: Mismatch: <TS_CONTEXT> tokens={n_ts_placeholders}, ts_embeds_valid={n_ts_tokens}"

            try:
                inputs_embeds[ts_placeholder] = inputs_embeds[ts_placeholder] * 0.0 + ts_features
            except Exception as e:
                print(f'warning: {e}, inputs_embeds[selected].shape={inputs_embeds[ts_placeholder].shape}, ts_embeds_valid.shape={ts_features.shape}')
                inputs_embeds[ts_placeholder] = inputs_embeds[ts_placeholder] * 0.0 + n_ts_tokens[:n_ts_placeholders]

            inputs_embeds = inputs_embeds.reshape(B, N, C)
            # input_ids = input_ids.reshape(B, N)

        if position_ids is None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            if cache_position is not None:
                position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return Qwen3VLMoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
        )


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class InternS1ProForConditionalGeneration(InternS1ProPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: InternS1ProConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = InternS1ProModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_ts_feature(self, ts_values, ts_lens, sr):
        return self.model.get_ts_feature(ts_values, ts_lens, sr)
    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual
    
    def time_series(self):
        return self.model.time_series

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        ts_values: Optional[Union[torch.FloatTensor, list[torch.FloatTensor]]] = None,
        ts_lens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        ts_sr: Optional[Union[torch.FloatTensor, list[torch.FloatTensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLMoeCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

        >>> model = Qwen3VLMoeForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image in short."},
                ],
            }
        ]

        >>> # Preparation for inference
        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        >>> inputs = inputs.to(model.device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=128)
        >>> generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        >>> processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "A woman in a plaid shirt sits on a sandy beach at sunset, smiling as she gives a high-five to a yellow Labrador Retriever wearing a harness. The ocean waves roll in the background."
        ```"""

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            ts_values=ts_values,
            ts_lens=ts_lens,
            ts_sr=ts_sr,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        return Qwen3VLMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        ts_values=None,
        ts_lens=None,
        ts_sr=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            ts_values=ts_values,
            ts_lens=ts_lens,
            ts_sr=ts_sr,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["ts_values"] = None
            model_inputs["ts_lens"] = None
            model_inputs["ts_sr"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

__all__ = [
    "InternS1ProVisionModel",
    "InternS1ProTimeSeriesModel",
    "InternS1ProForConditionalGeneration",
    "InternS1ProModel",
    "InternS1ProPreTrainedModel",
]
