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

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers import WhisperConfig


class InternS1ProTextConfig(PretrainedConfig):
    model_type = "interns1_pro_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        attention_bias=False,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=1408,
        num_experts_per_tok=4,
        num_experts=60,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        rope_scaling=None,
        head_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim or hidden_size // num_attention_heads

        rope_config_validation(self, ignore_keys={"fope_init_factor", "fope_sep_head", "num_inv_freq"})

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class InternS1ProVisionConfig(PretrainedConfig):
    model_type = "interns1_pro_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range

class InternS1ProTimeSeriesConfig(WhisperConfig):

    model_type = "interns1_pro_time_series"
    base_config_key = "ts_config"

    def __init__(
        self,
        ts_adapt_in_dim: int=256,
        ts_adapt_out_dim: int=1024,
        ts_hidden_dim: int=1024,
        ts_cnn_channels: list[int]=[1, 32, 64, 128, 128],      
        ts_cnn_kernel_sizes: list[int]=[3, 5, 5, 5],   
        ts_cnn_strides: list[int]=[2, 4, 4, 5],               
        ts_cnn_paddings: list[int]=[1, 2, 2, 2],       
        ts_concat_subsampling_in_channels: int=128,       
        ts_concat_subsampling_concat_size: int=2,
        use_flash_attn: bool=False,           
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ts_cnn_channels = ts_cnn_channels
        self.ts_cnn_kernel_sizes = ts_cnn_kernel_sizes
        self.ts_cnn_strides = ts_cnn_strides
        self.ts_cnn_paddings = ts_cnn_paddings
        self.ts_concat_subsampling_in_channels = ts_concat_subsampling_in_channels
        self.ts_concat_subsampling_concat_size = ts_concat_subsampling_concat_size

        self.ts_adapt_in_dim = ts_adapt_in_dim
        self.ts_adapt_out_dim = ts_adapt_out_dim

        self.ts_hidden_dim = ts_hidden_dim
        self.use_flash_attn = use_flash_attn

        assert self.ts_adapt_out_dim == self.ts_hidden_dim, "ts_adapt_out_dim should be equal to ts_hidden_dim"
        assert self.ts_concat_subsampling_in_channels == self.ts_cnn_channels[-1], "ts_concat_subsampling_in_channels should be equal to the out_channel of the last cnn layer"


class InternS1ProConfig(PretrainedConfig):
    model_type = "interns1_pro"
    sub_configs = {"vision_config": InternS1ProVisionConfig, "text_config": InternS1ProTextConfig, 'ts_config':InternS1ProTimeSeriesConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        ts_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        ts_token_id=151685,
        ts_start_id=151683,
        ts_end_id=151684,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        if isinstance(ts_config, dict):
            self.ts_config = self.sub_configs["ts_config"](**ts_config)
        elif ts_config is None:
            self.ts_config = self.sub_configs["ts_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.ts_token_id = ts_token_id
        self.ts_start_id = ts_start_id
        self.ts_end_id = ts_end_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


__all__ = ["InternS1ProConfig", "InternS1ProTextConfig", "InternS1ProVisionConfig", "InternS1ProTimeSeriesConfig"]