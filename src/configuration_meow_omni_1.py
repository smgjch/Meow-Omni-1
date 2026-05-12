#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Union, Dict, Any

from transformers import PretrainedConfig
from transformers.utils import logging

from src.minicpm_utils.configuration_minicpmo import MiniCPMOConfig
from src.interns1_utils.configuration_interns1_pro import InternS1ProTimeSeriesConfig

logger = logging.get_logger(__name__)

class MeowOmni1Config(MiniCPMOConfig):
    model_type = "meow_omni_1"

    def __init__(
        self,
        ts_config: Union[Dict[str, Any], InternS1ProTimeSeriesConfig, None] = None,
        init_time_series: bool = True,
        ts_start_id: int = 151748,
        ts_unit_id: int = 151749,
        ts_end_id: int = 151750,
        **kwargs
    ):
        # Initialize the base MiniCPM-o configuration (handles text, vision, audio, tts)
        super().__init__(**kwargs)
        
        if ts_config is None:
            self.ts_config = InternS1ProTimeSeriesConfig()
            logger.info("ts_config is None, using default InternS1ProTimeSeriesConfig")
        elif isinstance(ts_config, dict):
            self.ts_config = InternS1ProTimeSeriesConfig(**ts_config)
        elif isinstance(ts_config, InternS1ProTimeSeriesConfig):
            self.ts_config = ts_config
        else:
            raise ValueError(f"Unknown type for ts_config: {type(ts_config)}")

        # ==========================================
        # Meow Omni 1 Special Token IDs
        # ==========================================
        self.ts_start_id = ts_start_id
        self.ts_unit_id = ts_unit_id
        self.ts_end_id = ts_end_id
        
        # ==========================================
        # Initialization Toggles
        # ==========================================
        self.init_time_series = init_time_series
        
        # Ensure base modalities are preserved unless explicitly overridden
        self.init_vision = kwargs.get("init_vision", True)
        self.init_audio = kwargs.get("init_audio", True)
        self.init_tts = kwargs.get("init_tts", True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        """
        Custom loader to ensure sub-configs (like ts_config) are properly instantiated 
        when loading the main config from a huggingface hub directory or local path.
        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # Handle nested Time-Series config dictionary during load
        if "ts_config" in config_dict and isinstance(config_dict["ts_config"], dict):
            config_dict["ts_config"] = InternS1ProTimeSeriesConfig(**config_dict["ts_config"])

        return cls.from_dict(config_dict, **kwargs)