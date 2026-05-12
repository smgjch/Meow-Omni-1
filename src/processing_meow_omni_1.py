#!/usr/bin/env python
from typing import Union, Optional, List
import os
import torch
import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import logging
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from src.minicpm_utils.processing_minicpmo import MiniCPMOProcessor

logger = logging.get_logger(__name__)

class MeowOmni1Processor(MiniCPMOProcessor):
    """
    Constructs a MeowOmni1 processor which wraps text, vision, audio, and time-series.
    """
    attributes = ["image_processor", "audio_processor", "tokenizer"]
    
    image_processor_class = "AutoImageProcessor"
    audio_processor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, audio_processor=None, tokenizer=None, **kwargs):
        
        super().__init__(
            image_processor=image_processor, 
            audio_processor=audio_processor, 
            tokenizer=tokenizer, 
            **kwargs
        )
        
        # Define the Meow Omni 1 Time-Series Tokens
        self.ts_start_token = "<|ts_start|>"
        self.ts_unit_token = "<|ts_unit|>"
        self.ts_end_token = "<|ts_end|>"
        
    # ==============================================================================
    # GRAFTED FROM INTERN-S1 PRO
    # ==============================================================================
    def _get_num_ts_tokens(self, sampling_rates, ts_lens):
        strides = np.floor(160 / ((1 + np.exp(-sampling_rates / 100))**6))
        patch_sizes = strides * 2
        embed_lengths = (np.ceil((ts_lens - patch_sizes) / strides) + 1).astype(np.int64)
        num_ts_tokens = [(embed_length // 2 + 1) // 2 for embed_length in embed_lengths]
        return num_ts_tokens

    def time_series_processor(self, ts_paths: list[str], sampling_rates: list[float], do_normalize=True, do_truncate=True):
        assert len(ts_paths) == len(sampling_rates), "ts_paths and sampling_rates must have the same length"

        time_series_values = []
        sr_list = []
        ts_lens_list = []

        for idx, ts_path in enumerate(ts_paths):
            sr = sampling_rates[idx]
            ext = os.path.splitext(ts_path)[-1].lower()
            
            if ext in [".wav", '.mp3', '.flac']:
                try:
                    import soundfile as sf
                except ImportError:
                    raise ImportError("Please install soundfile to process audio files.")
                ts_input, sr = sf.read(ts_path) 
            elif ext == ".csv":
                pd = __import__("pandas")
                df = pd.read_csv(ts_path, header=None)
                ts_input = df.values  
            elif ext == ".npy":
                ts_input = np.load(ts_path) 
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            if not isinstance(ts_input, np.ndarray):
                ts_input = np.array(ts_input, dtype=np.float32)

            if do_normalize:
                mean = ts_input.mean(axis=0, keepdims=True)   
                std = ts_input.std(axis=0, keepdims=True)  
                ts_input = (ts_input - mean) / (std + 1e-8)

            if do_truncate and len(ts_input) > 240000:
                ts_input = ts_input[:240000] 
            
            if ts_input.ndim == 1:
                ts_input = ts_input[:, None] 

            ts_len = ts_input.shape[0]

            if sr is None or sr == 0:   
                sr = ts_len / 4
            
            time_series_values.append(torch.tensor(ts_input, dtype=torch.float32))
            sr_list.append(sr)
            ts_lens_list.append(ts_len)
        
        ts_lens_arr = np.array(ts_lens_list)
        ts_sr_arr = np.array(sr_list)
        
        num_ts_tokens = self._get_num_ts_tokens(sampling_rates=ts_sr_arr, ts_lens=ts_lens_arr)
        
        return {
            "time_series_values": time_series_values, # List of Tensors
            "sr": torch.tensor(ts_sr_arr, dtype=torch.float32), 
            "ts_lens": torch.tensor(ts_lens_arr, dtype=torch.long),
            "num_ts_tokens": num_ts_tokens
        }


    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        time_series_paths: Optional[List[str]] = None,
        time_series_sampling_rates: Optional[List[float]] = None,
        images=None,
        audios=None,
        audio_parts=None,
        **kwargs,
    ) -> BatchFeature:
        
        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        ts_data_dict = {}

        # 1. Process Time-Series & Inject TS Placeholders First
        if time_series_paths is not None:
            if time_series_sampling_rates is None:
                raise ValueError("time_series_sampling_rates must be provided if time_series_paths are used.")
            
            ts_kwargs = kwargs.pop("time_series_kwargs", {"do_normalize": True, "do_truncate": True})
            
            ts_data = self.time_series_processor(
                ts_paths=time_series_paths, 
                sampling_rates=time_series_sampling_rates,
                **ts_kwargs
            )
            
            # Stack the tensors for the batch
            ts_data_dict["time_series_values"] = torch.stack(ts_data["time_series_values"])
            ts_data_dict["ts_lens"] = ts_data["ts_lens"]
            ts_data_dict["sr"] = ts_data["sr"]
            num_ts_tokens = ts_data["num_ts_tokens"]
            
            target_block = f"{self.ts_start_token}{self.ts_unit_token}{self.ts_end_token}"
            for i in range(len(text)):
                if target_block in text[i]:
                    ts_placeholder = self.ts_start_token + (self.ts_unit_token * num_ts_tokens[i]) + self.ts_end_token
                    text[i] = text[i].replace(target_block, ts_placeholder, 1)

        # 2. Delegate Vision, Audio, and Text Tokenization to MiniCPM-o
        batch_features = super().__call__(
            text=text,
            images=images,
            audios=audios,
            audio_parts=audio_parts,
            **kwargs
        )

        # 3. Merge Time-Series Tensors into the Final Batch
        if time_series_paths is not None:
            batch_features.data.update(ts_data_dict)
            
        return batch_features