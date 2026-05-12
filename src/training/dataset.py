#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from typing import Dict, List, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from decord import VideoReader, cpu

IGNORE_INDEX = -100

class MeowOmni1Dataset(Dataset):
    def __init__(self, data_path: str, processor, max_length: int = 2048):
        super().__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_video_frames(video_path: str, max_frames: int = 32) -> List[Image.Image]:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames <= 0:
                return []
            
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
            
            frames = vr.get_batch(frame_indices).asnumpy()
            
            return [Image.fromarray(f).convert("RGB") for f in frames]
            
        except Exception as e:
            print(f"Warning: Failed to load video {video_path}: {e}")
            return []
        
    @staticmethod
    def _build_prompt(conversations: List[Dict]) -> str:
        parts = []
        for turn in conversations:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n".join(parts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        conversations = sample.get("conversations", [])

        # =========================================================
        # STEP 1: Collect all modality data and track which user
        # turn each modality belongs to (always first user turn
        # for this dataset schema)
        # =========================================================
        images = []
        audios_list = []
        ts_paths = None
        ts_srs = None

        # Placeholders to inject, keyed by conversation turn index
        # Each value is a list of placeholder strings to prepend in order
        placeholders_per_turn: Dict[int, List[str]] = {}

        def _first_user_turn_idx():
            for i, turn in enumerate(conversations):
                if turn["role"] == "user":
                    return i
            return 0  # fallback

        first_user_idx = _first_user_turn_idx()

        # --- Images ---
        if sample.get("image_path"):
            img = Image.open(sample["image_path"]).convert("RGB")
            images.append(img)
            placeholders_per_turn.setdefault(first_user_idx, []).append(
                "<image>./</image>"
            )

        # --- Video frames (each frame = one <image> placeholder) ---
        if sample.get("video_path"):
            video_frames = self._load_video_frames(sample["video_path"], max_frames=20)
            if video_frames:
                images.extend(video_frames)
                # One placeholder per frame, concatenated into a single block
                frame_placeholders = "".join(["<image>./</image>"] * len(video_frames))
                placeholders_per_turn.setdefault(first_user_idx, []).append(
                    frame_placeholders
                )

        # --- Audio ---
        # MiniCPM-o expects: one <audio>./</audio> per audio segment in text,
        # and the raw numpy array in the audios list, in matching order.
        if sample.get("audio_path"):
            try:
                import soundfile as sf
                audio_arr, sr = sf.read(sample["audio_path"])
                audio_arr = audio_arr.astype(np.float32)
                target_sr = 16000
                max_samples = 30 * target_sr  # hard cap at 30s
                if len(audio_arr) > max_samples:
                    audio_arr = audio_arr[:max_samples]
                audios_list.append(audio_arr)
                placeholders_per_turn.setdefault(first_user_idx, []).append(
                    "<audio>./</audio>"
                )
            except Exception as e:
                print(f"Warning: Failed to load audio {sample['audio_path']}: {e}")

        # --- Time-Series ---
        # Meow Omni 1 processor looks for the sentinel block
        # "<|ts_start|><|ts_unit|><|ts_end|>" in the text and expands
        # <|ts_unit|> to the correct number of tokens.
        if sample.get("ts_path"):
            ts_paths = [sample["ts_path"]]
            ts_srs = [float(sample.get("ts_sr", 100.0))]
            placeholders_per_turn.setdefault(first_user_idx, []).append(
                "<|ts_start|><|ts_unit|><|ts_end|>"
            )

        # =========================================================
        # STEP 2: Inject placeholders into a deep-copied conversation
        # Prepend all placeholders for that turn, each on its own
        # line, before the user's actual text content — matching
        # MiniCPM-o's "\n".join(cur_msgs) convention in chat().
        # =========================================================
        conversations_with_placeholders = [dict(t) for t in conversations]
        for turn_idx, placeholder_list in placeholders_per_turn.items():
            original_content = conversations_with_placeholders[turn_idx]["content"]
            # Join all placeholders for this turn, then append the original text
            injected = "\n".join(placeholder_list + [original_content])
            conversations_with_placeholders[turn_idx]["content"] = injected

        text = self._build_prompt(conversations_with_placeholders)

        # =========================================================
        # STEP 3: Call processor with all modalities
        # =========================================================
        proc_kwargs: Dict[str, Any] = dict(
            text=text,
            max_length=self.max_length,
            truncation=True
        )
        if images:
            proc_kwargs["images"] = images
        if audios_list:
            proc_kwargs["audios"] = audios_list
        if ts_paths is not None:
            proc_kwargs["time_series_paths"] = ts_paths
            proc_kwargs["time_series_sampling_rates"] = ts_srs

        encoding = self.processor(**proc_kwargs, return_tensors="pt")

        # =========================================================
        # STEP 4: Strip batch dim
        # =========================================================
        item: Dict[str, Any] = {}
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
                item[k] = v.squeeze(0)
            elif isinstance(v, list) and len(v) == 1 and k != "time_series_values":
                item[k] = v[0]
            else:
                item[k] = v

        # =========================================================
        # STEP 5: Build labels
        # =========================================================
        input_ids = item["input_ids"]
        labels = input_ids.clone()

        assistant_token_ids = self.processor.tokenizer.encode(
            "Assistant:", add_special_tokens=False
        )
        seq = input_ids.tolist()
        mask_end = 0
        for i in range(len(seq) - len(assistant_token_ids)):
            if seq[i : i + len(assistant_token_ids)] == assistant_token_ids:
                mask_end = i + len(assistant_token_ids)
        labels[:mask_end] = IGNORE_INDEX
        item["labels"] = labels

        return item
   
# Collator
def meow_data_collator(
    features: List[Dict[str, Any]],
    max_length: int = 2048,
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    

    for f in features:
        for k in ["pixel_values", "image_sizes", "image_bound", "tgt_sizes", "audio_bounds", "spk_bounds"]:
            if k not in f or f[k] is None or f[k] == [[]] or f[k] == []:
                f[k] = []
                
        if "audio_feature_lens" not in f or f["audio_feature_lens"] is None:
            f["audio_feature_lens"] = []
            
    batch: Dict[str, Any] = {}
    all_keys = set()
    for f in features: 
        all_keys.update(f.keys())

    for key in all_keys:
        # 1. 3D Tensor Padding
        if key == "time_series_values":
            ts_list = [f[key] for f in features if key in f and isinstance(f[key], torch.Tensor)]
            if len(ts_list) > 0:
                batch[key] = pad_sequence(ts_list, batch_first=True, padding_value=0.0)
            continue
            
        if key == "audio_features":
            af_list = []
            for f in features:
                feat = f.get(key)
                if feat is not None and isinstance(feat, torch.Tensor):
                    af_list.append(feat.squeeze(0).permute(1, 0))

            if len(af_list) > 0:
                padded_audio = pad_sequence(af_list, batch_first=True, padding_value=0.0)
                batch[key] = padded_audio.permute(0, 2, 1)
            else:
                batch[key] = []
            continue

        if key == "audio_feature_lens":
            lens_list = []
            for f in features:
                val = f.get(key)
                
                if isinstance(val, (list, torch.Tensor)) and len(val) > 0:
                    l = val[0]
                elif isinstance(val, (int, float)):
                    l = val
                else:
                    l = 0                
                lens_list.append(torch.tensor([int(l)], dtype=torch.long))
            
            batch[key] = lens_list
            continue

        values = [f[key] for f in features if key in f]
        
        if key in ["pixel_values", "image_sizes", "image_bound", "tgt_sizes", "audio_feature_lens", "audio_bounds", "spk_bounds"]:
            batch[key] = values
            continue

        if not isinstance(values[0], torch.Tensor):
            try:
                batch[key] = torch.stack(values)
            except Exception:
                batch[key] = values
            continue

        # 1D text Tensor (input_ids, labels) Padding
        if values[0].dim() == 1:
            max_len = min(max(v.shape[0] for v in values), max_length)
            pad_id = IGNORE_INDEX if key == "labels" else pad_token_id
            padded = torch.full((len(values), max_len), fill_value=pad_id, dtype=values[0].dtype)
            for i, v in enumerate(values):
                length = min(v.shape[0], max_len)
                padded[i, :length] = v[:length]
            batch[key] = padded
            
        elif values[0].dim() >= 2:
            try:
                batch[key] = torch.stack(values)
            except RuntimeError:
                batch[key] = values
        else:
            batch[key] = torch.stack(values)

    native_data_keys = [
        "input_ids", "attention_mask", "pixel_values", "image_sizes", 
        "image_bound", "tgt_sizes", "audio_features", "audio_feature_lens", 
        "audio_bounds", "spk_bounds"
    ]
    
    data_dict = {}
    for k in list(batch.keys()):
        if k in native_data_keys:
            data_dict[k] = batch.pop(k)
            
    batch["data"] = data_dict

    return batch