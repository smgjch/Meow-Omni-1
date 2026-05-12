import torch
import json
import os
import re
import sys
import argparse
import math
from collections import Counter
import numpy as np
import soundfile as sf
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.modeling_meow_omni_1 import MeowOmni1ForCausalLM
from src.processing_meow_omni_1 import MeowOmni1Processor

class MeowOmni1Evaluator:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"Loading Processor & Model from {model_path}...")
        self.processor = MeowOmni1Processor.from_pretrained(model_path, trust_remote_code=True)
        self.model = MeowOmni1ForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        
        if device == "cuda":
            print(f"🔍 Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    def _to_device(self, data):
        """Recursively moves tensors to the correct device, handling lists/dicts."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_device(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._to_device(v) for v in data)
        return data

    def _load_video_frames(self, video_path, max_frames=16):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            if total_frames <= 0: return []
            
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
            
            frames = vr.get_batch(frame_indices).asnumpy()
            return [Image.fromarray(f).convert("RGB") for f in frames]
        except Exception as e:
            print(f"Video Load Fail: {video_path} - {e}")
            return []

    def parse_options_from_prompt(self, prompt):
        options = {}
        pattern = r'([A-D])[.\):]?\s*([^\n]+)'
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        for letter, text in matches:
            options[letter.upper()] = text.strip()
        return options

    def normalize_text(self, text):
        if not text: return ""
        text = text.lower()
        text = re.sub(r'[_\-\.\s]+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def extract_answer(self, text, options=None):
        if not text: return None, 'none'
        text_upper = text.upper().strip()
        match = re.search(r'\b([A-D])\b', text_upper)
        if match: return match.group(1), 'letter'
        
        if options and len(options) > 0:
            norm_response = self.normalize_text(text)
            matched_letters = []
            for letter, content in options.items():
                norm_content = self.normalize_text(content)
                if norm_content and norm_content in norm_response:
                    matched_letters.append(letter)
            
            if len(matched_letters) == 1:
                return matched_letters[0], 'content'
        return None, 'none'

    def check_correctness(self, pred_letter, gt_letter, raw_text, options):
        if pred_letter == gt_letter: return True, "Letter Match"
        if options and gt_letter in options:
            norm_response = self.normalize_text(raw_text)
            gt_content = self.normalize_text(options[gt_letter])
            contains_gt = gt_content and gt_content in norm_response
            contains_wrong = False
            for letter, content in options.items():
                if letter != gt_letter:
                    wrong_content = self.normalize_text(content)
                    if wrong_content and wrong_content in norm_response:
                        contains_wrong = True
                        break
            if contains_gt and not contains_wrong:
                return True, "Content Match"
        return False, "Mismatch"

    def _prepare_inputs(self, item, active_modals):
        """Helper to load all modalities and build the prompt for both eval modes."""
        raw_prompt = item['conversations'][0]['content']
        images, audios, ts_paths, ts_srs, placeholders = [], None, None, None, []

        if active_modals is None or 'video' in active_modals:
            if item.get("image_path") and os.path.exists(item["image_path"]):
                try:
                    images.append(Image.open(item["image_path"]).convert("RGB"))
                    placeholders.append("<image>./</image>")
                except: pass
                
            if item.get("video_path") and os.path.exists(item["video_path"]):
                frames = self._load_video_frames(item["video_path"], max_frames=16)
                if frames:
                    images.extend(frames)
                    frame_placeholders = "".join(["<image>./</image>"] * len(frames))
                    placeholders.append(frame_placeholders)
                    
        images = images if images else None

        if active_modals is None or 'audio' in active_modals:
            path = item.get("audio_path")
            if path and os.path.exists(path):
                try:
                    audio_arr, sr = sf.read(path)
                    audio_arr = audio_arr.astype(np.float32)
                    audios = [audio_arr[:480000]]
                    placeholders.append("<audio>./</audio>")
                except: pass

        if active_modals is None or 'ts' in active_modals:
            path = item.get("ts_path")
            if path and os.path.exists(path):
                ts_paths = [path]
                ts_srs = [float(item.get("ts_sr", 100.0))]
                placeholders.append("<|ts_start|><|ts_unit|><|ts_end|>")

        injected_prompt = "\n".join(placeholders + [raw_prompt])
        prompt = f"User: {injected_prompt}\nAssistant:"
        
        return prompt, images, audios, ts_paths, ts_srs, raw_prompt

    def _robust_decode(self, output):
        """Safely decodes model output into a string."""
        try:
            if hasattr(output, "sequences"):
                return self.processor.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
            elif isinstance(output, torch.Tensor):
                return self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
            elif isinstance(output, str):
                return output
            elif isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], str): return output[0]
                elif isinstance(output[0], torch.Tensor):
                    return self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
                return str(output[0])
            elif isinstance(output, tuple):
                first_elem = output[0]
                if isinstance(first_elem, str): return first_elem
                elif isinstance(first_elem, torch.Tensor):
                    ids_to_decode = first_elem[0] if first_elem.ndim > 1 else first_elem
                    return self.processor.tokenizer.decode(ids_to_decode, skip_special_tokens=True)
                return str(first_elem)
            return str(output)
        except Exception as e:
            print(f"\nDecoding warning: {e}. Coercing output to string.")
            return str(output)

    # 1. STANDARD BENCHMARK (ACCURACY & ABLATION)
    @torch.no_grad()
    def predict(self, item, active_modals=None):
        prompt, images, audios, ts_paths, ts_srs, raw_prompt = self._prepare_inputs(item, active_modals)
        options = self.parse_options_from_prompt(raw_prompt)

        encoding = self.processor(
            text=[prompt], # Wrapped in list to fix Vec error
            images=images, audios=audios,
            time_series_paths=ts_paths, time_series_sampling_rates=ts_srs,
            ids=[item.get("id", "unknown")], # Fix tracking errors
            return_tensors="pt"
        )

        encoding = dict(encoding.data) if hasattr(encoding, "data") else dict(encoding.items())
        encoding = self._to_device(encoding) # Recursive fix for BFloat16 error
        batch_size = encoding["input_ids"].shape[0]

        if "image_bound" not in encoding or encoding["image_bound"] is None:
            encoding["image_bound"] = torch.zeros(batch_size, 0, 2, dtype=torch.long, device=self.device)
        if "audio_feature_lens" not in encoding or not encoding["audio_feature_lens"]:
            encoding["audio_feature_lens"] = [torch.tensor([0], device=self.device) for _ in range(batch_size)]
        if "audio_bounds" not in encoding or not encoding["audio_bounds"]:
            encoding["audio_bounds"] = [[] for _ in range(batch_size)]

        gen_kwargs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding.get("attention_mask"),
            "tokenizer": self.processor.tokenizer,
            "max_new_tokens": 128,
            "do_sample": False, # Greedy search for standard eval
        }
        for k in ["pixel_values", "tgt_sizes", "image_bound", "audio_features", "audio_feature_lens", "audio_bounds"]:
            if k in encoding: gen_kwargs[k] = encoding[k]

        output = self.model.generate(**gen_kwargs)
        raw_text = self._robust_decode(output)
        
        pred_letter, match_type = self.extract_answer(raw_text, options)
        return pred_letter, raw_text, match_type, options

    def run_benchmark(self, benchmark_path, output_file, active_modals=None):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        results = []
        processed_ids = set()
        
        # --- RESUME MECHANISM ---
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    results = existing_data.get("details", [])
                    processed_ids = {item["id"] for item in results}
                print(f"Resuming session. Found {len(processed_ids)} completed evaluations.")
            except Exception as e: pass

        mode_label = "+".join(active_modals).upper() if active_modals else "ALL"
        print(f"Evaluating {len(samples)} samples [Modals: {mode_label}]...")
        
        for i, item in enumerate(tqdm(samples)):
            item_id = item.get("id", f"missing_id_{i}")
            if item_id in processed_ids: continue
                
            conv = item["conversations"][1]
            gt = conv.get("content", conv.get("value", "")).strip().upper()
            
            try:
                pred, raw_text, match_type, options = self.predict(item, active_modals=active_modals)
                is_correct, reason = self.check_correctness(pred, gt, raw_text, options)
            except Exception as e:
                pred, raw_text, is_correct, reason = None, f"Error: {str(e)}", False, "Execution Error"
            
            results.append({
                "id": item_id, "gt": gt, "pred": pred,
                "raw_response": raw_text, "correct": is_correct, "reason": reason
            })
            processed_ids.add(item_id)
            
            # --- AUTO-SAVE ---
            correct_count = sum(1 for r in results if r["correct"])
            current_acc = correct_count / len(results) if results else 0.0
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "config": {"mode": "benchmark", "modals_active": active_modals if active_modals else "all"},
                    "accuracy": current_acc, "details": results
                }, f, indent=2, ensure_ascii=False)

        final_correct = sum(1 for r in results if r["correct"])
        final_acc = final_correct / len(results) if results else 0.0
        print(f"\nFinal Accuracy: {final_acc:.2%} ({final_correct}/{len(results)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--modals", type=str, nargs='*', help="Modals to keep active (e.g., video audio ts). Empty implies all.")
    
    args = parser.parse_args()
    
    evaluator = MeowOmni1Evaluator(args.model_path, args.device)
    
    evaluator.run_benchmark(
        benchmark_path=args.data_path,
        output_file=args.output_file,
        active_modals=args.modals
    )
   