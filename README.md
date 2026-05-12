# 🐱 Meow-Omni 1

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-2605.09152-b31b1b.svg?style=for-the-badge" alt="arXiv"/>
  <img src="https://img.shields.io/badge/🤗%20Model-smgjch%2FMeow--Omni--1-yellow?style=for-the-badge" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge" alt="Python"/>
</p>

> **The official codebase for the paper:**  
> [Meow-Omni 1: A Multimodal Large Language Model for Feline Ethology](https://arxiv.org/abs/2605.09152v1)  
> Jucheng Hu, Zhangquan Chen, Yulin Chen, Chengjie Hong, Liang Zhou, Tairan Wang, Sifei Li, Giulio Zhu, Feng Zhou, Yiheng Zeng, Suorong Yang, Dongzhan Zhou

---

## Overview

**Meow-Omni 1** is the first open-source, **quad-modal** Multimodal Large Language Model (MLLM) purpose-built for **computational ethology**, specifically, deciphering feline intent from heterogeneous sensory data.

A central challenge in animal behavioral understanding is *semantic aliasing*: the same external signal (e.g., a cat's purr) can correspond to radically different internal states depending on physiological context. Meow-Omni 1 addresses this by natively fusing four modalities:

| Modality | Description |
|---|---|
| 🎥 **Video** | Visual behavior and body language |
| 🔊 **Audio** | Vocalizations (meows, purrs, hisses) |
| 📈 **Physiological Time-Series** | Heart rate, respiration, accelerometer data |
| 💬 **Text** | Natural-language reasoning and Q&A |

Built on top of [MiniCPM-o 4.5](https://github.com/OpenBMB/MiniCPM-o) and augmented with a transplanted [InternS1-Pro](https://huggingface.co/OpenGVLab/InternVL) time-series encoder, Meow-Omni 1 achieves **71.16% intent-recognition accuracy** on **MeowBench**, which is a novel, expert-verified quad-modal benchmark. Meow Omni 1 substantially outperforming all vision-language and omni-modal baselines.

---

## 🔗 Resources

| Resource | Link |
|---|---|
| 📄 Paper | [arXiv:2605.09152](https://arxiv.org/abs/2605.09152v1) |
| 🤗 Model Weights | [smgjch/Meow-Omni-1](https://huggingface.co/smgjch/Meow-Omni-1) |
| 📦 Dataset (Meow-10K) | [smgjch/Meow-10K](https://huggingface.co/smgjch/Meow-Omni-1) |

---


**Training is done in two stages:**
1. **Stage 1 — Projector Pre-training:** Only the TSPM projector is trained; all other components are frozen.
2. **Stage 2 — Multimodal Supervised Fine-Tuning (MSFT):** The LLM backbone is fine-tuned with all encoders frozen.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (CUDA 12.1+ recommended)
- 1× GPU with ≥24 GB VRAM for inference; 8× GPUs recommended for training

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Meow-Omni-1.git
cd Meow-Omni-1
```

### 2. Create a virtual environment

```bash
conda create -n meow python=3.10 -y
conda activate meow
```

### 3. Install dependencies

Please follow the official installation guide at [pytorch.org](https://pytorch.org) to install PyTorch appropriate for your system (CPU/CUDA version, OS, package manager). For example:
```bash
# CUDA 12.1 example — use the exact command from pytorch.org
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

```
Then,
```bash
pip install -r requirements.txt
```

---

## Quick Start

### Inference

```python
import torch
from src.modeling_meow_omni_1 import MeowOmni1ForCausalLM
from src.processing_meow_omni_1 import MeowOmni1Processor

MODEL_PATH = "smgjch/Meow-Omni-1"  # or local checkpoint path

processor = MeowOmni1Processor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = MeowOmni1ForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to("cuda").eval()
```

#### Text + Video + Audio + Time-Series (Full Quad-Modal)

```python
from PIL import Image
import soundfile as sf
import numpy as np

# Load modalities
image = Image.open("cat_frame.jpg").convert("RGB")
audio_array, sr = sf.read("cat_audio.wav")

prompt = (
    "<image>./</image>\n"
    "<audio>./</audio>\n"
    "<|ts_start|><|ts_unit|><|ts_end|>\n"
    "User: Based on the video, audio, and physiological signals, "
    "what is the cat's current emotional state?\n"
    "Assistant:"
)

inputs = processor(
    text=[prompt],
    images=[image],
    audios=[audio_array.astype(np.float32)],
    time_series_paths=["cat_physio.npy"],
    time_series_sampling_rates=[100.0],
    return_tensors="pt"
)
inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

print(processor.tokenizer.decode(output[0], skip_special_tokens=True))
```

#### Text + Video Only

```python
from PIL import Image

image = Image.open("cat_frame.jpg").convert("RGB")
prompt = "<image>./</image>\nUser: What is this cat doing?\nAssistant:"

inputs = processor(text=[prompt], images=[image], return_tensors="pt")
inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)

print(processor.tokenizer.decode(output[0], skip_special_tokens=True))
```

---

> **Video files:** The example above passes a single frame. For full video input, uniformly sample multiple frames and pass them as a list of PIL images, with one `<image>./</image>` placeholder per frame. The evaluator ([`eval_meow.py`](src/evaluation/eval_meow.py)) provides a ready-to-use `_load_video_frames()` helper that handles this automatically (defaulting to 16 frames via `decord`):
> ```python
> import numpy as np
> from decord import VideoReader, cpu
>
> def load_video_frames(video_path, max_frames=16):
>     vr = VideoReader(video_path, ctx=cpu(0))
>     indices = np.linspace(0, len(vr) - 1, min(max_frames, len(vr)), dtype=int)
>     frames = vr.get_batch(indices.tolist()).asnumpy()
>     return [Image.fromarray(f).convert("RGB") for f in frames]
>
> frames = load_video_frames("cat_video.mp4")
> prompt = "".join(["<image>./</image>"] * len(frames)) + "\nUser: What is this cat doing?\nAssistant:"
> inputs = processor(text=[prompt], images=frames, return_tensors="pt")
> ```
 

## Supported Time-Series Formats

The `MeowOmni1Processor` supports loading time-series data from the following file format:

| Format | Extension | Notes |
|---|---|---|
| NumPy array | `.npy` | Shape `(T,)` or `(T, C)` |

---

## Evaluation

### Running MeowBench

```bash
bash eval_meow_omni_1.sh
```

Or manually:

```bash
python src/evaluation/eval_meow.py \
    --model_path smgjch/Meow-Omni-1 \
    --data_path ./data/benchmark_omni_synthesis.json \
    --output_file ./results/eval_results/evaluation_results.json \
    --device cuda \
    --modals video audio ts
```

**Arguments:**

| Argument | Description |
|---|---|
| `--model_path` | Path to local checkpoint or HuggingFace model ID |
| `--data_path` | Path to the benchmark JSON file |
| `--output_file` | Where to write results (auto-saves and resumes) |
| `--device` | `cuda` or `cpu` |
| `--modals` | Space-separated list of active modalities: `video`, `audio`, `ts`. Omit for all. |

The evaluator supports **automatic resumption** — if interrupted, it picks up from where it left off.

---

## Training

### Data Format

Training data should be a JSON file containing a list of sample dictionaries:

```json
[
  {
    "id": "sample_001",
    "conversations": [
      {
        "role": "user",
        "content": "What is this cat's intent?"
      },
      {
        "role": "assistant",
        "content": "B"
      }
    ],
    "image_path": "/data/images/cat_001.jpg",
    "audio_path": "/data/audio/cat_001.wav",
    "ts_path": "/data/ts/cat_001.npy",
    "ts_sr": 100.0,
    "video_path": null
  }
]
```

Any combination of modality fields may be present or omitted per sample.

### Stage 1: Projector Pre-training

Trains only the time-series projector head. All other weights are frozen.

```bash
bash run_pretrain.sh
```

Key hyperparameters (edit `run_pretrain.sh`):

| Parameter | Default | Description |
|---|---|---|
| `--learning_rate` | `1e-3` | Higher LR for projector-only training |
| `--num_train_epochs` | `1` | Usually 1 epoch is sufficient |
| `--per_device_train_batch_size` | `8` | Adjust per GPU memory |
| `--model_max_length` | `2048` | Token sequence length cap |

### Stage 2: Supervised Fine-Tuning

Fine-tunes the LLM with encoders frozen. Start from the Stage 1 checkpoint.

```bash
bash run_postrain.sh
```

Edit `run_postrain.sh` to set `--model_name_or_path` to your Stage 1 checkpoint directory.

Key hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `--learning_rate` | `2e-5` | Standard SFT learning rate |
| `--num_train_epochs` | `4` | |
| `--tune_llm` | `True` | Fine-tune LLM backbone |
| `--tune_ts` | `False` | Keep TS encoder frozen |
| `--use_lora` | `False` | Set `True` for LoRA fine-tuning |

### LoRA Fine-tuning

To run parameter-efficient fine-tuning with LoRA, set `--use_lora True` in Stage 2. LoRA is applied to the LLM's attention projections (`q_proj`, `k_proj`, `v_proj`) by default.

```bash
torchrun --nproc_per_node=8 ./src/training/train.py \
    --model_name_or_path PATH/TO/STAGE1/CHECKPOINT \
    --data_path PATH/TO/DATA \
    --output_dir output/ \
    --stage "2" \
    --use_lora True \
    --tune_llm False \
    --lora_r 64 \
    --lora_alpha 64 \
    --bf16 True
```

---

## Project Structure

```
.
├── src/
│   ├── modeling_meow_omni_1.py       # Model definition (extends MiniCPM-o)
│   ├── configuration_meow_omni_1.py  # Config class (adds ts_config)
│   ├── processing_meow_omni_1.py     # Processor (adds time-series handling)
│   ├── training/
│   │   ├── train.py                  # Main training script
│   │   ├── dataset.py                # MeowOmni1Dataset + collator
│   │   ├── meow_trainer.py           # Custom Trainer (compute_loss)
│   │   └── trainer.py                # ZeRO-3 safe trainer with save_model
│   └── evaluation/
│       └── eval_meow.py              # Benchmark evaluation script
├── run_pretrain.sh                   # Stage 1 launcher
├── run_postrain.sh                   # Stage 2 launcher
├── eval_meow_omni_1.sh               # Evaluation launcher
└── requirements.txt
```

---

## Results

## Results
 
All models are evaluated on **MeowBench**. It is a expert-verified, quad-modal multiple-choice questions set covering 30 feline intent categories.
 
### Comparison with State-of-the-Art Baselines
 
| Model | Vision | Audio | TS | Accuracy |
|---|:---:|:---:|:---:|:---:|
| Acoustic SOTA (Ntalampiras et al., SVM/HMM) | | ✓ | | 36.86% |
| TS SOTA (Chen et al., 1D-CNN + LSTM on IMU) | | | ✓ | 48.98% |
| Video SOTA (Qwen3.5-122B-A10B, zero-shot) | ✓ | | | 61.95% |
| Qwen3.5-Omni-Plus (V + A) | ✓ | ✓ | | 65.36% |
| Qwen3.5-Omni-Plus (V + TS†) | ✓ | | ✓† | 66.21% |
| Qwen3.5-Omni-Plus (TS† + A) | | ✓ | ✓† | 42.15% |
| Qwen3.5-Omni-Plus (V + A + TS†) | ✓ | ✓ | ✓† | 66.89% |
| **Meow-Omni 1 (Ours)** | ✓ | ✓ | ✓ | **71.16%** |
 
† Qwen3.5-Omni-Plus does not accept raw time-series as a native modality; TS data was injected as a structured textual summary (array statistics per channel). Meow-Omni 1 processes raw TS natively.

---

## Citation

If you use Meow-Omni 1 in your research, please cite:

```bibtex
@article{hu2026meowomni,
  title   = {Meow-Omni 1: A Multimodal Large Language Model for Feline Ethology},
  author  = {Hu, Jucheng and Chen, Zhangquan and Chen, Yulin and Hong, Chengjie and
             Zhou, Liang and Wang, Tairan and Li, Sifei and Zhu, Giulio and
             Zhou, Feng and Zeng, Yiheng and Yang, Suorong and Zhou, Dongzhan},
  journal = {arXiv preprint arXiv:2605.09152},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.09152v1}
}
```

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

Portions of the codebase are adapted from:
- [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) — Apache 2.0
- [InternVL / InternS1-Pro](https://github.com/OpenGVLab/InternVL) — Apache 2.0

---

## Acknowledgements

We thank the teams behind MiniCPM-o and InternS1-Pro for their foundational open-source contributions, and the veterinary experts who contributed to MeowBench annotation.