# SAM3 (Inference-Only Restructure)

This repository is an inference-focused restructure of the original SAM3 project from Meta AI. It keeps the core model/predictor code for running image/video inference, while excluding training workflows.

## Upstream Source

- Original project: https://github.com/facebookresearch/sam3

## What This Repo Includes

- Core SAM3 model code under `sam3/`
- Video inference predictor API (`build_sam3_video_predictor`)
- Example inference script: `main.py`

## What Is Out of Scope

- Training support (disabled in this build)
- Training pipelines and dataset workflows

## Requirements

- Python 3.8+
- NVIDIA GPU + CUDA (required for `build_sam3_video_predictor`, which runs on CUDA)
- PyTorch + torchvision compatible with your CUDA version
- Additional runtime packages:
  - `opencv-python` (default MP4/video loading path)
  - `psutil`
  - `Pillow`

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

# Install a CUDA-matching torch/torchvision build first (from pytorch.org)
pip install -e .
pip install opencv-python psutil pillow
```

## Model Weights

By default, the builder downloads weights from Hugging Face:

- Model: `facebook/sam3`
- Checkpoint: `sam3.pt`

If access is gated, authenticate first:

```bash
huggingface-cli login
```

You can also pass a local checkpoint path:

```python
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(checkpoint_path="path/to/sam3.pt")
```

## Quick Start (Video Inference)

```python
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor()
video_path = "assets/videos/bedroom.mp4"  # MP4, single image, or JPEG-frame directory

# 1) Start session
session = predictor.handle_request(
    {"type": "start_session", "resource_path": video_path}
)
session_id = session["session_id"]

# 2) Add prompt on a frame
result = predictor.handle_request(
    {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": "person",
    }
)
print(result["outputs"].keys())
# out_obj_ids, out_probs, out_boxes_xywh, out_binary_masks, frame_stats

# 3) Propagate prompt through video
for step in predictor.handle_stream_request(
    {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",  # forward | backward | both
    }
):
    frame_idx = step["frame_index"]
    outputs = step["outputs"]

# 4) Close session when done
predictor.handle_request({"type": "close_session", "session_id": session_id})
```

To run the included example script:

```bash
python main.py
```

## Request API (Video Predictor)

`handle_request` supports:

- `start_session` with `resource_path`
- `add_prompt` with:
  - `session_id`
  - `frame_index`
  - optional `text`
  - optional `points` + `point_labels`
  - optional `bounding_boxes` + `bounding_box_labels`
  - optional `obj_id`
- `remove_object` with `session_id`, `obj_id`
- `reset_session` with `session_id`
- `close_session` with `session_id`

`handle_stream_request` supports:

- `propagate_in_video` with:
  - `session_id`
  - optional `propagation_direction` (`both`, `forward`, `backward`)
  - optional `start_frame_index`
  - optional `max_frame_num_to_track`

## License

This repository is distributed under the **SAM License** in `LICENSE.txt` (Last Updated: November 19, 2025).

Important requirements from that license include:

- Redistributions of SAM materials and derivatives must remain under the same SAM License terms.
- A copy of the license must be provided when distributing SAM materials or derivatives.
- Publications based on this work must acknowledge use of SAM materials.

## Attribution and Citation

This repository is derived from:

- Meta AI, SAM3: https://github.com/facebookresearch/sam3

If you use this codebase, please cite the original SAM3 project and paper:

- Paper page: https://arxiv.org/abs/2511.16719

```bibtex
@article{carion2025sam3,
  title={SAM 3: Segment Anything in Images and Videos},
  author={Carion, Nicolas and others},
  journal={arXiv preprint arXiv:2511.16719},
  year={2025}
}
```
