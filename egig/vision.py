"""Vision models shared by Agent 2 (curating reference photos) and Agent 4 (verifying).

- Grounding DINO: open-vocabulary detector. Give it a word ("dog"), get boxes + scores.
- CLIP: turns images (and text) into vectors; cosine similarity = how alike they look.

Both are loaded lazily and need `torch` + `transformers`. They run on CPU too (slower).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from .geometry import Box, nms, normalize_pixels


def vision_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def default_vision_device() -> str:
    """On Kaggle (2 GPUs) keep GPU 0 for SDXL and put the vision models on GPU 1."""
    import torch

    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "cpu"


@dataclass
class RawDetection:
    label: str
    score: float
    box: Box  # normalized


class VisionTools:
    def __init__(self, gdino_model: str, clip_model: str, device: Optional[str] = None,
                 box_threshold: float = 0.35, text_threshold: float = 0.25):
        self.gdino_model = gdino_model
        self.clip_model = clip_model
        self.device = device or default_vision_device()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._gd_model = self._gd_proc = None
        self._clip = self._clip_proc = None

    # -------------------------------------------------------------- detection
    def detect(self, image: Image.Image, label: str, box_threshold: Optional[float] = None) -> List[RawDetection]:
        """Boxes for one label, after NMS (duplicate boxes on the same object removed)."""
        import torch

        if self._gd_model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._gd_proc = AutoProcessor.from_pretrained(self.gdino_model)
            self._gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_model)
            self._gd_model.to(self.device).eval()

        image = image.convert("RGB")
        text = label.strip().lower().rstrip(".") + "."  # Grounding DINO wants lowercase + trailing dot
        inputs = self._gd_proc(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._gd_model(**inputs)

        threshold = self.box_threshold if box_threshold is None else box_threshold
        post = self._gd_proc.post_process_grounded_object_detection
        params = inspect.signature(post).parameters
        kwargs = {"text_threshold": self.text_threshold, "target_sizes": [(image.height, image.width)]}
        kwargs["threshold" if "threshold" in params else "box_threshold"] = threshold  # renamed in newer transformers
        if "input_ids" in params:
            kwargs["input_ids"] = inputs["input_ids"]
        result = post(outputs, **kwargs)[0]

        dets = [
            RawDetection(label=label, score=float(s), box=normalize_pixels(b, image.width, image.height))
            for b, s in zip(result["boxes"].tolist(), result["scores"].tolist())
        ]
        return nms(dets, 0.5)

    # ------------------------------------------------------------------- CLIP
    def _load_clip(self) -> None:
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            self._clip_proc = CLIPProcessor.from_pretrained(self.clip_model)
            self._clip = CLIPModel.from_pretrained(self.clip_model)
            self._clip.to(self.device).eval()

    def image_embeddings(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        self._load_clip()
        inputs = self._clip_proc(images=[im.convert("RGB") for im in images], return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = _features(self._clip.get_image_features(**inputs))
        return _unit(feats)

    def text_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._load_clip()
        inputs = self._clip_proc(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            feats = _features(self._clip.get_text_features(**inputs))
        return _unit(feats)


def _features(out: object) -> np.ndarray:
    """`get_*_features` returns a tensor in most transformers versions, an output object in some."""
    import torch

    if not torch.is_tensor(out):
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(out, attr, None)
            if value is not None:
                out = value
                break
    return out.float().cpu().numpy()  # type: ignore[union-attr]


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))
