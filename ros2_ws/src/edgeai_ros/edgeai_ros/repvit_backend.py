from __future__ import annotations

import gc
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class TinyFallbackCNN(nn.Module):
    """Small CNN used when RepViT cannot be imported.

    The fallback keeps the ROS/OS runtime testable without model downloads. Its
    predictions are not meaningful and should not be used for accuracy claims.
    """

    def __init__(self, width: int = 32, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


@dataclass
class ModelLoadResult:
    model: nn.Module
    fallback_model: int
    source: str
    device: torch.device
    load_latency_ms: float = 0.0
    cache_hit: int = 0


def _candidate_repvit_paths(explicit_path: str = ""):
    paths = []
    if explicit_path:
        paths.append(explicit_path)
    for env_name in ["REPViT_PATH", "REPVIT_PATH", "EDGEAI_REPVIT_PATH"]:
        value = os.environ.get(env_name)
        if value:
            paths.append(value)
    paths.extend(
        [
            "/workspace/OS2026/RepViT",
            "/workspace/EdgeAI-ROS/RepViT",
            "/mnt/usr1/azure005/OS2026/OS2026/RepViT",
        ]
    )
    seen = set()
    for path in paths:
        path = os.path.abspath(path)
        if path not in seen and os.path.exists(path):
            seen.add(path)
            yield path


def _load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device, logger=None):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return False
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    if logger is not None:
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return True


@torch.no_grad()
def _warmup_model(model: nn.Module, device: torch.device, logger=None):
    try:
        x = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if logger is not None:
            logger.info(f"Completed {device.type} model warm-up")
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Model warm-up failed: {exc}")


class AdaptiveModelCache:
    """LRU cache for RepViT/fallback models selected by the scheduler."""

    def __init__(
        self,
        device: torch.device,
        cache_size: int = 2,
        repvit_path: str = "",
        checkpoint_dir: str = "",
        use_timm_fallback: bool = True,
        logger=None,
    ):
        self.device = device
        self.cache_size = max(1, int(cache_size))
        self.repvit_path = repvit_path
        self.checkpoint_dir = checkpoint_dir
        self.use_timm_fallback = use_timm_fallback
        self.logger = logger
        self.cache = OrderedDict()

    def get(self, model_name: str, device: torch.device | None = None) -> ModelLoadResult:
        device = device or self.device
        key = (model_name, str(device))
        if key in self.cache:
            result = self.cache.pop(key)
            self.cache[key] = result
            return ModelLoadResult(
                model=result.model,
                fallback_model=result.fallback_model,
                source=result.source,
                device=result.device,
                load_latency_ms=0.0,
                cache_hit=1,
            )

        t0 = time.perf_counter()
        result = self._load(model_name, device)
        result.load_latency_ms = (time.perf_counter() - t0) * 1000.0
        result.cache_hit = 0
        self.cache[key] = result
        while len(self.cache) > self.cache_size:
            _, old = self.cache.popitem(last=False)
            del old
            gc.collect()
            if any(str(cache_key[1]).startswith("cuda") for cache_key in self.cache.keys()):
                torch.cuda.empty_cache()
        return result

    def _load(self, model_name: str, device: torch.device) -> ModelLoadResult:
        result = self._load_official_repvit(model_name, device)
        if result is not None:
            return result
        if self.use_timm_fallback:
            result = self._load_timm(model_name, device)
            if result is not None:
                return result
        width = 16 if "m0" in model_name or "small" in model_name else 32
        model = TinyFallbackCNN(width=width).eval().to(device)
        return ModelLoadResult(model=model, fallback_model=1, source="tiny_fallback_cnn", device=device)

    def _load_official_repvit(self, model_name: str, device: torch.device) -> Optional[ModelLoadResult]:
        for path in _candidate_repvit_paths(self.repvit_path):
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                from model import repvit

                constructor_name = model_name.replace("-", "_")
                if "." in constructor_name:
                    constructor_name = constructor_name.split(".")[0]
                if not hasattr(repvit, constructor_name):
                    continue
                model = getattr(repvit, constructor_name)(pretrained=False)
                checkpoint_path = self._checkpoint_path(constructor_name)
                try:
                    _load_checkpoint(model, checkpoint_path, device, self.logger)
                except Exception as exc:
                    if self.logger is not None:
                        self.logger.warning(f"Checkpoint load failed for {checkpoint_path}: {exc}")
                model = model.eval().to(device)
                _warmup_model(model, device, self.logger)
                return ModelLoadResult(model=model, fallback_model=0, source=f"official:{path}", device=device)
            except Exception as exc:
                if self.logger is not None:
                    self.logger.warning(f"Official RepViT load failed from {path}: {exc}")
        return None

    def _load_timm(self, model_name: str, device: torch.device) -> Optional[ModelLoadResult]:
        try:
            import timm

            model = timm.create_model(model_name, pretrained=False, num_classes=1000)
            model = model.eval().to(device)
            _warmup_model(model, device, self.logger)
            return ModelLoadResult(model=model, fallback_model=0, source="timm", device=device)
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning(f"timm load failed for {model_name}: {exc}")
            return None

    def _checkpoint_path(self, model_name: str) -> str:
        if not self.checkpoint_dir:
            return ""
        candidates = [
            f"{model_name}.pth",
            f"{model_name}_distill_300e.pth",
            f"{model_name}_distill_450e.pth",
        ]
        for name in candidates:
            path = os.path.join(self.checkpoint_dir, name)
            if os.path.exists(path):
                return path
        return os.path.join(self.checkpoint_dir, candidates[0])


def image_msg_to_bgr(msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    channels = 3
    if msg.encoding.lower() in ["mono8", "8uc1"]:
        channels = 1
    arr = arr.reshape((msg.height, msg.width, channels))
    if channels == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if msg.encoding.lower() == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def preprocess_bgr(frame_bgr: np.ndarray, image_size: int, device: torch.device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = ((arr - mean) / std).transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0).float().to(device)


@torch.no_grad()
def infer_once(model: nn.Module, frame_bgr: np.ndarray, image_size: int, device: torch.device):
    x = preprocess_bgr(frame_bgr, image_size, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0
    try:
        pred = int(torch.argmax(y, dim=1).item())
    except Exception:
        pred = -1
    return infer_ms, pred
