"""
detectors.py -- Algorithm interface + registry for the speaker-change-detection
evaluation harness.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from SCD.scd_model import SCDNet


@dataclass
class BoundaryEvent:
    """A single predicted speaker-change boundary."""
    boundary_time_s: float          # the algorithm's estimate of *when* the change happened
    confidence: float               # 0..1, higher = more confident
    extra: Dict[str, Any] = field(default_factory=dict)   # free-form debug info


class SpeakerChangeDetector(abc.ABC):
    """Common interface every candidate algorithm implements."""

    def __init__(self, sample_rate: int, **kwargs: Any) -> None:
        self.sample_rate = sample_rate
        self.config = kwargs

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all internal state. Called once per audio file."""
        raise NotImplementedError

    @abc.abstractmethod
    def process_chunk(self, chunk: np.ndarray, chunk_start_s: float) -> List[BoundaryEvent]:
        """Consume one streaming chunk of mono float32 PCM samples in [-1, 1]."""
        raise NotImplementedError

    def finalize(self, stream_end_s: float) -> List[BoundaryEvent]:
        return []


# ---------------------------------------------------------------------------
# Plan 2 -- End-to-End SCD Classifier (Integrated with scd_model.py)
# ---------------------------------------------------------------------------
class SCDClassifierDetector(SpeakerChangeDetector):
    """
    Feeds sliding windows of raw audio through `SCDNet` from `scd_model.py`
    and uses Non-Maximum Suppression (NMS) over time frames to output
    speaker change boundaries.

    Expected config kwargs:
        weights_path: str                   # path to trained PyTorch .pth / .pt state_dict
        window_ms: int = 3000               # model input audio window size in ms
        hop_ms: int = 1000                  # window hop size in ms
        threshold: float = 0.5              # probability threshold for change-bump
        nms_window_ms: int = 300            # time window to suppress duplicate peaks
        device: str = "cpu"                 # "cpu" or "cuda"
    """

    def __init__(self, sample_rate: int, **kwargs: Any) -> None:
        super().__init__(sample_rate, **kwargs)
        self.device = torch.device(self.config.get("device", "cpu"))

        weights_path = self.config.get("weights_path")
        ckpt = None
        state_dict = None
        model_cfg = {}

        if weights_path:
            ckpt = torch.load(weights_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
                model_cfg = ckpt.get("config", {})
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                state_dict = ckpt

        # Build network configuration matching train.py defaults with checkpoint overrides
        n_mels = model_cfg.get("n_mels", self.config.get("n_mels", 64))
        ch = model_cfg.get("ch", self.config.get("ch", 64))
        kernel_size = model_cfg.get("kernel_size", self.config.get("kernel_size", 5))
        dilations = model_cfg.get("dilations", self.config.get("dilations", (1, 2, 4, 8, 16)))
        dropout = model_cfg.get("dropout", self.config.get("dropout", 0.15))
        causal = model_cfg.get("causal", self.config.get("causal", False))
        n_outputs = model_cfg.get("n_outputs", self.config.get("n_outputs", 2))
        dense = model_cfg.get("dense", self.config.get("dense", False))

        # Instantiate SCDNet matching saved training topology
        self.model = SCDNet(
            n_mels=n_mels,
            ch=ch,
            kernel_size=kernel_size,
            dilations=tuple(dilations) if isinstance(dilations, list) else dilations,
            dropout=dropout,
            causal=causal,
            augment=False,  # Keep specaug disabled during evaluation/inference
            n_outputs=n_outputs,
            dense=dense,
        )

        if state_dict is not None:
            # Handle model saved with augment=True module weights if present
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def reset(self) -> None:
        self.window_ms = self.config.get("window_ms", 3000)
        self.hop_ms = self.config.get("hop_ms", 1000)
        self.threshold = self.config.get("threshold", 0.5)
        self.nms_window_ms = self.config.get("nms_window_ms", 300)

        self.window_samples = int(self.sample_rate * self.window_ms / 1000.0)
        self.hop_samples = int(self.sample_rate * self.hop_ms / 1000.0)

        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_s = 0.0
        self._last_processed_sample = 0
        self._last_emitted_s: Optional[float] = None

    def process_chunk(self, chunk: np.ndarray, chunk_start_s: float) -> List[BoundaryEvent]:
        if len(self._audio_buffer) == 0:
            self._buffer_start_s = chunk_start_s

        self._audio_buffer = np.concatenate([self._audio_buffer, chunk])
        events: List[BoundaryEvent] = []

        while len(self._audio_buffer) - self._last_processed_sample >= self.window_samples:
            start_idx = self._last_processed_sample
            end_idx = start_idx + self.window_samples
            window_audio = self._audio_buffer[start_idx:end_idx]

            window_start_s = self._buffer_start_s + (start_idx / self.sample_rate)
            window_events = self._run_model_on_window(window_audio, window_start_s)
            events.extend(window_events)

            self._last_processed_sample += self.hop_samples

        return events

    def _run_model_on_window(self, window_audio: np.ndarray, window_start_s: float) -> List[BoundaryEvent]:
        wav_tensor = torch.from_numpy(window_audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(wav_tensor)  # shape: (1, n_outputs, T)
            # Channel 0 contains the change-bump logits
            probs = torch.sigmoid(out[0, 0, :]).cpu().numpy()

        num_frames = len(probs)
        duration_s = len(window_audio) / self.sample_rate
        frame_dur_s = duration_s / num_frames if num_frames > 0 else 0.01

        events: List[BoundaryEvent] = []
        nms_frames = int(round((self.nms_window_ms / 1000.0) / frame_dur_s)) if frame_dur_s > 0 else 1

        # Peak detection with Non-Maximum Suppression (NMS)
        for i in range(num_frames):
            score = probs[i]
            if score >= self.threshold:
                i_start = max(0, i - nms_frames)
                i_end = min(num_frames, i + nms_frames + 1)
                
                if score == np.max(probs[i_start:i_end]):
                    boundary_time_s = window_start_s + (i * frame_dur_s)

                    # Suppress duplicated emissions across overlapping windows
                    if (
                        self._last_emitted_s is None
                        or (boundary_time_s - self._last_emitted_s) > (self.nms_window_ms / 1000.0)
                    ):
                        events.append(
                            BoundaryEvent(
                                boundary_time_s=float(boundary_time_s),
                                confidence=float(score),
                            )
                        )
                        self._last_emitted_s = boundary_time_s

        return events


# ---------------------------------------------------------------------------
# Stubs for other detectors
# ---------------------------------------------------------------------------
class AdjacentWindowDetector(SpeakerChangeDetector):
    def reset(self) -> None: raise NotImplementedError
    def process_chunk(self, chunk: np.ndarray, chunk_start_s: float) -> List[BoundaryEvent]: raise NotImplementedError

class VADCentroidDetector(SpeakerChangeDetector):
    reset = process_chunk = lambda self, *a, **kw: None

class SemanticEndpointDetector(SpeakerChangeDetector):
    reset = process_chunk = lambda self, *a, **kw: None

class CosineBaselineDetector(SpeakerChangeDetector):
    reset = process_chunk = lambda self, *a, **kw: None

class NoOpDetector(SpeakerChangeDetector):
    def reset(self) -> None: pass
    def process_chunk(self, chunk: np.ndarray, chunk_start_s: float) -> List[BoundaryEvent]: return []


ALGORITHM_REGISTRY: Dict[str, Type[SpeakerChangeDetector]] = {
    "adjacent_window": AdjacentWindowDetector,
    "scd_classifier": SCDClassifierDetector,
    "vad_centroid": VADCentroidDetector,
    "semantic_endpoint": SemanticEndpointDetector,
    "cosine_baseline": CosineBaselineDetector,
    "noop": NoOpDetector,
}


def build_detector(name: str, sample_rate: int, **kwargs: Any) -> SpeakerChangeDetector:
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHM_REGISTRY)}")
    return ALGORITHM_REGISTRY[name](sample_rate=sample_rate, **kwargs)