"""
Streaming wrapper around SCDNet: a sliding 3 s window over VAD-gated audio
-> ChangeEvents stamped on a *gated-audio clock*.

Key ideas (see SCD_CONVERT_PLAN_v2.md, sections 4.2 and 4.3):

- ONE CLOCK. `clock_s` is the total number of gated samples ever pushed,
  divided by 16000. It only advances for audio passed to push() (the same
  audio the ASR sees) and is NEVER reset. reset() only clears the audio
  window, so event times stay comparable across silences.
- The model was trained on 3.0 s windows (48000 samples -> 301 frames) with
  changes only 0.5-2.5 s into the window. So a peak is only trusted inside
  [left_guard, window - right_guard]. A change is therefore first reported
  about `right_guard` .. `right_guard + hop` seconds after it happened.
- Decision rule mirrors eval_utils: sigmoid(channel 0) -> 5-frame uniform
  smoothing (edge replicated) -> threshold -> local peak.

The backend is swappable: anything callable as
    backend(window: float32[48000]) -> float32[301] of per-frame P(change)
works (TorchBackend now, an ONNX backend later - plan Phase 8).
"""

import inspect
import json
from dataclasses import dataclass

import numpy as np

SR = 16000
FRAME_S = 0.01  # mel hop = 160 samples
WINDOW_S = 3.0
EXPECTED_FRAMES = 301  # output length for a 3.0 s window


@dataclass
class ChangeEvent:
    time_s: float  # change position on the audio clock
    prob: float  # smoothed peak probability
    detected_at_s: float  # end of the window that fired it (chunk-size independent; latency = detected_at_s - time_s)


class TorchBackend:
    """window (48000,) float32 -> per-frame P(change) (301,) float32."""

    def __init__(self, checkpoint: str, device: str = "cpu", num_threads: int = 1):
        import torch

        from scd_model import SCDNet

        torch.set_num_threads(num_threads)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)  # trusted local file
        cfg = dict(ckpt["config"])
        allowed = set(inspect.signature(SCDNet.__init__).parameters) - {"self"}
        ignored = sorted(set(cfg) - allowed)
        if ignored:
            print(f"[SCD] ignoring non-model config keys: {ignored}")
        cfg = {k: v for k, v in cfg.items() if k in allowed}
        cfg["augment"] = False
        sd = {
            k.removeprefix("module.").removeprefix("_orig_mod."): v
            for k, v in ckpt["model"].items()
        }
        model = SCDNet(**cfg)
        model.load_state_dict(sd, strict=True)  # hyperparameter mismatch must fail loudly
        self.model, self.device, self.torch = model.to(device).eval(), device, torch
        self.causal = bool(cfg.get("causal", False))

        # Warm-up (first call is slow) + output-shape sanity check.
        out = self(np.zeros(int(WINDOW_S * SR), dtype=np.float32))
        if out.ndim != 1 or len(out) != EXPECTED_FRAMES:
            raise ValueError(
                f"SCDNet output shape {out.shape} != ({EXPECTED_FRAMES},); "
                "the window/frame assumptions in scd_detector.py no longer hold"
            )
        print(f"[SCD] loaded {checkpoint} (device={device}, causal={self.causal})")

    def __call__(self, window: np.ndarray) -> np.ndarray:
        x = self.torch.from_numpy(window).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return self.torch.sigmoid(self.model(x)[0, 0]).cpu().numpy()  # channel 0 only


def smooth(p: np.ndarray, k: int = 5) -> np.ndarray:
    """Same as eval_utils.smooth (uniform_filter1d, mode='nearest') without scipy."""
    if k <= 1:
        return p
    q = np.pad(p, k // 2, mode="edge")
    return np.convolve(q, np.ones(k, np.float32) / k, mode="valid")


class StreamingSpeakerChangeDetector:
    def __init__(
        self,
        backend,
        window_s: float = WINDOW_S,
        hop_s: float = 0.25,
        threshold: float = 0.5,
        smooth_k: int = 5,
        left_guard_s: float = 0.5,
        right_guard_s: float = 1.0,
        min_gap_s: float = 1.0,
        nms_radius_s: float = 0.1,
        max_catchup: int = 8,
        debug_dump_path: str | None = None,
    ):
        if hop_s <= 0:
            raise ValueError("hop_s must be > 0")
        if left_guard_s + right_guard_s >= window_s:
            raise ValueError("left_guard_s + right_guard_s must be < window_s")
        self.backend = backend
        self.win, self.hop = int(window_s * SR), int(hop_s * SR)
        self.threshold, self.smooth_k = threshold, smooth_k
        self.left_guard_s, self.right_guard_s = left_guard_s, right_guard_s
        self.min_gap_s, self.nms = min_gap_s, max(1, int(nms_radius_s / FRAME_S))
        self.max_catchup = max_catchup
        self._buf = np.zeros(0, np.float32)
        self._total = 0  # gated samples pushed; NEVER reset
        self._next_end = self.win  # absolute sample index where the next window ends
        self._last_emit_s = -1e9
        self._dump = open(debug_dump_path, "a", encoding="utf-8") if debug_dump_path else None

    @property
    def clock_s(self) -> float:
        return self._total / SR

    def reset(self) -> None:
        """Clear the audio window (after a long gap). The clock keeps running."""
        self._buf = np.zeros(0, np.float32)
        self._next_end = self._total + self.win

    def close(self) -> None:
        if self._dump is not None:
            self._dump.close()
            self._dump = None

    def push(self, samples: np.ndarray) -> list[ChangeEvent]:
        """Any chunk length. Runs every window whose end lies inside the new audio."""
        samples = np.asarray(samples, dtype=np.float32)
        self._buf = np.concatenate([self._buf, samples])
        self._total += len(samples)
        due = []
        while self._next_end <= self._total:
            due.append(self._next_end)
            self._next_end += self.hop
        if len(due) > self.max_catchup:  # fell far behind: skip old windows, keep the newest
            due = due[-self.max_catchup:]
        events: list[ChangeEvent] = []
        for end in due:
            end_rel = len(self._buf) - (self._total - end)
            window = self._buf[end_rel - self.win:end_rel]
            events += self._decide(self.backend(window), win_start_s=(end - self.win) / SR, win_end_s=end / SR)
        self._buf = self._buf[-self.win:]
        return events

    def _decide(self, prob: np.ndarray, win_start_s: float, win_end_s: float) -> list[ChangeEvent]:
        p = smooth(prob, self.smooth_k)
        n, w = len(p), self.nms
        lo = max(int(self.left_guard_s / FRAME_S), w)
        hi = min(n - int(self.right_guard_s / FRAME_S), n - w)
        if self._dump is not None:
            self._dump.write(
                json.dumps({"win_start_s": round(win_start_s, 3),
                            "p": [round(float(v), 3) for v in p]}) + "\n"
            )
        out = []
        for i in range(lo, hi):
            v = float(p[i])
            if v >= self.threshold and v >= p[i - w:i + w + 1].max():
                t = win_start_s + i * FRAME_S
                if abs(t - self._last_emit_s) >= self.min_gap_s:
                    out.append(ChangeEvent(t, v, win_end_s))
                    self._last_emit_s = t
        return out