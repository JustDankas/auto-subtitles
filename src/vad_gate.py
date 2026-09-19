"""
Speech gating via sherpa-onnx's built-in Silero VAD support.

Deliberately uses sherpa_onnx.VoiceActivityDetector (bundled with the
sherpa-onnx package we already installed for ASR) instead of the official
`silero-vad` pip package, which pulls in a full torch/torchaudio install
just to run a ~2MB model. Same onnxruntime, one dependency, one less thing
to version-mismatch.

Silero VAD's window size is fixed at 512 samples (32ms) @ 16kHz - this
class handles buffering arbitrary-length input (whatever the ring buffer
drain produces) into exact windows internally.
"""

from dataclasses import dataclass

import numpy as np
import sherpa_onnx

WINDOW_SIZE = 512  # fixed by the Silero VAD model - do not change


@dataclass
class SpeechSegment:
    start_sample: int  # offset into the VAD's internal continuous sample stream
    samples: np.ndarray  # float32 mono @ sample_rate
    sample_rate: int = 16000

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


class SpeechGate:
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        threshold: float = 0.4,
        min_silence_duration: float = 0.5,
        min_speech_duration: float = 0.25,
        max_speech_duration: float = 20.0,
        buffer_size_in_seconds: float = 60.0,
    ):
        """
        threshold: higher = stricter. Raise this (e.g. 0.6-0.7) if music or
            background noise is being classified as speech. Lower it
            (e.g. 0.3-0.4) if soft/quiet speech is being missed.
        min_silence_duration: how much trailing silence is required before
            an in-progress speech segment is finalized and emitted. Lower
            = lower latency but choppier segments (may split mid-sentence
            on natural pauses). Higher = fewer, longer, more complete
            segments but more latency before text appears.
        max_speech_duration: force-finalize very long uninterrupted speech
            (e.g. a lecturer talking for a full minute) so segments don't
            grow unboundedly and text isn't withheld indefinitely.
        """
        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=model_path,
                threshold=threshold,
                min_silence_duration=min_silence_duration,
                min_speech_duration=min_speech_duration,
                max_speech_duration=max_speech_duration,
                window_size=WINDOW_SIZE,
            ),
            sample_rate=sample_rate,
            num_threads=1,
            provider="cpu",  # VAD is tiny; CPU is plenty, keeps the GPU free for ASR
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(
            vad_config, buffer_size_in_seconds=buffer_size_in_seconds
        )
        self._leftover = np.empty(0, dtype=np.float32)
        self.sample_rate = sample_rate

    def process(self, new_samples: np.ndarray) -> list[SpeechSegment]:
        """
        Feed new audio (any length - buffering into exact VAD windows is
        handled internally). Returns any speech segments that became
        finalized as a result of this call (i.e. speech followed by enough
        trailing silence, or a force-finalize on max_speech_duration).
        """
        buf = np.concatenate([self._leftover, new_samples])
        n_windows = len(buf) // WINDOW_SIZE
        usable = n_windows * WINDOW_SIZE
        self._leftover = buf[usable:].copy()

        for i in range(n_windows):
            window = buf[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
            self._vad.accept_waveform(window)

        return self._drain()

    def is_speech_active(self) -> bool:
        """True if the VAD currently believes we're mid-speech (segment not
        yet finalized). Useful later for a 'listening...' UI indicator."""
        return self._vad.is_speech_detected()

    def flush(self) -> list[SpeechSegment]:
        """Call at shutdown to force out any in-progress (not yet silence-
        terminated) segment, so trailing speech isn't lost."""
        self._vad.flush()
        return self._drain()

    def _drain(self) -> list[SpeechSegment]:
        segments = []
        while not self._vad.empty():
            seg = self._vad.front  # property, not a method, in this binding
            segments.append(
                SpeechSegment(
                    start_sample=seg.start,
                    samples=np.array(seg.samples, dtype=np.float32),
                    sample_rate=self.sample_rate,
                )
            )
            self._vad.pop()
        return segments
