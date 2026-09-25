"""Speaker embedding extraction and real-time z-score change detection."""

import numpy as np
import sherpa_onnx


class InstantChangeDetector:
    def __init__(
        self,
        k: float = 2.5,
        min_window: int = 3,
        std_floor: float = 0.05,
        max_window: int = 20,
    ):
        self.window: list[np.ndarray] = []
        self.k = k
        self.min_window = min_window
        self.std_floor = std_floor
        self.max_window = max_window

    def reset(self) -> None:
        self.window.clear()

    def process(self, emb: np.ndarray) -> tuple[bool, float]:
        # L2-normalize incoming vector
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        if len(self.window) < self.min_window:
            self.window.append(emb)
            return False, 0.0  # Cold start burn-in

        # Compute centroid of active speaker window
        mean_vec = np.mean(self.window, axis=0)
        c_norm = np.linalg.norm(mean_vec)
        centroid = mean_vec / c_norm if c_norm > 0 else mean_vec

        # Cosine similarity for unit vectors is dot product
        sims = [float(np.dot(centroid, e)) for e in self.window]
        mu = float(np.mean(sims))
        sigma = max(float(np.std(sims)), self.std_floor)

        curr_sim = float(np.dot(centroid, emb))
        z = (mu - curr_sim) / sigma

        if z > self.k:
            self.window = [emb]  # Reset window to new speaker
            return True, z

        self.window.append(emb)
        if len(self.window) > self.max_window:
            self.window.pop(0)
        return False, z


class SpeakerEmbeddingService:
    def __init__(
        self,
        model_path: str,
        provider: str = "cpu",
        num_threads: int = 2,
        sample_rate: int = 16000,
        min_audio_seconds: float = 0.8,
    ):
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            num_threads=num_threads,
            provider=provider,
        )
        if not config.validate():
            raise ValueError(f"Invalid SpeakerEmbeddingExtractorConfig for {model_path}")

        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.sample_rate = sample_rate
        self.min_samples = int(min_audio_seconds * sample_rate)
        self._buffer = np.zeros(0, dtype=np.float32)

    def extract(self, samples: np.ndarray) -> np.ndarray | None:
        """Accumulate VAD speech audio until min_samples (~0.8s) is reached for CAM++."""
        self._buffer = np.concatenate([self._buffer, samples])
        if len(self._buffer) < self.min_samples:
            return None

        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=self.sample_rate, waveform=self._buffer)
        stream.input_finished()

        if self.extractor.is_ready(stream):
            emb = np.array(self.extractor.compute(stream), dtype=np.float32)
            # Keep a small tail to prevent audio boundary discontinuities
            keep_samples = int(0.2 * self.sample_rate)
            self._buffer = (
                self._buffer[-keep_samples:]
                if len(self._buffer) > keep_samples
                else np.zeros(0, dtype=np.float32)
            )
            return emb

        return None

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)