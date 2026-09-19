"""
A fixed-size, thread-safe circular buffer of float32 mono audio samples.

Design intent: one producer (the audio capture callback, on PortAudio's own
thread) writes small chunks continuously. One or more consumers (later: a
VAD/ASR thread) periodically drain it. If a consumer falls behind for longer
than `max_seconds`, the OLDEST audio is overwritten and we count it as
"dropped" rather than growing memory unboundedly or blocking the audio
callback (blocking the audio thread causes audible glitches/underruns).

For a real-time subtitle app, a few seconds of ring buffer is plenty -
if the consumer is more than a few seconds behind, something downstream is
too slow and we want to know that (via `stats`), not silently buffer minutes
of audio.
"""

import threading

import numpy as np


class AudioRingBuffer:
    def __init__(self, sample_rate: int = 16000, max_seconds: float = 30.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self._buffer = np.zeros(self.max_samples, dtype=np.float32)
        self._write_pos = 0
        self._filled = 0
        self._lock = threading.Lock()

        # Diagnostics - read via the `stats` property.
        self._total_written = 0
        self._total_dropped = 0

    def write(self, samples: np.ndarray) -> None:
        """Append samples, called from the producer (audio capture callback)."""
        if samples.size == 0:
            return
        with self._lock:
            n = len(samples)
            if n > self.max_samples:
                # A single write larger than the whole buffer: keep only the tail.
                samples = samples[-self.max_samples:]
                n = len(samples)

            end = self._write_pos + n
            if end <= self.max_samples:
                self._buffer[self._write_pos:end] = samples
            else:
                first_part = self.max_samples - self._write_pos
                self._buffer[self._write_pos:] = samples[:first_part]
                self._buffer[: end - self.max_samples] = samples[first_part:]
            self._write_pos = end % self.max_samples

            overflow = max(0, self._filled + n - self.max_samples)
            self._filled = min(self.max_samples, self._filled + n)
            self._total_written += n
            self._total_dropped += overflow

    def read_all_available(self) -> np.ndarray:
        """Drain and return everything currently buffered, oldest sample first."""
        with self._lock:
            if self._filled == 0:
                return np.empty(0, dtype=np.float32)
            start = (self._write_pos - self._filled) % self.max_samples
            if start + self._filled <= self.max_samples:
                out = self._buffer[start:start + self._filled].copy()
            else:
                first_part = self.max_samples - start
                out = np.concatenate(
                    [self._buffer[start:], self._buffer[: self._filled - first_part]]
                )
            self._filled = 0
            return out

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "buffered_seconds": self._filled / self.sample_rate,
                "total_written_seconds": self._total_written / self.sample_rate,
                "total_dropped_seconds": self._total_dropped / self.sample_rate,
            }
