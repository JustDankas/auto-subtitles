"""
Continuous WASAPI loopback capture -> 16kHz mono float32 -> AudioRingBuffer.

Runs via a PortAudio stream callback (not a blocking read loop), so capture
happens on PortAudio's own high-priority thread with minimal added latency.
The callback must be fast and must never block, so:
  - downmix + resample happen inline (cheap, pure numpy/audioop)
  - the ring buffer write is a quick lock/copy, not I/O

This is the class that Phase 4+ will import directly into the final app.
"""

import audioop
import threading

import numpy as np
import pyaudiowpatch as pyaudio

from ring_buffer import AudioRingBuffer

TARGET_RATE = 16000


class LoopbackAudioCapture:
    def __init__(self, ring_buffer: AudioRingBuffer, chunk_frames: int = 1024):
        self.ring_buffer = ring_buffer
        self.chunk_frames = chunk_frames

        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._device = None
        self._src_rate = None
        self._src_channels = None
        self._resample_state = None

        self._running = False
        self._last_error: Exception | None = None
        self._error_lock = threading.Lock()

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        self._device = self._pa.get_default_wasapi_loopback()
        self._src_rate = int(self._device["defaultSampleRate"])
        self._src_channels = self._device["maxInputChannels"]
        self._resample_state = None
        self._running = True

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._src_channels,
            rate=self._src_rate,
            frames_per_buffer=self.chunk_frames,
            input=True,
            input_device_index=self._device["index"],
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status_flags):
        # status_flags carries PortAudio xrun/overflow bits. We don't treat
        # these as fatal (brief hiccups happen), but a downstream metric
        # (ring buffer's total_dropped_seconds) will catch anything that
        # actually costs us audio.
        try:
            mono_bytes = self._downmix(in_data, self._src_channels)
            resampled, self._resample_state = audioop.ratecv(
                mono_bytes, 2, 1, self._src_rate, TARGET_RATE, self._resample_state
            )
            samples = np.frombuffer(resampled, dtype=np.int16).astype(np.float32) / 32768.0
            self.ring_buffer.write(samples)
        except Exception as e:  # noqa: BLE001 - must not raise inside a PortAudio callback
            with self._error_lock:
                self._last_error = e

        return (None, pyaudio.paContinue if self._running else pyaudio.paComplete)

    @staticmethod
    def _downmix(raw_bytes: bytes, channels: int) -> bytes:
        if channels == 1:
            return raw_bytes
        if channels == 2:
            return audioop.tomono(raw_bytes, 2, 0.5, 0.5)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, channels)
        mono = samples.astype(np.float32).mean(axis=1).clip(-32768, 32767).astype(np.int16)
        return mono.tobytes()

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    @property
    def error(self) -> Exception | None:
        with self._error_lock:
            return self._last_error

    @property
    def device_info(self) -> dict | None:
        return self._device
