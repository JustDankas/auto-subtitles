import queue
import threading
import time

import numpy as np
import torch
import torchaudio.transforms as T
from faster_whisper import WhisperModel

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("Please install PyAudioWPatch: pip install PyAudioWPatch")
    exit(1)


class AudioPipeline:
    def __init__(self, model_size="base", vad_threshold=0.5):
        self.model_size = model_size
        self.vad_threshold = vad_threshold

        # Pipelines Queues
        self.audio_queue = queue.Queue()
        self.asr_queue = queue.Queue()
        self.text_queue = queue.Queue()  # Thread-safe Queue B for output consumption

        # Thread Control Flag
        self._stop_event = threading.Event()

        # Handles
        self._vad_thread = None
        self._asr_thread = None
        self._audio_stream = None
        self._pyaudio_instance = None

    def _get_default_loopback_device(self, p):
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        if default_speakers.get("isLoopbackDevice", False):
            return default_speakers
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                return loopback
        raise RuntimeError("Could not find a matching WASAPI loopback device.")

    def _asr_worker(self):
        print(f"[Pipeline] Loading Faster-Whisper model '{self.model_size}' (int8)...")
        model = WhisperModel(self.model_size, device="cuda", compute_type="int8")
        print("[Pipeline] ASR Worker ready.")

        while not self._stop_event.is_set():
            try:
                audio_segment = self.asr_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if audio_segment is None:
                break

            start_time = time.time()
            segments, _ = model.transcribe(audio_segment, beam_size=5, language="en")
            text = " ".join([seg.text for seg in segments]).strip()
            latency = time.time() - start_time

            if text:
                self.text_queue.put((text, latency))

    def _vad_worker(self, input_rate, channels):
        print("[Pipeline] Loading Silero VAD model...")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        model.eval()

        target_rate = 16000
        chunk_size = 512
        silence_timeout_frames = 15
        max_segment_frames = 1000

        resample_transform = T.Resample(orig_freq=input_rate, new_freq=target_rate)
        buffer = np.array([], dtype=np.float32)

        is_speaking = False
        silence_counter = 0
        speech_frames = 0
        utterance_buffer = []

        print("[Pipeline] VAD Worker ready.")

        while not self._stop_event.is_set():
            try:
                raw_data = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if raw_data is None:
                break

            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels).mean(axis=1)

            audio_tensor = torch.from_numpy(audio_data)
            audio_16k = resample_transform(audio_tensor).numpy()
            buffer = np.concatenate((buffer, audio_16k))

            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]

                if is_speaking:
                    utterance_buffer.append(chunk)

                chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
                with torch.no_grad():
                    speech_prob = model(chunk_tensor, target_rate).item()

                if speech_prob >= self.vad_threshold:
                    if not is_speaking:
                        is_speaking = True
                        utterance_buffer.append(chunk)
                    silence_counter = 0
                    speech_frames += 1
                else:
                    if is_speaking:
                        silence_counter += 1
                        speech_frames += 1

                if is_speaking and (
                    silence_counter >= silence_timeout_frames
                    or speech_frames >= max_segment_frames
                ):
                    is_speaking = False
                    if utterance_buffer:
                        full_segment = np.concatenate(utterance_buffer)
                        self.asr_queue.put(full_segment)

                    silence_counter = 0
                    speech_frames = 0
                    utterance_buffer = []

    def start(self):
        """Starts audio capture, VAD, and ASR threads."""
        self._stop_event.clear()
        self._pyaudio_instance = pyaudio.PyAudio()
        device = self._get_default_loopback_device(self._pyaudio_instance)
        channels = device["maxInputChannels"]
        sample_rate = int(device["defaultSampleRate"])

        self._asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
        self._vad_thread = threading.Thread(
            target=self._vad_worker, args=(sample_rate, channels), daemon=True
        )
        self._asr_thread.start()
        self._vad_thread.start()

        def callback(in_data, frame_count, time_info, status):
            if not self._stop_event.is_set():
                self.audio_queue.put(in_data)
            return (in_data, pyaudio.paContinue)

        self._audio_stream = self._pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device["index"],
            stream_callback=callback,
        )
        self._audio_stream.start_stream()
        print("[Pipeline] System Audio Subtitle Pipeline started.")

    def stop(self):
        """Stops all threads and terminates audio streams safely."""
        print("[Pipeline] Stopping audio pipeline...")
        self._stop_event.set()

        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()

        if self._pyaudio_instance:
            self._pyaudio_instance.terminate()

        print("[Pipeline] Pipeline stopped.")


if __name__ == "__main__":
    pipeline = AudioPipeline(model_size="base")
    pipeline.start()

    print("Pipeline running headless. Press Ctrl+C to stop...\n")
    try:
        while True:
            try:
                text, latency = pipeline.text_queue.get(timeout=0.1)
                print(f"[{latency:.2f}s] {text}")
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pipeline.stop()