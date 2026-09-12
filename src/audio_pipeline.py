import queue
import threading
import time
from collections import deque

import numpy as np
import torch
import torchaudio.transforms as T
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("Please install PyAudioWPatch: pip install PyAudioWPatch")
    exit(1)


BEAM_SIZE = 2                    # Reduced from 5 to prevent thread blocking on final pass
PARTIAL_BEAM_SIZE = 1            # Fast greedy pass for interim text
PARTIAL_INTERVAL_SECONDS = 0.25  # Frequent partial polling (~4Hz)
VAD_CHUNK_SAMPLES = 512          # Silero VAD chunk size at 16kHz (~32ms)
MIN_SILENCE_DURATION_MS = 380    # Lowered from 550ms to catch micro-pauses quicker
SPEECH_PAD_MS = 200              # Padding around detected speech edges
PRE_ROLL_CHUNKS = 10             # ~320ms rolling onset buffer
MAX_SEGMENT_SECONDS = 3.2        # Lowered from 7.0s to force frequent chunks on monologues
MIN_SEGMENT_SECONDS = 0.3        # Ignores noise blips
PROMPT_RESET_GAP_SECONDS = 6.0   # Context reset threshold after silence


class AudioPipeline:
    def __init__(self, model_size="base", vad_threshold=0.5):
        self.model_size = model_size
        self.vad_threshold = vad_threshold

        self.audio_queue = queue.Queue()
        self.asr_queue = queue.Queue()
        self.text_queue = queue.Queue()

        self._stop_event = threading.Event()
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
        print("[Pipeline] Loading Faster-Whisper model...")
        model = WhisperModel(self.model_size, device="cuda", compute_type="int8")
        print("[Pipeline] ASR Worker ready.")

        last_text = ""
        last_emit_time = 0.0

        while not self._stop_event.is_set():
            try:
                item = self.asr_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            audio_segment = item["audio"]
            is_final = item["is_final"]
            utterance_id = item["utterance_id"]

            now = time.time()
            if last_text and (now - last_emit_time) > PROMPT_RESET_GAP_SECONDS:
                last_text = ""

            start_time = time.time()
            if is_final:
                segments, _ = model.transcribe(
                    audio_segment,
                    beam_size=BEAM_SIZE,
                    language="en",
                    condition_on_previous_text=False,
                    initial_prompt=last_text[-150:] if last_text else None,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=250),
                )
            else:
                segments, _ = model.transcribe(
                    audio_segment,
                    beam_size=PARTIAL_BEAM_SIZE,
                    language="en",
                    condition_on_previous_text=False,
                )

            text = " ".join([seg.text for seg in segments]).strip()
            latency = time.time() - start_time

            print(f"[Pipeline] {'FINAL' if is_final else 'partial'} "
                  f"(id={utterance_id}, {latency * 1000:.0f}ms): {text!r}")

            if text:
                self.text_queue.put((text, latency, is_final, utterance_id))
                if is_final:
                    last_text = f"{last_text} {text}".strip() if last_text else text
                    last_emit_time = time.time()

    def _vad_worker(self, input_rate, channels):
        print("[Pipeline] Loading Silero VAD model...")
        model = load_silero_vad()
        vad_iterator = VADIterator(
            model,
            threshold=self.vad_threshold,
            sampling_rate=16000,
            min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )

        target_rate = 16000
        resample_transform = T.Resample(orig_freq=input_rate, new_freq=target_rate)
        buffer = np.array([], dtype=np.float32)

        history = deque(maxlen=PRE_ROLL_CHUNKS)
        utterance_chunks = []
        recording = False
        utterance_id = 0
        last_partial_time = 0.0

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

            while len(buffer) >= VAD_CHUNK_SAMPLES:
                chunk = buffer[:VAD_CHUNK_SAMPLES]
                buffer = buffer[VAD_CHUNK_SAMPLES:]

                history.append(chunk)
                if recording:
                    utterance_chunks.append(chunk)

                chunk_tensor = torch.from_numpy(chunk)
                with torch.no_grad():
                    event = vad_iterator(chunk_tensor, return_seconds=False)

                if event and "start" in event and not recording:
                    recording = True
                    utterance_id += 1
                    last_partial_time = time.time()
                    utterance_chunks = list(history)

                elif event and "end" in event and recording:
                    self._emit_segment(utterance_chunks, target_rate, utterance_id, is_final=True)
                    utterance_chunks = []
                    recording = False

                if recording:
                    duration = len(utterance_chunks) * VAD_CHUNK_SAMPLES / target_rate

                    if duration >= MAX_SEGMENT_SECONDS:
                        self._emit_segment(utterance_chunks, target_rate, utterance_id, is_final=True)
                        utterance_chunks = []
                        vad_iterator.reset_states()
                        utterance_id += 1
                        last_partial_time = time.time()

                    elif (time.time() - last_partial_time) >= PARTIAL_INTERVAL_SECONDS:
                        self._emit_segment(utterance_chunks, target_rate, utterance_id, is_final=False)
                        last_partial_time = time.time()

    def _emit_segment(self, chunks, sample_rate, utterance_id, is_final):
        if not chunks:
            return
        full_segment = np.concatenate(chunks).astype(np.float32)
        duration = len(full_segment) / sample_rate
        if is_final and duration < MIN_SEGMENT_SECONDS:
            return

        # Purge stale pending partials while preserving the internal deque type
        if not is_final:
            with self.asr_queue.mutex:
                self.asr_queue.queue = deque(
                    item for item in self.asr_queue.queue if item["is_final"]
                )

        self.asr_queue.put({
            "audio": full_segment,
            "is_final": is_final,
            "utterance_id": utterance_id,
        })

    def start(self):
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
            if self._stop_event.is_set():
                return (None, pyaudio.paComplete)
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
        if self._stop_event.is_set():
            return
        print("[Pipeline] Stopping audio pipeline...")
        self._stop_event.set()

        if self._audio_stream:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass

        if self._pyaudio_instance:
            try:
                self._pyaudio_instance.terminate()
            except Exception:
                pass

        if self._vad_thread and self._vad_thread.is_alive():
            self._vad_thread.join(timeout=1.0)
        if self._asr_thread and self._asr_thread.is_alive():
            self._asr_thread.join(timeout=1.0)

        print("[Pipeline] Pipeline stopped successfully.")