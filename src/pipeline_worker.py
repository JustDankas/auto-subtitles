"""
Runs capture -> VAD gate -> incremental streaming ASR -> logging on a
background QThread, emitting throttled partial updates and immediate
finalized-line signals for the GUI.

Two behaviors worth calling out:

1. PRE-ROLL BUFFER: the VAD needs a brief window of confirmed speech before
   is_speech_active() flips True, which would otherwise clip the first
   word each time someone starts talking. We keep a short rolling buffer of
   recent (not-yet-classified-as-speech) audio and replay it into the ASR
   stream the moment speech is confirmed.

2. THROTTLED PARTIALS: partial text can change many times per second as
   the model decodes; we only emit new_partial_text when the text has
   actually changed AND at most ~4 times/sec, so the GUI redraws calmly
   instead of flickering.
"""

import collections
import time
from pathlib import Path

from asr_engine import StreamingAsrEngine
from audio_capture import TARGET_RATE, LoopbackAudioCapture
from PyQt6.QtCore import QThread, pyqtSignal
from ring_buffer import AudioRingBuffer
from text_formatter import format_line
from transcript_logger import TranscriptLogger
from vad_gate import SpeechGate

DRAIN_INTERVAL_S = 0.1
PRE_ROLL_SECONDS = 0.3
PARTIAL_EMIT_MIN_INTERVAL_S = 0.25  # caps partial redraw rate to ~4/sec


def find_model_files(model_dir: Path, want_int8: bool) -> dict:
    encoders = sorted(model_dir.glob("encoder*.onnx"))
    decoders = sorted(model_dir.glob("decoder*.onnx"))
    joiners = sorted(model_dir.glob("joiner*.onnx"))
    tokens = model_dir / "tokens.txt"
    if not encoders or not decoders or not joiners or not tokens.exists():
        raise FileNotFoundError(f"Could not find model files under {model_dir}")

    def pick(paths):
        int8_matches = [p for p in paths if "int8" in p.name]
        fp32_matches = [p for p in paths if "int8" not in p.name]
        chosen = (int8_matches or paths) if want_int8 else (fp32_matches or paths)
        return str(chosen[0])

    return {
        "encoder": pick(encoders),
        "decoder": pick(decoders),
        "joiner": pick(joiners),
        "tokens": str(tokens),
    }

def _find_overlap(prev_words: list[str], new_words: list[str]) -> int:
    """Returns the number of overlapping words at the boundary."""
    max_match = min(len(prev_words), len(new_words))
    prev_words = [w.lower() for w in prev_words]
    new_words = [w.lower() for w in new_words]
    for i in range(max_match, 0, -1):
        if prev_words[-i:] == new_words[:i]:
            return i
    return 0




class PipelineWorker(QThread):
    partial_updated = pyqtSignal(str)  # throttled, in-progress line
    line_finalized = pyqtSignal(str)  # a completed line, emitted once per line
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        vad_model_path: str,
        asr_model_dir: str,
        log_file: str,
        provider: str = "cpu",
        int8: bool = False,
        vad_threshold: float = 0.5,
        min_silence: float = 0.5,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 12.0,
        numbers: bool = True,
        number_threshold: float = 3.0,
        num_threads: int = 3,
        overlap_seconds: float = 1.0,
        parent=None,
    ):
        super().__init__(parent)
        self.vad_model_path = vad_model_path
        self.asr_model_dir = asr_model_dir
        self.log_file = log_file
        self.provider = provider
        self.int8 = int8
        self.vad_threshold = vad_threshold
        self.min_silence = min_silence
        self.rule2_min_trailing_silence = rule2_min_trailing_silence
        self.rule3_min_utterance_length = rule3_min_utterance_length
        self.numbers = numbers
        self.number_threshold = number_threshold
        self.num_threads = num_threads
        self.overlap_seconds = overlap_seconds
        
        self._stop_requested = False
        self.capture: LoopbackAudioCapture | None = None

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        import sys
        import traceback

        logger = None
        error_message = None
        try:
            self.status_changed.emit("Loading ASR model...")
            model_files = find_model_files(Path(self.asr_model_dir), self.int8)
            asr = StreamingAsrEngine(
                encoder=model_files["encoder"],
                decoder=model_files["decoder"],
                joiner=model_files["joiner"],
                tokens=model_files["tokens"],
                num_threads=self.num_threads,
                sample_rate=TARGET_RATE,
                provider=self.provider,
                rule2_min_trailing_silence=self.rule2_min_trailing_silence,
                rule3_min_utterance_length=self.rule3_min_utterance_length,
                overlap_seconds=self.overlap_seconds,
            )

            ring_buffer = AudioRingBuffer(sample_rate=TARGET_RATE, max_seconds=10.0)
            self.capture = LoopbackAudioCapture(ring_buffer)
            gate = SpeechGate(
                model_path=self.vad_model_path,
                sample_rate=TARGET_RATE,
                threshold=self.vad_threshold,
                min_silence_duration=self.min_silence,
            )
            logger = TranscriptLogger(self.log_file)

            self.capture.start()
            self.status_changed.emit(f"Listening: {self.capture.device_info['name']}")

            pre_roll_max_chunks = int(PRE_ROLL_SECONDS / DRAIN_INTERVAL_S) + 2
            pre_roll = collections.deque(maxlen=pre_roll_max_chunks)
            was_speech_active = False
            last_partial_emit_time = 0.0
            last_emitted_partial = ""

            # Keep up to 5 words from the previous finalized line to catch overlaps
            overlap_memory: list[str] = []



            def handle_update(update) -> None:
                nonlocal last_partial_emit_time, last_emitted_partial, overlap_memory
                if update.finalized_text is not None:
                    raw_words = update.finalized_text.split()
                    overlap_count = _find_overlap(overlap_memory, raw_words)
                    deduped_text = " ".join(raw_words[overlap_count:])
                    
                    formatted = format_line(
                        deduped_text,
                        numbers=self.numbers,
                        number_threshold=self.number_threshold,
                    )
                    logger.log(
                        text=formatted,
                        confidence=update.confidence,
                        duration=update.duration_seconds or 0.0,
                    )
                    self.line_finalized.emit(formatted)
                    last_emitted_partial = ""
                    # Store the end of this finalized text to deduplicate the next stream
                    overlap_memory = raw_words[-5:] if raw_words else []                    
                    return

                
                # Handle partial updates
                raw_words = update.partial_text.split()
                overlap_count = _find_overlap(overlap_memory, raw_words)
                deduped_text = " ".join(raw_words[overlap_count:])

                now = time.perf_counter()
                formatted_partial = format_line(
                    deduped_text,
                    numbers=self.numbers,
                    number_threshold=self.number_threshold,
                )
                if formatted_partial != last_emitted_partial and (
                    now - last_partial_emit_time >= PARTIAL_EMIT_MIN_INTERVAL_S
                ):
                    self.partial_updated.emit(formatted_partial)
                    last_emitted_partial = formatted_partial
                    last_partial_emit_time = now

            while not self._stop_requested:
                if self.capture.error is not None:
                    self.error_occurred.emit(str(self.capture.error))
                    break

                samples = ring_buffer.read_all_available()
                if samples.size > 0:
                    # Drain VAD's internal finalized-segment queue for
                    # hygiene (we don't use the segments themselves in this
                    # design - the ASR's own endpoint detector decides line
                    # boundaries - but must still pop() or its buffer grows
                    # unboundedly over a long session).
                    for _ in gate.process(samples):
                        pass

                    is_active = gate.is_speech_active()
                    if is_active:
                        if not was_speech_active:
                            print(f"VAD: {is_active}, pre_roll: {len(pre_roll)}")
                            for chunk in pre_roll:
                                handle_update(asr.feed(chunk))
                            pre_roll.clear()
                        handle_update(asr.feed(samples))
                    else:
                        print(f"VAD: {is_active}, pre_roll: {len(pre_roll)}")
                        pre_roll.append(samples)
                    was_speech_active = is_active

                self.msleep(int(DRAIN_INTERVAL_S * 1000))

            final_update = asr.flush()
            if final_update is not None:
                handle_update(final_update)

        except Exception as e:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            error_message = str(e) or type(e).__name__
            self.error_occurred.emit(error_message)
        finally:
            if self.capture is not None:
                self.capture.stop()
            if logger is not None:
                logger.close()
            if error_message is None:
                self.status_changed.emit("Stopped")
