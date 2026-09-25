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

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from asr_engine import StreamingAsrEngine
from audio_capture import TARGET_RATE, LoopbackAudioCapture
from ring_buffer import AudioRingBuffer
from speaker_detector import InstantChangeDetector, SpeakerEmbeddingService
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
        # Speaker detection parameters
        speaker_model_path: str | None = None,
        speaker_k: float = 2.5,
        speaker_min_window: int = 3,
        std_floor: float = 0.05,
        max_window: int = 20,
        speaker_split_backdate_seconds: float = 0.7,
        # Speaker detection parameters
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
        # Speaker detection parameters
        self.speaker_model_path = speaker_model_path
        self.speaker_k = speaker_k
        self.speaker_min_window = speaker_min_window
        self.std_floor = std_floor
        self.max_window = max_window
        self.speaker_split_backdate_seconds = speaker_split_backdate_seconds
        # Speaker detection parameters

        
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

            
            speaker_service = None
            detector = None
            if self.speaker_model_path:
                speaker_service = SpeakerEmbeddingService(
                    model_path=self.speaker_model_path,
                    provider=self.provider,
                    num_threads=1,
                    sample_rate=TARGET_RATE,
                )
                detector = InstantChangeDetector(
                    k=self.speaker_k,
                    min_window=self.speaker_min_window,
                    std_floor=self.std_floor,
                    max_window=self.max_window,
                )

            self.capture.start()
            self.status_changed.emit(f"Listening: {self.capture.device_info['name']}")

            pre_roll_max_chunks = int(PRE_ROLL_SECONDS / DRAIN_INTERVAL_S) + 2
            pre_roll = collections.deque(maxlen=pre_roll_max_chunks)
            was_speech_active = False
            last_partial_emit_time = 0.0
            last_emitted_partial = ""

            # Keep up to 5 words from the previous finalized line to catch overlaps
            # (words re-decoded from the audio_history replay after a REAL
            # backend reset -- unrelated to speaker-change splitting below).
            overlap_memory: list[str] = []

            # --- UI-only speaker-change line splitting state ---
            # These never touch the ASR backend. pending_ui_offset counts
            # how many words of the CURRENT (still-open) backend utterance
            # have already been emitted as a previous UI line via
            # emit_ui_split(); it resets to 0 only when the backend truly
            # finalizes/resets (in handle_update's finalized branch below).
            # last_known_words mirrors the most recent decoded+deduped word
            # list for the current utterance, so emit_ui_split() has
            # something to slice even though it runs from process_chunk,
            # not handle_update. total_duration_at_last_split lets both a
            # UI split and the eventual real finalize log a duration scoped
            # to just their own words, instead of the whole utterance's
            # cumulative duration.
            #
            # word_count_history records (wall_clock_time, word_count) every
            # time last_known_words grows. emit_ui_split() uses this to find
            # the word count as it stood --speaker-split-backdate seconds in
            # the past, instead of splitting at the current instant. Two
            # problems this solves at once: (1) text decoded in the last
            # ~second is often still settling -- a transducer needs a little
            # trailing audio context to firm up its last word or two, and
            # splitting on it live was producing truncated/garbled fragments
            # (e.g. a lone low-confidence "Ask" or a hallucinated "As the
            # thing" that doesn't even appear in a clean decode of the same
            # audio); (2) the detector itself needs some accumulation window
            # to confirm a change, so by the time it fires, a bit of the
            # INCOMING speaker's audio has usually already been decoded as
            # part of the still-open (outgoing) line -- backdating the split
            # leaves those trailing words pending instead of gluing them onto
            # the wrong line, so they show up at the start of the next line
            # instead.
            pending_ui_offset = 0
            last_known_words: list[str] = []
            total_duration_at_last_split = 0.0
            word_count_history: collections.deque = collections.deque()

            def process_chunk(chunk: np.ndarray) -> None:
                # 1. Feed audio array to ASR engine
                update = asr.feed(chunk)
                handle_update(update)

                # 2. Feed the SAME audio array to speaker detector
                if speaker_service and detector:
                    emb = speaker_service.extract(chunk)
                    if emb is not None:
                        is_change, z_score = detector.process(emb)
                        if is_change:
                            print(f"[SPEAKER CHANGE] Triggered with Z-score: {z_score} (Threshold k={self.speaker_k})")
                            emit_ui_split()
                            speaker_service.reset()



            def handle_update(update) -> None:
                nonlocal last_partial_emit_time, last_emitted_partial, overlap_memory
                nonlocal pending_ui_offset, last_known_words, total_duration_at_last_split
                nonlocal word_count_history

                if update.finalized_text is not None:
                    raw_words = update.finalized_text.split()
                    overlap_count = _find_overlap(overlap_memory, raw_words)
                    deduped_words = raw_words[overlap_count:]

                    # Words already shown via an earlier UI-only speaker-change
                    # split within THIS SAME backend utterance (emit_ui_split)
                    # are already on screen -- only emit what's new since then.
                    new_words = deduped_words[pending_ui_offset:]

                    total_now = update.duration_seconds or 0.0
                    segment_duration = max(total_now - total_duration_at_last_split, 0.0)

                    # The backend utterance is truly finalizing/resetting now,
                    # so all UI-split bookkeeping for it resets too. Word
                    # count history from this utterance is meaningless once
                    # word indices restart at 0 for the next one.
                    overlap_memory = raw_words[-5:] if raw_words else []
                    pending_ui_offset = 0
                    last_known_words = []
                    total_duration_at_last_split = 0.0
                    word_count_history = collections.deque()

                    if new_words:
                        formatted = format_line(
                            " ".join(new_words),
                            numbers=self.numbers,
                            number_threshold=self.number_threshold,
                        )
                        # NOTE: confidence is the whole utterance's estimate,
                        # not scoped to just these trailing words -- splitting
                        # it precisely would need per-token timestamps (the
                        # ASR result already carries them) rather than the
                        # single averaged score used here. Duration IS scoped
                        # correctly via the delta above.
                        logger.log(
                            text=formatted,
                            confidence=update.confidence,
                            duration=segment_duration,
                        )
                        self.line_finalized.emit(formatted)
                    last_emitted_partial = ""
                    return

                # Handle partial updates
                raw_words = update.partial_text.split()
                overlap_count = _find_overlap(overlap_memory, raw_words)
                deduped_words = raw_words[overlap_count:]
                if len(deduped_words) != len(last_known_words):
                    word_count_history.append((time.perf_counter(), len(deduped_words)))
                    # Trim history older than we'll ever need to look back.
                    cutoff = time.perf_counter() - self.speaker_split_backdate_seconds - 2.0
                    while len(word_count_history) > 1 and word_count_history[0][0] < cutoff:
                        word_count_history.popleft()
                last_known_words = deduped_words

                # Only display words after the last UI-only split point, so a
                # speaker change doesn't leave the previous speaker's words
                # re-appearing at the front of the new partial line.
                display_words = deduped_words[pending_ui_offset:]

                now = time.perf_counter()
                formatted_partial = format_line(
                    " ".join(display_words),
                    numbers=self.numbers,
                    number_threshold=self.number_threshold,
                )
                if formatted_partial != last_emitted_partial and (
                    now - last_partial_emit_time >= PARTIAL_EMIT_MIN_INTERVAL_S
                ):
                    self.partial_updated.emit(formatted_partial)
                    last_emitted_partial = formatted_partial
                    last_partial_emit_time = now

            def emit_ui_split() -> None:
                """Ends the current UI line, backdated by
                --speaker-split-backdate seconds, WITHOUT touching the ASR
                backend -- the stream, its endpoint detector, and the
                audio_history overlap buffer all continue completely
                undisturbed. Only the display/log layer is split, and only
                up to where word_count_history says the decode stood
                self.speaker_split_backdate_seconds seconds ago, not the current instant.

                This replaces the old force_finalize() approach, which reset
                the ASR stream mid-utterance and lost whatever word(s) the
                decoder hadn't yet committed at that exact sample. Splitting
                on the LIVE snapshot (an earlier version of this function)
                fixed the word-loss but introduced a different problem:
                the live edge of a streaming decode is often still settling,
                and the detector itself needs an accumulation window before
                it fires, so a live split both produced occasional garbled/
                truncated fragments at the boundary and left some of the
                incoming speaker's already-decoded words glued onto the
                outgoing line. Backdating targets an already-settled point
                in the decode and one that's closer to when the acoustic
                change actually happened, addressing both at once.

                Any words newer than the backdated point stay pending
                (pending_ui_offset does NOT advance past them) -- they'll
                correctly appear at the start of the next line instead of
                being dropped.
                """
                nonlocal pending_ui_offset, last_emitted_partial, total_duration_at_last_split

                if not last_known_words:
                    return

                target_time = time.perf_counter() - self.speaker_split_backdate_seconds
                backdated_count = pending_ui_offset  # fallback: no usable history yet
                for t, count in word_count_history:
                    if t <= target_time:
                        backdated_count = count
                    else:
                        break
                split_index = max(pending_ui_offset, min(backdated_count, len(last_known_words)))

                new_words = last_known_words[pending_ui_offset:split_index]
                if not new_words:
                    return

                # Approximation: this counts audio time up to NOW, not up to
                # the backdated split point, so it slightly overstates this
                # segment's duration (by up to ~self.speaker_split_backdate_seconds seconds).
                # A precise version would use the ASR's per-token timestamps
                # to find the exact audio time of the last included word.
                snapshot = asr.current_snapshot()
                total_now = snapshot.duration_seconds or 0.0
                segment_duration = max(total_now - total_duration_at_last_split, 0.0)

                formatted = format_line(
                    " ".join(new_words),
                    numbers=self.numbers,

                    number_threshold=self.number_threshold,
                )
                logger.log(
                    text=formatted,
                    confidence=snapshot.confidence,
                    duration=segment_duration,
                )
                self.line_finalized.emit(formatted)

                # NOT len(last_known_words) -- leave any words newer than
                # split_index pending so they surface at the start of the
                # next line instead of being silently swallowed here.
                pending_ui_offset = split_index
                total_duration_at_last_split = total_now
                last_emitted_partial = ""

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
                            for chunk in pre_roll:
                                process_chunk(chunk)
                            pre_roll.clear()
                        process_chunk(samples)
                    else:
                        pre_roll.append(samples)
                        if was_speech_active and speaker_service:
                            speaker_service.reset()

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