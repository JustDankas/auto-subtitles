"""
Runs capture -> VAD gate -> (speaker-change detector) + incremental
streaming ASR -> logging on a background QThread, emitting throttled
partial updates and immediate finalized-line signals for the GUI.

Three behaviors worth calling out:

1. PRE-ROLL BUFFER: the VAD needs a brief window of confirmed speech before
   is_speech_active() flips True, which would otherwise clip the first
   word each time someone starts talking. We keep a short rolling buffer of
   recent (not-yet-classified-as-speech) audio and replay it into the ASR
   stream the moment speech is confirmed.

2. THROTTLED PARTIALS: partial text can change many times per second as
   the model decodes; we only emit new_partial_text when the text has
   actually changed AND at most ~4 times/sec, so the GUI redraws calmly
   instead of flickering.

3. SPEAKER-CHANGE LINE SPLITS (SCDNet): the same gated audio goes to the
   StreamingSpeakerChangeDetector and to the ASR. The detector reports a
   change time on its AUDIO CLOCK (gated samples pushed / 16000). Because
   Nemotron gives no token timestamps, we stamp (clock, word_count) every
   time the partial word count changes, and later split the open utterance
   at count_at(t_change + asr_decode_lag): "the words that had appeared by
   the time the ASR could have decoded everything said before the change".
   The split is UI/log-only; the ASR stream is never touched.
"""

import collections
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from scd_detector import StreamingSpeakerChangeDetector, TorchBackend

from asr_engine import StreamingAsrEngine
from audio_capture import TARGET_RATE, LoopbackAudioCapture
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
        # Speaker change detection (SCDNet). Empty/None checkpoint = disabled.
        scd_checkpoint: str | None = None,
        scd_device: str = "cpu",
        scd_threads: int = 1,
        scd_threshold: float = 0.6,
        scd_hop: float = 0.25,
        scd_left_guard: float = 0.5,
        scd_right_guard: float = 1.0,
        scd_min_gap: float = 1.0,
        scd_reset_gap: float = 1.5,
        scd_settle: float = 0.1,
        scd_debug_dump: str | None = None,
        asr_decode_lag: float = 0.8,
        # ASR / VAD / formatting
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
        # Speaker change detection
        self.scd_checkpoint = scd_checkpoint
        self.scd_device = scd_device
        self.scd_threads = scd_threads
        self.scd_threshold = scd_threshold
        self.scd_hop = scd_hop
        self.scd_left_guard = scd_left_guard
        self.scd_right_guard = scd_right_guard
        self.scd_min_gap = scd_min_gap
        self.scd_reset_gap = scd_reset_gap
        self.scd_settle = scd_settle
        self.scd_debug_dump = scd_debug_dump
        self.asr_decode_lag = asr_decode_lag

        self._stop_requested = False
        self.capture: LoopbackAudioCapture | None = None

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        import sys
        import traceback

        logger = None
        scd = None
        error_message = None
        late_changes = 0  # changes that arrived after their line was already emitted
        changes_applied = 0
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

            if self.scd_checkpoint:
                self.status_changed.emit("Loading speaker-change model...")
                scd = StreamingSpeakerChangeDetector(
                    TorchBackend(
                        self.scd_checkpoint,
                        device=self.scd_device,
                        num_threads=self.scd_threads,
                    ),
                    hop_s=self.scd_hop,
                    threshold=self.scd_threshold,
                    left_guard_s=self.scd_left_guard,
                    right_guard_s=self.scd_right_guard,
                    min_gap_s=self.scd_min_gap,
                    debug_dump_path=self.scd_debug_dump,
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
            # These never touch the ASR backend.
            #   pending_ui_offset : how many words of the CURRENT (still-open)
            #       backend utterance were already emitted as earlier UI lines.
            #       Resets to 0 only when the backend truly finalizes.
            #   last_known_words  : latest decoded+deduped word list of the
            #       open utterance (what apply_change slices).
            #   word_count_history: (scd.clock_s, word_count) stamped whenever
            #       the partial word count changes. Keyed to the AUDIO clock,
            #       so it is independent of processing speed and stalls.
            #   pending_changes   : change times (audio clock) waiting until
            #       the ASR has had asr_decode_lag (+settle) to decode the
            #       words spoken before them.
            #   seg_start_clock   : audio-clock start of the current UI line;
            #       durations are t_change - seg_start_clock.
            pending_ui_offset = 0
            last_known_words: list[str] = []
            word_count_history: collections.deque = collections.deque()
            pending_changes: list[float] = []
            seg_start_clock = 0.0
            inactive_audio_s = 0.0

            def count_at(q: float) -> int:
                """Word count as of the last history stamp at or before audio-clock q."""
                n = None
                for t, c in word_count_history:
                    if t <= q:
                        n = c
                    else:
                        break
                return pending_ui_offset if n is None else n

            def apply_change(t_c: float, final: bool, final_confidence) -> None:
                """Emit the words spoken before audio-clock time t_c as a
                UI-only line. Words newer than the split stay pending and
                open the next line."""
                nonlocal pending_ui_offset, last_emitted_partial, seg_start_clock
                nonlocal late_changes, changes_applied

                if t_c <= seg_start_clock:
                    # Belongs to an utterance/line that was already emitted.
                    late_changes += 1
                    print(f"[SCD] late change @ {t_c:.2f}s (segment starts {seg_start_clock:.2f}s) dropped")
                    return
                split = max(
                    pending_ui_offset,
                    min(count_at(t_c + self.asr_decode_lag), len(last_known_words)),
                )
                words = last_known_words[pending_ui_offset:split]
                if not words:
                    return  # change coincides with a line start / silence only

                confidence = final_confidence if final else asr.current_snapshot().confidence
                formatted = format_line(
                    " ".join(words),
                    numbers=self.numbers,
                    number_threshold=self.number_threshold,
                )
                logger.log(
                    text=formatted,
                    confidence=confidence,
                    duration=max(t_c - seg_start_clock, 0.0),
                )
                self.line_finalized.emit(formatted)
                changes_applied += 1
                pending_ui_offset = split
                seg_start_clock = t_c
                last_emitted_partial = ""

            def resolve_pending(final: bool, confidence=None) -> None:
                """Apply every queued change that is due (or all of them when
                the backend utterance is finalizing)."""
                if not pending_changes:
                    return
                ready = [
                    t for t in pending_changes
                    if final or scd.clock_s >= t + self.asr_decode_lag + self.scd_settle
                ]
                if not ready:
                    return
                pending_changes[:] = [t for t in pending_changes if t not in ready]
                for t in sorted(ready):
                    apply_change(t, final, confidence)

            def process_chunk(chunk: np.ndarray) -> None:
                # SCD goes BEFORE the ASR so scd.clock_s already includes this
                # chunk when handle_update stamps the word-count history.
                if scd:
                    for ev in scd.push(chunk):
                        pending_changes.append(ev.time_s)
                        print(
                            f"[SCD] change @ {ev.time_s:.2f}s p={ev.prob:.2f} "
                            f"lag={ev.detected_at_s - ev.time_s:.2f}s"
                        )
                handle_update(asr.feed(chunk))
                if scd and pending_changes:
                    resolve_pending(final=False)

            def handle_update(update) -> None:
                nonlocal last_partial_emit_time, last_emitted_partial, overlap_memory
                nonlocal pending_ui_offset, last_known_words, seg_start_clock

                if update.finalized_text is not None:
                    raw_words = update.finalized_text.split()
                    overlap_count = _find_overlap(overlap_memory, raw_words)
                    deduped_words = raw_words[overlap_count:]

                    # Resolve any queued speaker changes against the FINAL word
                    # list first (they all belong to this utterance).
                    last_known_words = deduped_words
                    if scd:
                        resolve_pending(final=True, confidence=update.confidence)

                    # Words already shown via an earlier UI-only split within
                    # THIS SAME backend utterance are already on screen --
                    # only emit what's new since then.
                    new_words = deduped_words[pending_ui_offset:]

                    if scd:
                        segment_duration = max(scd.clock_s - seg_start_clock, 0.0)
                    else:
                        segment_duration = update.duration_seconds or 0.0

                    # The backend utterance is truly finalizing/resetting now,
                    # so all UI-split bookkeeping for it resets too.
                    overlap_memory = raw_words[-5:] if raw_words else []
                    pending_ui_offset = 0
                    last_known_words = []
                    word_count_history.clear()
                    pending_changes.clear()
                    if scd:
                        seg_start_clock = scd.clock_s

                    if new_words:
                        formatted = format_line(
                            " ".join(new_words),
                            numbers=self.numbers,
                            number_threshold=self.number_threshold,
                        )
                        # NOTE: confidence is the whole utterance's estimate,
                        # not scoped to just these trailing words.
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
                if scd and len(deduped_words) != len(last_known_words):
                    word_count_history.append((scd.clock_s, len(deduped_words)))
                    # Trim history older than we'll ever need to look back, but
                    # always keep the newest entry at/before the cutoff so
                    # count_at() still has an answer for queries near it.
                    cutoff = scd.clock_s - (self.asr_decode_lag + self.scd_settle + 3.0)
                    while len(word_count_history) > 1 and word_count_history[1][0] <= cutoff:
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
                            # Real gap = skipped audio, excluding the pre-roll
                            # we are about to replay. Brief VAD flickers (and
                            # the VAD's forced max-speech cut) stay below
                            # scd_reset_gap and never reset the SCD window.
                            if scd:
                                pre_roll_s = sum(len(c) for c in pre_roll) / TARGET_RATE
                                if inactive_audio_s - pre_roll_s >= self.scd_reset_gap:
                                    scd.reset()
                            inactive_audio_s = 0.0
                            for chunk in pre_roll:
                                process_chunk(chunk)
                            pre_roll.clear()
                        process_chunk(samples)
                    else:
                        pre_roll.append(samples)
                        inactive_audio_s += samples.size / TARGET_RATE

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
            if scd is not None:
                print(f"[SCD] session stats: splits applied={changes_applied}, late changes={late_changes}")
                scd.close()
            if self.capture is not None:
                self.capture.stop()
            if logger is not None:
                logger.close()
            if error_message is None:
                self.status_changed.emit("Stopped")