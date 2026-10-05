"""
True incremental streaming ASR: a persistent OnlineStream fed audio
continuously while VAD says we're in a speech region. Uses the
RECOGNIZER'S OWN endpoint detector (not the VAD's segment logic) to decide
when a line is "done" - this is the fix for the original bug: latency was
previously bounded by silence duration, so uninterrupted speech (a lecture,
a call with no pauses) could run indefinitely before anything appeared.
rule3_min_utterance_length forces a line break after N seconds of
continuous speech even with zero pauses.

This exact is_ready/decode_stream/get_result/is_endpoint/reset pattern is
the one already validated working in your Phase 0 test (the CPU RTF=0.158
run) - only the rule1/rule2/rule3 endpoint-tuning kwargs below are new and
unverified against your installed version. If construction throws a
TypeError about an unexpected keyword, tell me the exact message and I'll
adjust the parameter names.
"""

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sherpa_onnx


@dataclass
class AsrUpdate:
    partial_text: str  # current in-progress text for this line ("" right after a finalize)
    finalized_text: Optional[str] = None  # non-None exactly when a line just finalized
    confidence: Optional[float] = None  # set only alongside finalized_text
    duration_seconds: Optional[float] = None  # audio duration of the finalized line


@dataclass
class WordTiming:
    """One word from the CURRENT in-progress line, reconstructed from the
    recognizer's per-token timestamps. start/end are seconds since this
    stream was last reset (same time base as AsrUpdate.duration_seconds).
    confidence is a per-word estimate (geometric mean of that word's token
    probabilities, same formula StreamingAsrEngine._estimate_confidence
    uses for a whole utterance) -- None if per-token probabilities weren't
    available. end is approximated as the START time of the NEXT word
    (sherpa-onnx only reports token START times, not durations); for the
    last word in the list, end falls back to that word's own last token's
    start time, which slightly understates it.
    """
    text: str
    start: float
    end: float
    confidence: Optional[float] = None


class StreamingAsrEngine:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        joiner: str,
        tokens: str,
        sample_rate: int = 16000,
        provider: str = "cpu",
        num_threads: int = 3,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
        overlap_seconds: float = 1.0,

    ):
        """
        rule1/rule2_min_trailing_silence: seconds of trailing silence before
            finalizing (rule2 applies once some speech has been decoded;
            it's the one that mostly matters here and is set lower than the
            sherpa-onnx default to reduce latency at natural pauses).
        rule3_min_utterance_length: force-finalize after this many seconds
            of CONTINUOUS speech, regardless of silence. This is what
            bounds latency during uninterrupted talking - the actual fix
            for the reported issue.
        """
        self.sample_rate = sample_rate
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=rule1_min_trailing_silence,
            rule2_min_trailing_silence=rule2_min_trailing_silence,
            rule3_min_utterance_length=rule3_min_utterance_length,
        )
        self._stream = self._recognizer.create_stream()
        self._samples_fed_since_reset = 0
        # Audio overlap state
        self._max_history_samples = int(overlap_seconds * sample_rate)
        self._audio_history = np.zeros(0, dtype=np.float32)



    def feed(self, samples: np.ndarray) -> AsrUpdate:
        """Feed one chunk of speech audio (caller is responsible for VAD
        gating - don't feed music/silence). Returns the current partial
        text, plus finalized_text if the endpoint detector fired."""
        self._stream.accept_waveform(self.sample_rate, samples)
        self._samples_fed_since_reset += len(samples)

        # Keep a rolling buffer of the most recent audio
        self._audio_history = np.concatenate([self._audio_history, samples])[-self._max_history_samples:]



        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        text = self._recognizer.get_result(self._stream).strip()

        if self._recognizer.is_endpoint(self._stream):
            finalized_text = text if text else None
            confidence = self._estimate_confidence(self._stream) if finalized_text else None
            duration = self._samples_fed_since_reset / self.sample_rate
            self._recognizer.reset(self._stream)
            self._samples_fed_since_reset = 0

                        
            # Immediately inject the overlapping audio tail into the fresh stream
            if len(self._audio_history) > 0:
                self._stream.accept_waveform(self.sample_rate, self._audio_history)
                self._samples_fed_since_reset += len(self._audio_history)
                # Clear history so we don't replay it again if another cut happens immediately
                self._audio_history = np.zeros(0, dtype=np.float32)


            return AsrUpdate(
                partial_text="",
                finalized_text=finalized_text,
                confidence=confidence,
                duration_seconds=duration if finalized_text else None,
            )

        return AsrUpdate(partial_text=text)

    def current_snapshot(self) -> AsrUpdate:
        """Best-effort look at the in-progress line WITHOUT finalizing or
        resetting the stream -- doesn't touch decode state at all beyond
        what feed() already did. Used to support UI-only line breaks (e.g.
        a detected speaker change) that must not disturb the continuous
        backend decode: see PipelineWorker.emit_ui_split().

        duration_seconds here is the total audio fed since the last REAL
        reset (same accounting as a normal finalize), not scoped to any
        UI-only split -- callers wanting a duration for just the audio
        since their last UI split should track the delta between
        successive calls themselves (PipelineWorker does this).
        """
        text = self._recognizer.get_result(self._stream).strip()
        confidence = self._estimate_confidence(self._stream) if text else None
        duration = self._samples_fed_since_reset / self.sample_rate if text else None
        return AsrUpdate(partial_text=text, confidence=confidence, duration_seconds=duration)

    def current_word_timings(self) -> Optional[list]:
        """Best-effort per-word (start, end, confidence) for the CURRENT
        in-progress line, reconstructed from the recognizer's per-token
        timestamps -- an alternative to current_snapshot()'s wall-clock
        approximation for UI-only line splitting (see
        PipelineWorker.emit_ui_split_token_based()). Returns a list of
        WordTiming, in the same order as text.split() would give.

        Returns None (the caller should fall back to a coarser method) if:
        - this sherpa-onnx build/result doesn't expose token timestamps
          under the keys this expects ("tokens", "timestamps");
        - the tokenizer doesn't use the SentencePiece '\u2581' word-boundary
          marker this assumes (most sherpa-onnx streaming Zipformer/
          Nemotron models do, but not guaranteed for every checkpoint) --
          detected via a sanity check against get_result()'s own word count,
          since a wrong tokenization convention would badly under- or
          over-count words.

        This is intentionally conservative: returning None and letting the
        caller fall back is much safer than silently returning a
        misaligned word list.
        """
        data = None
        for attr_name in ("get_result_as_json_string", "get_result_all"):
            method = getattr(self._recognizer, attr_name, None)
            if method is None:
                continue
            try:
                raw = method(self._stream)
                data = json.loads(raw) if isinstance(raw, str) else getattr(raw, "__dict__", None)
                if data:
                    break
            except Exception:
                data = None
                continue
        if not data:
            return None

        tokens = data.get("tokens")
        timestamps = data.get("timestamps")
        if not tokens or not timestamps or len(tokens) != len(timestamps):
            return None

        ys_probs = data.get("ys_probs")
        if ys_probs is not None and len(ys_probs) != len(tokens):
            ys_probs = None  # length mismatch -- don't trust it

        words: list = []
        piece_buf: list = []
        prob_buf: list = []
        word_start: Optional[float] = None

        def flush(end_time: float) -> None:
            if not piece_buf:
                return
            text = "".join(piece_buf).replace("\u2581", "").strip()
            if text:
                conf = float(np.exp(np.mean(prob_buf))) if prob_buf else None
                words.append(WordTiming(text=text, start=word_start, end=end_time, confidence=conf))
            piece_buf.clear()
            prob_buf.clear()

        for i, tok in enumerate(tokens):
            is_word_start = tok.startswith("\u2581") or i == 0
            if is_word_start and piece_buf:
                flush(end_time=timestamps[i])
                word_start = timestamps[i]
            elif word_start is None:
                word_start = timestamps[i]
            piece_buf.append(tok)
            if ys_probs is not None:
                prob_buf.append(ys_probs[i])
        flush(end_time=timestamps[-1] if timestamps else (word_start or 0.0))

        # Sanity check: bail out to fallback if the reconstructed word count
        # is way off from get_result()'s own word count -- a sign the '\u2581'
        # convention doesn't apply to this model's tokenizer.
        reference_text = self._recognizer.get_result(self._stream).strip()
        reference_word_count = len(reference_text.split())
        if reference_word_count and abs(len(words) - reference_word_count) > max(2, reference_word_count // 4):
            return None

        return words

    def force_finalize(self) -> Optional[AsrUpdate]:
        """Force finalize current speech segment immediately.

        NOTE: no longer called for speaker-change handling (see
        PipelineWorker.emit_ui_split() instead) -- resetting the stream
        here cuts off before the decoder has trailing context to commit
        its last word(s), which is what was dropping words at each speaker
        change. Left in place in case you have another use for a genuine
        forced backend cut; just be aware of that tradeoff if you call it.
        """
        text = self._recognizer.get_result(self._stream).strip()
        if not text:
            self._recognizer.reset(self._stream)
            self._samples_fed_since_reset = 0
            self._audio_history = np.zeros(0, dtype=np.float32)
            return None

        confidence = self._estimate_confidence(self._stream)
        duration = self._samples_fed_since_reset / self.sample_rate

        self._recognizer.reset(self._stream)
        self._samples_fed_since_reset = 0
        # Discard history so previous speaker's tail doesn't overlap into new speaker
        self._audio_history = np.zeros(0, dtype=np.float32)

        return AsrUpdate(
            partial_text="",
            finalized_text=text,
            confidence=confidence,
            duration_seconds=duration,
        )

    def flush(self) -> Optional[AsrUpdate]:
        """Call at shutdown to force out any in-progress line."""
        if self._samples_fed_since_reset == 0:
            return None
        tail = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        self._stream.accept_waveform(self.sample_rate, tail)
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._recognizer.get_result(self._stream).strip()
        if not text:
            return None
        confidence = self._estimate_confidence(self._stream)
        duration = self._samples_fed_since_reset / self.sample_rate
        return AsrUpdate(
            partial_text="",
            finalized_text=text,
            confidence=confidence,
            duration_seconds=duration,
        )

    def _estimate_confidence(self, stream) -> Optional[float]:
        """Same best-effort extraction as Phase 3 - already confirmed
        working against your installed version."""
        for attr_name in ("get_result_as_json_string", "get_result_all"):
            method = getattr(self._recognizer, attr_name, None)
            if method is None:
                continue
            try:
                raw = method(stream)
                data = json.loads(raw) if isinstance(raw, str) else getattr(raw, "__dict__", None)
                if data and data.get("ys_probs"):
                    probs = np.array(data["ys_probs"], dtype=np.float64)
                    return float(np.exp(np.mean(probs)))
            except Exception:
                continue
        return None