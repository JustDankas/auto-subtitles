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
