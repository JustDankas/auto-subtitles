"""
speaker_tuning_lib.py

Reusable pieces for tuning the InstantChangeDetector's parameters
(speaker-k, std-floor, min-window, max-window) against synthetic
multi-speaker sessions built from single-speaker clips.

Design choice: this evaluates the change detector IN ISOLATION, given
perfect segment boundaries (the true clip boundaries), not the app's
VAD. That deliberately decouples "does the detector correctly flip on
a genuine speaker change" from "does the VAD correctly cut a segment
in the right place." VAD tuning is a separate problem; see MANUAL.md.

A consequence: gap=0.0 sessions (no silence between speakers) are
included here as a stress test, but in the real app a VAD would very
likely NOT split two back-to-back speakers with no pause into two
segments in the first place, so the detector would never even get a
chance to fire. Treat gap=0.0 results as an upper bound on what the
detector could do if it got a clean segment boundary to work with,
not as a promise of real-world performance on fast back-and-forth
speech.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------
# Audio / dataset loading
# --------------------------------------------------------------------------

def discover_speaker_clips(dataset_dir: str) -> dict[str, list[str]]:
    """dataset_dir/<speaker_id>/*.wav  ->  {speaker_id: [wav paths]}"""
    speaker_clips: dict[str, list[str]] = {}
    for speaker_dir in sorted(glob.glob(os.path.join(dataset_dir, "*"))):
        if not os.path.isdir(speaker_dir):
            continue
        wavs = sorted(glob.glob(os.path.join(speaker_dir, "*.wav")))
        if wavs:
            speaker_clips[os.path.basename(speaker_dir)] = wavs
    return speaker_clips


def load_wav_mono16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr} Hz. "
            "Resample your dataset before tuning (see MANUAL.md)."
        )
    return audio


# --------------------------------------------------------------------------
# Synthetic session construction
# --------------------------------------------------------------------------

@dataclass
class SyntheticSegment:
    audio: np.ndarray
    speaker_id: str
    true_change: bool  # True if this segment's speaker differs from the previous one


@dataclass
class SyntheticSession:
    name: str
    segments: list[SyntheticSegment]
    gap_seconds: float


def build_synthetic_session(
    speaker_clips: dict[str, list[str]],
    name: str,
    num_speakers: int,
    gap_seconds: float,
    turns_per_speaker: int = 3,
    seed: int = 0,
    run_length_range: tuple[int, int] = (1, 3),
) -> SyntheticSession:
    """
    Picks `num_speakers` speakers and builds a session as a sequence of
    speaker RUNS: each run is 1..run_length_range[1] consecutive segments
    from the same speaker (using distinct clips), followed by a switch to
    a different speaker. `turns_per_speaker` is a rough budget on how many
    total segments each speaker contributes across the session, not a
    literal turn count.

    Why runs, not strict alternation: InstantChangeDetector needs several
    consecutive same-speaker segments to build a stable baseline (window
    mean/std) before it can tell a genuine change apart from normal voice
    variation. A session where every segment is a different speaker (the
    old behavior of this function) gives the detector zero chance to ever
    stabilize -- true changes arrive faster than the window can fill, which
    caps recall regardless of how k/std_floor are tuned. It's also not
    representative of the app's real input: a VAD very often cuts one
    person's continuous turn into 2-4 separate segments (short mid-sentence
    pauses), so most segment-to-segment transitions in production are NOT
    speaker changes. run_length_range=(1, 3) means about 1 in 2 segment
    boundaries is a true change on average -- adjust if you want a harder
    (shorter runs) or easier (longer runs) mix.

    gap_seconds is recorded on the session but NOT physically inserted
    as silence into any single returned array here -- each segment stays
    a separate array, matching "one segment = one embedding" in the app.
    The gap value is metadata used only to report results and to build
    the "hard mode" (gap=0.0) stress sessions.
    """
    rng = random.Random(seed)
    available = [s for s in speaker_clips if len(speaker_clips[s]) >= 2]
    if len(available) < num_speakers:
        raise ValueError(
            f"Need {num_speakers} speakers with >= 2 clips each, "
            f"only have {len(available)}. See MANUAL.md for dataset size."
        )
    chosen = rng.sample(available, k=num_speakers)
    # Shuffle each chosen speaker's own clip list so consecutive same-speaker
    # segments within a run use distinct clips, not the same clip repeated.
    shuffled_clips = {s: rng.sample(speaker_clips[s], k=len(speaker_clips[s])) for s in chosen}
    clip_idx = {s: 0 for s in chosen}

    total_budget = turns_per_speaker * num_speakers
    segments: list[SyntheticSegment] = []
    prev_speaker = None
    total = 0
    while total < total_budget:
        candidates = [s for s in chosen if s != prev_speaker] if prev_speaker else chosen
        spk = rng.choice(candidates)
        run_len = rng.randint(*run_length_range)
        for _ in range(run_len):
            if total >= total_budget:
                break
            clips = shuffled_clips[spk]
            idx = clip_idx[spk] % len(clips)
            clip_idx[spk] += 1
            audio = load_wav_mono16k(clips[idx])
            segments.append(
                SyntheticSegment(
                    audio=audio,
                    speaker_id=spk,
                    true_change=(prev_speaker is not None and spk != prev_speaker),
                )
            )
            prev_speaker = spk
            total += 1

    return SyntheticSession(name=name, segments=segments, gap_seconds=gap_seconds)


def build_session_suite(
    speaker_clips: dict[str, list[str]],
    seed: int = 0,
) -> list[SyntheticSession]:
    """
    The default suite referenced in MANUAL.md: varies speaker count and
    gap. Gap is metadata here (see docstring above); it exists so you
    can filter sessions by difficulty when reporting results.
    """
    sessions = []
    # (num_speakers, turns_per_speaker, gap_seconds, n_sessions, run_length_range)
    configs = [
        (2, 6, 0.5, 3, (1, 3)),
        (4, 5, 0.5, 3, (1, 3)),
        (6, 4, 0.5, 3, (1, 3)),
        (2, 6, 0.2, 2, (1, 3)),   # harder: short natural pause
        (4, 5, 0.2, 2, (1, 3)),
        (2, 6, 0.0, 2, (1, 3)),   # stress test only, see module docstring
        # Long same-speaker runs (monologues/interview turns). These exist
        # specifically to give --speaker-max-window room to matter and to
        # test false-positive suppression across a sustained single-speaker
        # stretch, neither of which the short-run configs above can test
        # (their runs never exceed 3 segments, so a window cap above 3 is
        # never actually exercised).
        (2, 10, 0.5, 2, (4, 8)),
        (3, 9, 0.5, 2, (4, 8)),
    ]
    session_id = 0
    for num_speakers, turns, gap, n, run_range in configs:
        for i in range(n):
            sessions.append(
                build_synthetic_session(
                    speaker_clips,
                    name=f"spk{num_speakers}_gap{gap}_run{run_range[0]}-{run_range[1]}_{i}",
                    num_speakers=num_speakers,
                    gap_seconds=gap,
                    turns_per_speaker=turns,
                    run_length_range=run_range,
                    seed=seed + session_id,
                )
            )
            session_id += 1
    return sessions


# --------------------------------------------------------------------------
# Embedding extraction (sherpa-onnx)
# --------------------------------------------------------------------------

class EmbeddingExtractor:
    """Thin wrapper around sherpa_onnx.SpeakerEmbeddingExtractor.

    Import of sherpa_onnx is deferred to __init__ so this module can be
    imported (e.g. for unit-testing build_synthetic_session) without
    sherpa-onnx installed.
    """

    def __init__(self, model_path: str, num_threads: int = 1, provider: str = "cpu"):
        import sherpa_onnx  # deferred import

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            num_threads=num_threads,
            provider=provider,
        )
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dim = self.extractor.dim

    def compute(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=audio)
        stream.input_finished()
        emb = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


def embed_session(session: SyntheticSession, extractor: EmbeddingExtractor) -> list[np.ndarray]:
    return [extractor.compute(seg.audio) for seg in session.segments]


def embed_and_cache_suite(
    sessions: list[SyntheticSession],
    extractor: EmbeddingExtractor,
    cache_path: str | None = None,
) -> dict[str, list[np.ndarray]]:
    """Embeds every session once and optionally caches to .npz so repeated
    grid searches don't re-run the (slow-ish, model-load-heavy) extractor.
    Embeddings are cheap per your Phase 0 numbers, but re-running the
    extractor process N times during a grid search is not necessary --
    embed once, then grid-search purely on cached numpy arrays.
    """
    cache: dict[str, list[np.ndarray]] = {}
    for session in sessions:
        cache[session.name] = embed_session(session, extractor)

    if cache_path:
        flat = {}
        for name, embs in cache.items():
            flat[name] = np.stack(embs)
        np.savez(cache_path, **flat)
    return cache


def load_cached_embeddings(cache_path: str) -> dict[str, list[np.ndarray]]:
    data = np.load(cache_path)
    return {name: [row for row in data[name]] for name in data.files}


# --------------------------------------------------------------------------
# The detector under test
# --------------------------------------------------------------------------

class InstantChangeDetector:
    def __init__(self, k: float = 2.5, std_floor: float = 0.05,
                 min_window: int = 3, max_window: int = 20,
                 min_active_seconds: float = 0.0):
        """
        min_active_seconds: in addition to min_window (a segment COUNT),
        require at least this many seconds of audio to have accumulated in
        the current window before a change can be detected. min_window
        alone is a poor proxy for "the preceding speaker talked for at
        least N seconds" whenever segment durations vary -- two 0.4s
        segments satisfy min_window=2 while covering under a second of
        real speech. Set to 0.0 (default) to disable and rely on
        min_window alone, matching the original behavior.
        """
        self.k = k
        self.std_floor = std_floor
        self.min_window = min_window
        self.max_window = max_window
        self.min_active_seconds = min_active_seconds
        self.window: list[np.ndarray] = []
        self.window_durations: list[float] = []

    def reset(self):
        self.window = []
        self.window_durations = []

    def process(self, emb: np.ndarray, duration_sec: float = 0.0) -> bool:
        enough_count = len(self.window) >= self.min_window
        enough_duration = (self.min_active_seconds <= 0.0) or \
            (sum(self.window_durations) >= self.min_active_seconds)

        if not (enough_count and enough_duration):
            self.window.append(emb)
            self.window_durations.append(duration_sec)
            return False

        centroid = np.mean(self.window, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm

        sims = np.array([float(np.dot(centroid, e)) for e in self.window])
        mu, sigma = float(sims.mean()), max(float(sims.std()), self.std_floor)
        cur_sim = float(np.dot(centroid, emb))
        z = (mu - cur_sim) / sigma

        if z > self.k:
            self.window = [emb]
            self.window_durations = [duration_sec]
            return True

        self.window.append(emb)
        self.window_durations.append(duration_sec)
        if len(self.window) > self.max_window:
            self.window.pop(0)
            self.window_durations.pop(0)
        return False


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@dataclass
class SessionResult:
    session_name: str
    gap_seconds: float
    true_labels: list[bool]
    pred_labels: list[bool]


def run_detector_on_session(
    session: SyntheticSession,
    embeddings: list[np.ndarray],
    k: float, std_floor: float, min_window: int, max_window: int,
    min_active_seconds: float = 0.0,
) -> SessionResult:
    detector = InstantChangeDetector(k=k, std_floor=std_floor, min_window=min_window,
                                      max_window=max_window, min_active_seconds=min_active_seconds)
    durations = [len(seg.audio) / SAMPLE_RATE for seg in session.segments]
    true_labels, pred_labels = [], []
    # Feed segment 0 in to seed the window (matches production: the detector
    # sees every segment in order) but don't score it -- there's no
    # "previous speaker" for the first segment to have changed from.
    detector.process(embeddings[0], duration_sec=durations[0])
    for seg, emb, dur in zip(session.segments[1:], embeddings[1:], durations[1:]):
        pred = detector.process(emb, duration_sec=dur)
        true_labels.append(seg.true_change)
        pred_labels.append(pred)
    return SessionResult(session.name, session.gap_seconds, true_labels, pred_labels)


def prf_beta(true_labels: list[bool], pred_labels: list[bool], beta: float = 1.5):
    tp = sum(t and p for t, p in zip(true_labels, pred_labels))
    fp = sum((not t) and p for t, p in zip(true_labels, pred_labels))
    fn = sum(t and (not p) for t, p in zip(true_labels, pred_labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    fbeta = (
        (1 + b2) * precision * recall / (b2 * precision + recall)
        if (precision + recall) else 0.0
    )
    return {"precision": precision, "recall": recall, "fbeta": fbeta,
            "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------
# Grid search
# --------------------------------------------------------------------------

@dataclass
class GridPoint:
    k: float
    std_floor: float
    min_window: int
    max_window: int
    mean_fbeta: float
    min_fbeta: float          # worst session -- stability check
    mean_precision: float
    mean_recall: float
    min_active_seconds: float = 0.0
    per_session: dict = field(default_factory=dict)


def evaluate_params(
    sessions: list[SyntheticSession],
    embeddings_by_session: dict[str, list[np.ndarray]],
    k: float, std_floor: float, min_window: int, max_window: int,
    min_active_seconds: float = 0.0,
    beta: float = 1.5,
    include_gap_zero: bool = False,
) -> GridPoint:
    fbetas, precisions, recalls, per_session = [], [], [], {}
    for session in sessions:
        if not include_gap_zero and session.gap_seconds == 0.0:
            continue
        result = run_detector_on_session(
            session, embeddings_by_session[session.name],
            k, std_floor, min_window, max_window, min_active_seconds=min_active_seconds,
        )
        metrics = prf_beta(result.true_labels, result.pred_labels, beta=beta)
        fbetas.append(metrics["fbeta"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        per_session[session.name] = metrics

    return GridPoint(
        k=k, std_floor=std_floor, min_window=min_window, max_window=max_window,
        mean_fbeta=float(np.mean(fbetas)) if fbetas else 0.0,
        min_fbeta=float(np.min(fbetas)) if fbetas else 0.0,
        mean_precision=float(np.mean(precisions)) if precisions else 0.0,
        mean_recall=float(np.mean(recalls)) if recalls else 0.0,
        min_active_seconds=min_active_seconds,
        per_session=per_session,
    )


def grid_search(
    train_sessions: list[SyntheticSession],
    embeddings_by_session: dict[str, list[np.ndarray]],
    k_grid: list[float],
    std_floor_grid: list[float],
    window_grid: list[tuple[int, int]],
    min_active_seconds_grid: list[float] = (0.0,),
    beta: float = 1.5,
) -> list[GridPoint]:
    results = []
    for k, std_floor, (min_w, max_w), min_active in itertools.product(
        k_grid, std_floor_grid, window_grid, min_active_seconds_grid
    ):
        results.append(
            evaluate_params(train_sessions, embeddings_by_session, k, std_floor, min_w, max_w,
                             min_active_seconds=min_active, beta=beta)
        )
    results.sort(key=lambda gp: gp.mean_precision, reverse=True)
    return results


def train_val_split(sessions: list[SyntheticSession], val_fraction: float = 0.3, seed: int = 0):
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def results_table(results: list[GridPoint], top_n: int = 10) -> str:
    lines = [f"{'k':>5} {'std_floor':>10} {'min_w':>6} {'max_w':>6} {'min_sec':>8} "
             f"{'mean_F':>8} {'min_F':>8} {'prec':>6} {'rec':>6}"]
    for gp in results[:top_n]:
        lines.append(
            f"{gp.k:>5.2f} {gp.std_floor:>10.3f} {gp.min_window:>6} {gp.max_window:>6} "
            f"{gp.min_active_seconds:>8.2f} "
            f"{gp.mean_fbeta:>8.3f} {gp.min_fbeta:>8.3f} "
            f"{gp.mean_precision:>6.3f} {gp.mean_recall:>6.3f}"
        )
    return "\n".join(lines)


def save_results_json(results: list[GridPoint], path: str):
    payload = [
        {
            "k": gp.k, "std_floor": gp.std_floor,
            "min_window": gp.min_window, "max_window": gp.max_window,
            "min_active_seconds": gp.min_active_seconds,
            "mean_fbeta": gp.mean_fbeta, "min_fbeta": gp.min_fbeta,
            "mean_precision": gp.mean_precision, "mean_recall": gp.mean_recall,
            "per_session": gp.per_session,
        }
        for gp in results
    ]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
