import argparse
import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 16000


@dataclass
class GroundTruthTurn:
    file_id: str
    start_s: float
    end_s: float
    speaker_id: str
    has_overlap: bool = False
    noise_level: str = "zero"  # "zero", "mild", or "moderate"
    notes: str = ""


@dataclass
class SyntheticSessionResult:
    session_name: str
    audio_data: np.ndarray
    sample_rate: int
    noise_level: str
    turns: List[GroundTruthTurn]


def normalize_lufs(audio: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    """
    Simple RMS-based loudness normalization to ensure smooth volume transitions
    between concatenated clips from different sources/speakers.
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    current_db = 20 * np.log10(rms + 1e-9)
    gain = 10 ** ((target_db - current_db) / 20)
    return audio * gain


def generate_ambient_noise(num_samples: int, level_db: float = -45.0) -> np.ndarray:
    """
    Generates subtle continuous pink/white noise across the entire recording track
    to simulate a unified room acoustic floor across all transitions.
    """
    noise = np.random.normal(0, 1.0, num_samples)
    rms = np.sqrt(np.mean(noise**2))
    gain = 10 ** ((level_db - 20 * np.log10(rms + 1e-9)) / 20)
    return noise * gain


NOISE_DB_MAP = {"zero": None, "mild": -60.0, "moderate": -45.0}


def apply_background_noise(canvas: np.ndarray, noise_level: str) -> np.ndarray:
    target_db = NOISE_DB_MAP.get(noise_level)
    if target_db is not None:
        canvas = canvas + generate_ambient_noise(len(canvas), level_db=target_db)
    return canvas


def prevent_clipping(canvas: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(canvas))
    if max_val > 0.99:
        canvas = canvas / max_val * 0.99
    return canvas


def load_wav_mono16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz in {path}.")
    return audio


def derive_change_points(turns: List[GroundTruthTurn]) -> List[float]:
    """A change point is any turn whose speaker differs from the previous
    turn, sorted by start_s -- this already covers overlap onsets, since an
    overlapping turn is just another row with a different speaker_id."""
    ordered = sorted(turns, key=lambda t: t.start_s)
    points = []
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.speaker_id != prev.speaker_id:
            points.append(cur.start_s)
    return points


def export_frame_labels(
    turns: List[GroundTruthTurn],
    duration_s: float,
    frame_hop_ms: float = 10.0,
    tolerance_ms: float = 150.0,
    use_soft_labels: bool = True,
    std_scale: float = 0.5,
) -> np.ndarray:
    """
    Generate per-frame label array at `frame_hop_ms` resolution. Unchanged
    from the session-based pipeline -- it only needs a list of turns, so the
    new fixed-window generator below reuses it as-is by constructing a
    2-turn (or 1-turn, for no-change windows) turn list per window.
    """
    n_frames = int(np.ceil(duration_s * 1000.0 / frame_hop_ms))
    frame_times_s = np.arange(n_frames) * (frame_hop_ms / 1000.0)
    labels = np.zeros(n_frames, dtype=np.float32)

    change_points = derive_change_points(turns)
    tol_s = tolerance_ms / 1000.0

    if use_soft_labels:
        sigma = tol_s * std_scale
        for cp in change_points:
            dist = np.abs(frame_times_s - cp)
            mask = dist <= tol_s
            gaussian_vals = np.exp(-0.5 * (dist[mask] / sigma) ** 2)
            labels[mask] = np.maximum(labels[mask], gaussian_vals)
    else:
        for cp in change_points:
            labels[np.abs(frame_times_s - cp) <= tol_s] = 1.0

    return labels


def _save_window(
    output_dir: str,
    name: str,
    audio: np.ndarray,
    turns: List[GroundTruthTurn],
    window_duration_s: float,
    frame_hop_ms: float,
    label_tolerance_ms: float,
) -> None:
    # subtype="FLOAT" avoids the default PCM_16 write path silently clipping
    # anything outside [-1, 1] -- not expected to matter here since
    # prevent_clipping() already normalizes, but it's free precision to keep.
    sf.write(os.path.join(output_dir, "audio", f"{name}.wav"), audio, SAMPLE_RATE, subtype="FLOAT")
    labels = export_frame_labels(
        turns, window_duration_s, frame_hop_ms=frame_hop_ms, tolerance_ms=label_tolerance_ms
    )
    np.save(os.path.join(output_dir, "labels", f"{name}.npy"), labels)


# ---------------------------------------------------------------------------
# FIXED-DURATION SINGLE-CHANGE (OR NO-CHANGE) WINDOW GENERATION
# ---------------------------------------------------------------------------
def concat_clips_to_duration(
    clips: List[str],
    duration_s: float,
    start_idx: int,
    fade_ms: float = 5.0,
) -> Tuple[np.ndarray, int]:
    """
    Concatenate consecutive clips from a speaker's list (rotating through it,
    wrapping around as needed) until there's at least `duration_s` seconds,
    then trim to the exact length. Applies a short amplitude taper at each
    clip boundary to avoid audible clicks at the splice (a simple edge fade,
    not a true overlap-add crossfade -- plenty for feature-extraction
    purposes here). Returns (audio, next_start_idx) so callers can keep
    rotating a speaker's cursor forward across many windows instead of
    always restarting at clip 0 (which would make every window for a given
    speaker start identically).
    """
    target_samples = int(round(duration_s * SAMPLE_RATE))
    fade_samples = int(SAMPLE_RATE * fade_ms / 1000.0)
    out = np.zeros(0, dtype=np.float32)
    idx = start_idx
    attempts = 0
    max_attempts = len(clips) * 5 + 10
    while len(out) < target_samples and attempts < max_attempts:
        clip = normalize_lufs(load_wav_mono16k(clips[idx % len(clips)]))
        if len(out) > 0 and fade_samples > 0 and len(clip) > fade_samples:
            fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            clip = clip.copy()
            clip[:fade_samples] *= fade
            out[-fade_samples:] *= (1.0 - fade)
        out = np.concatenate([out, clip])
        idx += 1
        attempts += 1
    if len(out) < target_samples:
        raise ValueError(
            f"Could not gather {duration_s:.2f}s of audio from a {len(clips)}-clip "
            f"list (only got {len(out) / SAMPLE_RATE:.2f}s) -- add more/longer clips "
            "for this speaker."
        )
    return out[:target_samples], idx


def build_change_window(
    speaker_clips: Dict[str, List[str]],
    speaker_a: str,
    speaker_b: str,
    window_duration_s: float,
    change_time_s: float,
    gap_seconds: float,
    noise_level: str,
    clip_cursor: Dict[str, int],
) -> Tuple[np.ndarray, List[GroundTruthTurn]]:
    """
    One window containing exactly one speaker change at `change_time_s`.
    `gap_seconds` shifts the two speakers relative to that change point
    exactly like the session generator's gap semantics: 0 = hard cut,
    >0 = silence gap (A ends before change_time_s), <0 = overlap (A extends
    past change_time_s). `change_time_s` is always the label -- it's B's
    (the new speaker's) conceptual start, matching derive_change_points'
    definition used everywhere else in this file.
    """

    a_end_s = change_time_s - gap_seconds
    b_dur_s = window_duration_s - change_time_s
    if a_end_s <= 0 or b_dur_s <= 0:
        raise ValueError(
            f"change_time_s={change_time_s} / gap_seconds={gap_seconds} leaves no "
            f"room for both speakers in a {window_duration_s}s window"
        )

    a_audio, clip_cursor[speaker_a] = concat_clips_to_duration(
        speaker_clips[speaker_a], a_end_s, clip_cursor.get(speaker_a, 0)
    )
    b_audio, clip_cursor[speaker_b] = concat_clips_to_duration(
        speaker_clips[speaker_b], b_dur_s, clip_cursor.get(speaker_b, 0)
    )

    n_total = int(round(window_duration_s * SAMPLE_RATE))
    canvas = np.zeros(n_total, dtype=np.float32)
    canvas[: len(a_audio)] += a_audio
    b_start_sample = int(round(change_time_s * SAMPLE_RATE))
    b_end_sample = min(b_start_sample + len(b_audio), n_total)
    canvas[b_start_sample:b_end_sample] += b_audio[: b_end_sample - b_start_sample]

    canvas = apply_background_noise(canvas, noise_level)
    canvas = prevent_clipping(canvas)

    turns = [
        GroundTruthTurn(
            file_id="", start_s=0.0, end_s=round(len(a_audio) / SAMPLE_RATE, 3),
            speaker_id=speaker_a, has_overlap=False, noise_level=noise_level,
            notes=f"gap={gap_seconds:.2f}s",
        ),
        GroundTruthTurn(
            file_id="", start_s=round(change_time_s, 3), end_s=round(window_duration_s, 3),
            speaker_id=speaker_b, has_overlap=(gap_seconds < 0.0), noise_level=noise_level,
            notes=f"gap={gap_seconds:.2f}s",
        ),
    ]
    return canvas, turns


def build_no_change_window(
    speaker_clips: Dict[str, List[str]],
    speaker: str,
    window_duration_s: float,
    noise_level: str,
    clip_cursor: Dict[str, int],
) -> Tuple[np.ndarray, List[GroundTruthTurn]]:
    audio, clip_cursor[speaker] = concat_clips_to_duration(
        speaker_clips[speaker], window_duration_s, clip_cursor.get(speaker, 0)
    )
    n_total = int(round(window_duration_s * SAMPLE_RATE))
    canvas = apply_background_noise(audio[:n_total].copy(), noise_level)
    canvas = prevent_clipping(canvas)
    turns = [
        GroundTruthTurn(
            file_id="", start_s=0.0, end_s=round(window_duration_s, 3),
            speaker_id=speaker, has_overlap=False, noise_level=noise_level, notes="no_change",
        )
    ]
    return canvas, turns

# ---------------------------------------------------------------------------
# WORKER FUNCTIONS FOR PARALLEL EXECUTION
# ---------------------------------------------------------------------------

def _process_change_window_job(job_args: dict) -> List[dict]:
    """
    Worker task: generates and saves a single change window.
    """
    # Unpack parameters
    window_id = job_args["window_id"]
    output_dir = job_args["output_dir"]
    speaker_clips = job_args["speaker_clips"]
    speaker_a = job_args["speaker_a"]
    speaker_b = job_args["speaker_b"]
    window_duration_s = job_args["window_duration_s"]
    min_segment_s = job_args["min_segment_s"]
    change_position_step_s = job_args["change_position_step_s"]
    change_position_jitter_s = job_args["change_position_jitter_s"]
    gap_options = job_args["gap_options"]
    noise_options = job_args["noise_options"]
    noise_probs = job_args["noise_probs"]
    frame_hop_ms = job_args["frame_hop_ms"]
    label_tolerance_ms = job_args["label_tolerance_ms"]
    clip_cursors = job_args["clip_cursors"]
    worker_seed = job_args["seed"]

    rng = random.Random(worker_seed)
    
    positions = np.arange(
        min_segment_s, window_duration_s - min_segment_s + 1e-9, change_position_step_s
    )

    manifest_rows = []
    
    for idx, base_pos in enumerate(positions):
        sub_id = f"{window_id:06d}_{idx}"
        jitter = rng.uniform(-change_position_jitter_s, change_position_jitter_s)
        change_time_s = float(
            np.clip(base_pos + jitter, min_segment_s, window_duration_s - min_segment_s)
        )
        gap_seconds = rng.choice(gap_options)
        noise_level = rng.choices(noise_options, weights=noise_probs, k=1)[0]

        name = f"win_change_{sub_id}"
        audio, turns = build_change_window(
            speaker_clips, speaker_a, speaker_b, window_duration_s,
            change_time_s, gap_seconds, noise_level, clip_cursors,
        )
        for t in turns:
            t.file_id = name
        _save_window(output_dir, name, audio, turns, window_duration_s, frame_hop_ms, label_tolerance_ms)
        manifest_rows.extend([t.__dict__ for t in turns])

    return manifest_rows


def _process_no_change_window_job(job_args: dict) -> List[dict]:
    """
    Worker task: generates and saves a single no-change window.
    """
    window_id = job_args["window_id"]
    output_dir = job_args["output_dir"]
    speaker_clips = job_args["speaker_clips"]
    speaker = job_args["speaker"]
    window_duration_s = job_args["window_duration_s"]
    noise_options = job_args["noise_options"]
    noise_probs = job_args["noise_probs"]
    frame_hop_ms = job_args["frame_hop_ms"]
    label_tolerance_ms = job_args["label_tolerance_ms"]
    clip_cursor = job_args["clip_cursor"]
    worker_seed = job_args["seed"]

    rng = random.Random(worker_seed)
    noise_level = rng.choices(noise_options, weights=noise_probs, k=1)[0]
    name = f"win_nochange_{window_id:06d}"

    # Minimal cursor state dictionary for single-speaker processing
    cursor_dict = {speaker: clip_cursor}
    audio, turns = build_no_change_window(
        speaker_clips, speaker, window_duration_s, noise_level, cursor_dict
    )
    for t in turns:
        t.file_id = name
    _save_window(output_dir, name, audio, turns, window_duration_s, frame_hop_ms, label_tolerance_ms)
    
    return [t.__dict__ for t in turns]

# ---------------------------------------------------------------------------
# MAIN PARALLEL GENERATOR
# ---------------------------------------------------------------------------

def generate_window_dataset(
    speaker_clips: Dict[str, List[str]],
    output_dir: str,
    window_duration_s: float = 3.0,
    min_segment_s: float = 1.0,
    change_position_step_s: float = 0.5,
    change_position_jitter_s: float = 0.15,
    change_fraction: float = 0.5,
    gap_options: Tuple[float, ...] = (-0.2, -0.1, 0.0, 0.0, 0.0, 0.1, 0.2),
    noise_options: Tuple[str, ...] = ("zero", "mild", "moderate"),
    noise_probs: Tuple[float, ...] = (0.5, 0.25, 0.25),
    num_pairs: int = 300,
    frame_hop_ms: float = 10.0,
    label_tolerance_ms: float = 150.0,
    seed: int = 42,
    num_workers: int = None,  # Defaults to CPU core count
) -> None:
    rng = random.Random(seed)
    speakers = sorted([s for s in speaker_clips if len(speaker_clips[s]) >= 1])
    if len(speakers) < 2:
        raise ValueError("Need at least 2 speakers with clips to build change windows")

    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    unique_speaker_pairs = list(itertools.combinations(speakers, 2))
    rng.shuffle(unique_speaker_pairs)
    pair_pool = itertools.cycle(unique_speaker_pairs)

    # Pre-calculate jobs to pass to workers
    clip_cursors: Dict[str, int] = {s: 0 for s in speakers}
    change_jobs = []

    for i in range(num_pairs):
        speaker_a, speaker_b = next(pair_pool)
        if rng.random() > 0.5:
            speaker_a, speaker_b = speaker_b, speaker_a

        job_args = {
            "window_id": i,
            "output_dir": output_dir,
            "speaker_clips": speaker_clips,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "window_duration_s": window_duration_s,
            "min_segment_s": min_segment_s,
            "change_position_step_s": change_position_step_s,
            "change_position_jitter_s": change_position_jitter_s,
            "gap_options": gap_options,
            "noise_options": noise_options,
            "noise_probs": noise_probs,
            "frame_hop_ms": frame_hop_ms,
            "label_tolerance_ms": label_tolerance_ms,
            "clip_cursors": {
                speaker_a: clip_cursors[speaker_a],
                speaker_b: clip_cursors[speaker_b],
            },
            "seed": seed + i,
        }
        
        # Advance local clip offsets deterministicly
        clip_cursors[speaker_a] += 2
        clip_cursors[speaker_b] += 2
        change_jobs.append(job_args)

    # Determine total no-change windows needed
    positions_per_pair = len(
        np.arange(min_segment_s, window_duration_s - min_segment_s + 1e-9, change_position_step_s)
    )
    total_change_windows = num_pairs * positions_per_pair
    num_no_change = int(round(total_change_windows * (1 - change_fraction) / max(change_fraction, 1e-9)))

    no_change_jobs = []
    speaker_pool = itertools.cycle(speakers)
    
    for i in range(num_no_change):
        speaker = next(speaker_pool)
        job_args = {
            "window_id": i,
            "output_dir": output_dir,
            "speaker_clips": speaker_clips,
            "speaker": speaker,
            "window_duration_s": window_duration_s,
            "noise_options": noise_options,
            "noise_probs": noise_probs,
            "frame_hop_ms": frame_hop_ms,
            "label_tolerance_ms": label_tolerance_ms,
            "clip_cursor": clip_cursors[speaker],
            "seed": seed + num_pairs + i,
        }
        clip_cursors[speaker] += 2
        no_change_jobs.append(job_args)

    manifest_rows = []

    # Calculate exact total change windows for tqdm
    total_change_windows = num_pairs * positions_per_pair
    
    # Run Change Windows in Parallel
    print(f"Generating change windows using ProcessPoolExecutor ({num_workers or 'all'} cores)...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_change_window_job, job) for job in change_jobs]
        with tqdm(total=total_change_windows, desc="Change Windows") as pbar:
            for future in as_completed(futures):
                rows = future.result()
                manifest_rows.extend(rows)
                pbar.update(positions_per_pair)

    # Run No-Change Windows in Parallel
    print(f"Generating no-change windows...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_no_change_window_job, job) for job in no_change_jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="No-Change Windows"):
            manifest_rows.extend(future.result())

    df = pd.DataFrame(manifest_rows)
    df.to_csv(os.path.join(output_dir, "annotations", "manifest.csv"), index=False)
    print(f"Done! Generated dataset in '{output_dir}'.")

# def generate_window_dataset(
#     speaker_clips: Dict[str, List[str]],
#     output_dir: str,
#     window_duration_s: float = 3.0,
#     min_segment_s: float = 0.5,
#     change_position_step_s: float = 0.5,
#     change_position_jitter_s: float = 0.15,
#     change_fraction: float = 0.5,
#     gap_options: Tuple[float, ...] = (-0.2, -0.1, 0.0, 0.0, 0.0, 0.1, 0.2),
#     noise_options: Tuple[str, ...] = ("zero", "mild", "moderate"),
#     noise_probs: Tuple[float, ...] = (0.5, 0.25, 0.25),
#     num_pairs: int = 300,
#     frame_hop_ms: float = 10.0,
#     label_tolerance_ms: float = 150.0,
#     seed: int = 42,
# ) -> None:
#     """
#     Replaces the multi-turn session generator for classifier training: every
#     output file is a standalone `window_duration_s`-long clip containing
#     *at most one* speaker change, so there is never a risk of two changes
#     confusing the model within a single training example.

#     For each of `num_pairs` speaker pairs, generates one window per position
#     in `change_position_step_s`-spaced steps across the window (with small
#     random jitter) -- e.g. for a 3.0s window, 0.5s min_segment and 0.5s step:
#     change points at roughly 0.5, 1.0, 1.5, 2.0, 2.5s. This deliberately
#     covers the realistic range of "how long ago did the change happen"
#     a trailing window sees in production, instead of only ever training on
#     a change sitting at the window's midpoint.

#     `change_fraction` controls what fraction of the final dataset has a
#     change at all; the remainder are pure single-speaker no-change windows.
#     """

#     rng = random.Random(seed)
#     speakers = sorted([s for s in speaker_clips if len(speaker_clips[s]) >= 1])
#     if len(speakers) < 2:
#         raise ValueError("Need at least 2 speakers with clips to build change windows")

#     max_gap_abs = min_segment_s * 0.4
#     bad_gaps = [g for g in gap_options if abs(g) > max_gap_abs]
#     if bad_gaps:
#         raise ValueError(
#             f"gap_options {bad_gaps} exceed +-{max_gap_abs:.2f}s (0.4 * min_segment_s) -- "
#             "a gap/overlap this large relative to min_segment_s risks eating all of the "
#             "guaranteed clean audio on one side of the change. Increase min_segment_s or "
#             "shrink these gap values."
#         )

#     positions = np.arange(
#         min_segment_s, window_duration_s - min_segment_s + 1e-9, change_position_step_s
#     )
#     if len(positions) == 0:
#         raise ValueError(
#             "No valid change positions: window_duration_s - 2*min_segment_s must be >= 0"
#         )

#     os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

#     manifest_rows: List[GroundTruthTurn] = []
#     clip_cursor: Dict[str, int] = {}
#     window_id = 0

#     # 1. Generate all unique speaker combinations (unordered pairs)
#     unique_speaker_pairs = list(itertools.combinations(speakers, 2))
#     rng.shuffle(unique_speaker_pairs)

#     # 2. Cycle through unique speaker pairs until num_pairs count is satisfied
#     pair_pool = itertools.cycle(unique_speaker_pairs)

#     # tqdm
#     pairs_tqdm = tqdm(range(num_pairs), desc="Generating change windows")
#     for _ in pairs_tqdm:
#         speaker_a, speaker_b = next(pair_pool)
        
#         # Randomize direction (A->B or B->A)
#         if rng.random() > 0.5:
#             speaker_a, speaker_b = speaker_b, speaker_a

#         for base_pos in positions:
#             jitter = rng.uniform(-change_position_jitter_s, change_position_jitter_s)
#             change_time_s = float(
#                 np.clip(base_pos + jitter, min_segment_s, window_duration_s - min_segment_s)
#             )
#             gap_seconds = rng.choice(gap_options)
#             noise_level = rng.choices(noise_options, weights=noise_probs, k=1)[0]

#             name = f"win_change_{window_id:06d}"
#             audio, turns = build_change_window(
#                 speaker_clips, speaker_a, speaker_b, window_duration_s,
#                 change_time_s, gap_seconds, noise_level, clip_cursor,
#             )
#             for t in turns:
#                 t.file_id = name
#             _save_window(output_dir, name, audio, turns, window_duration_s, frame_hop_ms, label_tolerance_ms)
#             manifest_rows.extend(turns)
#             window_id += 1

#     num_change = window_id
#     num_no_change = int(round(num_change * (1 - change_fraction) / max(change_fraction, 1e-9)))
    
#     # Cycle through single speakers evenly for no-change windows
#     speaker_pool = itertools.cycle(speakers)
#     # tqdm
#     no_change_tqdm = tqdm(range(num_no_change), desc="Generating no-change windows")
#     for _ in no_change_tqdm:
#         speaker = next(speaker_pool)
#         noise_level = rng.choices(noise_options, weights=noise_probs, k=1)[0]
#         name = f"win_nochange_{window_id:06d}"
#         audio, turns = build_no_change_window(
#             speaker_clips, speaker, window_duration_s, noise_level, clip_cursor
#         )
#         for t in turns:
#             t.file_id = name
#         _save_window(output_dir, name, audio, turns, window_duration_s, frame_hop_ms, label_tolerance_ms)
#         manifest_rows.extend(turns)
#         window_id += 1

#     df = pd.DataFrame([t.__dict__ for t in manifest_rows])
#     df.to_csv(os.path.join(output_dir, "annotations", "manifest.csv"), index=False)
#     print(
#         f"Generated {num_change} change windows + {num_no_change} no-change windows "
#         f"({window_id} total) in '{output_dir}'"
#     )


def load_speaker_clips(data_dir: str) -> Dict[str, List[str]]:
    data_path = Path(data_dir)
    clips_db: Dict[str, List[str]] = {}
    for speaker_dir in sorted(data_path.iterdir()):
        if speaker_dir.is_dir():
            wav_files = sorted(str(p) for p in speaker_dir.glob("*.wav"))
            if wav_files:
                clips_db[speaker_dir.name] = wav_files
    return clips_db 

def generate_and_save_suite(
        data_dir: str,
        output_dir: str,
        window_duration_s: float = 3.0,
        min_segment_s: float = 1.0,
        change_position_step_s: float = 0.5,
        change_position_jitter_s: float = 0.15,
        change_fraction: float = 0.5,
        num_pairs: int = 300,
        frame_hop_ms: float = 10.0,
        label_tolerance_ms: float = 150.0,
        seed: int = 42,
):
    clips_db = load_speaker_clips(data_dir)
    generate_window_dataset(
        clips_db,
        output_dir=output_dir,
        window_duration_s=window_duration_s,
        min_segment_s=min_segment_s,
        change_position_step_s=change_position_step_s,
        change_position_jitter_s=change_position_jitter_s,
        change_fraction=change_fraction,
        num_pairs=num_pairs,
        frame_hop_ms=frame_hop_ms,
        label_tolerance_ms=label_tolerance_ms,
        seed=seed,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/speaker_tuning/speakers")
    parser.add_argument("--output-dir", type=str, default="data/speaker_tuning/synthetic")
    parser.add_argument("--frame-hop-ms", type=float, default=10.0)
    parser.add_argument("--label-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--window-duration-s", type=float, default=3.0)
    parser.add_argument("--min-segment-s", type=float, default=1.0)
    parser.add_argument("--change-position-step-s", type=float, default=0.5)
    parser.add_argument("--change-position-jitter-s", type=float, default=0.15)
    parser.add_argument("--change-fraction", type=float, default=0.5)
    parser.add_argument("--num-pairs", type=int, default=300)

    parser.add_argument(
    "--num-workers", 
    type=int, 
    default=None, 
    help="Number of parallel processes to use. Defaults to CPU core count."
    )

    args = parser.parse_args()
    clips_db = load_speaker_clips(args.data_dir)
    generate_window_dataset(
        clips_db,
        output_dir=args.output_dir,
        window_duration_s=args.window_duration_s,
        min_segment_s=args.min_segment_s,
        change_position_step_s=args.change_position_step_s,
        change_position_jitter_s=args.change_position_jitter_s,
        change_fraction=args.change_fraction,
        num_pairs=args.num_pairs,
        frame_hop_ms=args.frame_hop_ms,
        label_tolerance_ms=args.label_tolerance_ms,
        seed=args.seed,
    )