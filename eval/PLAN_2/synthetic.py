import argparse
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

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
# NEW: fixed-duration, single-change (or no-change) window generation
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


def generate_window_dataset(
    speaker_clips: Dict[str, List[str]],
    output_dir: str,
    window_duration_s: float = 3.0,
    min_segment_s: float = 0.5,
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
) -> None:
    """
    Replaces the multi-turn session generator for classifier training: every
    output file is a standalone `window_duration_s`-long clip containing
    *at most one* speaker change, so there is never a risk of two changes
    confusing the model within a single training example.

    For each of `num_pairs` speaker pairs, generates one window per position
    in `change_position_step_s`-spaced steps across the window (with small
    random jitter) -- e.g. for a 3.0s window, 0.5s min_segment and 0.5s step:
    change points at roughly 0.5, 1.0, 1.5, 2.0, 2.5s. This deliberately
    covers the realistic range of "how long ago did the change happen"
    a trailing window sees in production, instead of only ever training on
    a change sitting at the window's midpoint.

    `change_fraction` controls what fraction of the final dataset has a
    change at all; the remainder are pure single-speaker no-change windows.
    """
    rng = random.Random(seed)
    speakers = [s for s in speaker_clips if len(speaker_clips[s]) >= 1]
    if len(speakers) < 2:
        raise ValueError("Need at least 2 speakers with clips to build change windows")

    max_gap_abs = min_segment_s * 0.4
    bad_gaps = [g for g in gap_options if abs(g) > max_gap_abs]
    if bad_gaps:
        raise ValueError(
            f"gap_options {bad_gaps} exceed +-{max_gap_abs:.2f}s (0.4 * min_segment_s) -- "
            "a gap/overlap this large relative to min_segment_s risks eating all of the "
            "guaranteed clean audio on one side of the change. Increase min_segment_s or "
            "shrink these gap values."
        )

    positions = np.arange(
        min_segment_s, window_duration_s - min_segment_s + 1e-9, change_position_step_s
    )
    if len(positions) == 0:
        raise ValueError(
            "No valid change positions: window_duration_s - 2*min_segment_s must be >= 0"
        )

    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    manifest_rows: List[GroundTruthTurn] = []
    clip_cursor: Dict[str, int] = {}
    window_id = 0

    for _ in range(num_pairs):
        speaker_a, speaker_b = rng.sample(speakers, 2)
        for base_pos in positions:
            jitter = rng.uniform(-change_position_jitter_s, change_position_jitter_s)
            change_time_s = float(
                np.clip(base_pos + jitter, min_segment_s, window_duration_s - min_segment_s)
            )
            gap_seconds = rng.choice(gap_options)
            noise_level = rng.choices(noise_options, weights=noise_probs, k=1)[0]

            name = f"win_change_{window_id:06d}"
            audio, turns = build_change_window(
                speaker_clips, speaker_a, speaker_b, window_duration_s,
                change_time_s, gap_seconds, noise_level, clip_cursor,
            )
            for t in turns:
                t.file_id = name
            _save_window(output_dir, name, audio, turns, window_duration_s, frame_hop_ms, label_tolerance_ms)
            manifest_rows.extend(turns)
            window_id += 1

    num_change = window_id
    num_no_change = int(round(num_change * (1 - change_fraction) / max(change_fraction, 1e-9)))
    for _ in range(num_no_change):
        speaker = rng.choice(speakers)
        noise_level = rng.choices(noise_options, weights=noise_probs, k=1)[0]
        name = f"win_nochange_{window_id:06d}"
        audio, turns = build_no_change_window(
            speaker_clips, speaker, window_duration_s, noise_level, clip_cursor
        )
        for t in turns:
            t.file_id = name
        _save_window(output_dir, name, audio, turns, window_duration_s, frame_hop_ms, label_tolerance_ms)
        manifest_rows.extend(turns)
        window_id += 1

    df = pd.DataFrame([t.__dict__ for t in manifest_rows])
    df.to_csv(os.path.join(output_dir, "annotations", "manifest.csv"), index=False)
    print(
        f"Generated {num_change} change windows + {num_no_change} no-change windows "
        f"({window_id} total) in '{output_dir}'"
    )


# ---------------------------------------------------------------------------
# EXISTING: multi-turn continuous session generator (kept for other uses --
# e.g. Plan 1/Plan 3 full-session evaluation still wants realistic long
# multi-speaker recordings; only the *classifier* training data moved to
# fixed windows above).
# ---------------------------------------------------------------------------
def build_continuous_synthetic_session(
    speaker_clips: Dict[str, List[str]],
    name: str,
    num_speakers: int,
    gap_seconds: float,
    turns_per_speaker: int = 4,
    run_length_range: Tuple[int, int] = (1, 3),
    seed: int = 0,
    noise_level: str = "zero",
) -> SyntheticSessionResult:
    """
    Constructs a single continuously-mixed WAV track and corresponding ground truth turns.
    Handles gaps, exact boundaries (0s), and negative overlaps by adding floating-point
    waveforms directly together.
    """
    rng = random.Random(seed)
    available = [s for s in speaker_clips if len(speaker_clips[s]) >= 2]
    if len(available) < num_speakers:
        raise ValueError(
            f"Need {num_speakers} speakers with >= 2 clips, only found {len(available)}"
        )

    chosen = rng.sample(available, k=num_speakers)
    shuffled_clips = {s: rng.sample(speaker_clips[s], k=len(speaker_clips[s])) for s in chosen}
    clip_idx = {s: 0 for s in chosen}

    total_budget = turns_per_speaker * num_speakers
    turns: List[GroundTruthTurn] = []
    audio_placements: List[Tuple[int, int, np.ndarray]] = []

    current_sample_cursor = 0
    prev_speaker = None
    total_turns = 0

    while total_turns < total_budget:
        candidates = [s for s in chosen if s != prev_speaker] if prev_speaker else chosen
        spk = rng.choice(candidates)
        run_len = rng.randint(*run_length_range)

        for _ in range(run_len):
            if total_turns >= total_budget:
                break

            clips = shuffled_clips[spk]
            path = clips[clip_idx[spk] % len(clips)]
            clip_idx[spk] += 1

            raw_audio = load_wav_mono16k(path)
            norm_audio = normalize_lufs(raw_audio, target_db=-23.0)
            clip_samples = len(norm_audio)

            if total_turns > 0:
                offset_samples = int(gap_seconds * SAMPLE_RATE)
                current_sample_cursor += offset_samples
                current_sample_cursor = max(0, current_sample_cursor)

            start_sample = current_sample_cursor
            end_sample = start_sample + clip_samples

            start_s = round(start_sample / SAMPLE_RATE, 3)
            end_s = round(end_sample / SAMPLE_RATE, 3)

            is_overlap = gap_seconds < 0.0 and prev_speaker is not None and (prev_speaker != spk)

            turns.append(
                GroundTruthTurn(
                    file_id=name,
                    start_s=start_s,
                    end_s=end_s,
                    speaker_id=spk,
                    has_overlap=is_overlap,
                    noise_level=noise_level,
                    notes=f"gap={gap_seconds}s",
                )
            )

            audio_placements.append((start_sample, end_sample, norm_audio))

            current_sample_cursor = end_sample
            prev_speaker = spk
            total_turns += 1

    max_sample_len = max(end for _, end, _ in audio_placements)
    canvas = np.zeros(max_sample_len, dtype=np.float32)

    for start, end, chunk in audio_placements:
        canvas[start:end] += chunk

    canvas = apply_background_noise(canvas, noise_level)
    canvas = prevent_clipping(canvas)

    return SyntheticSessionResult(
        session_name=name,
        audio_data=canvas,
        sample_rate=SAMPLE_RATE,
        noise_level=noise_level,
        turns=turns,
    )


def generate_and_save_suite(
    speaker_clips: Dict[str, List[str]],
    output_dir: str,
    seed: int = 42,
    count: int = 4,
    frame_hop_ms: float = 10.0,
    label_tolerance_ms: float = 150.0,
):
    """
    Multi-scenario continuous-session generation (unchanged behavior from
    before, aside from frame_hop_ms/label_tolerance_ms now being explicit
    parameters instead of reaching for the module-level `args`).
    """
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    configs = [
        # (num_speakers, turns_per_spk, gap_seconds, count, run_range)
        (4, 5, -1.00, count, (1, 3)),
        (4, 5, -0.50, count, (1, 3)),
        (4, 5, 0.25, count, (1, 3)),
        (4, 5, 0.00, count, (1, 3)),
    ]

    noise_options = ["zero", "mild", "moderate"]
    noise_probs = [0.50, 0.25, 0.25]

    all_turns: List[GroundTruthTurn] = []
    session_id = 0
    rng = random.Random(seed)

    for num_speakers, turns, gap, n, run_range in configs:
        for i in range(n):
            selected_noise = rng.choices(noise_options, weights=noise_probs, k=1)[0]
            session_name = f"synth_spk{num_speakers}_gap{gap:.2f}_noise-{selected_noise}_{i}"

            result = build_continuous_synthetic_session(
                speaker_clips=speaker_clips,
                name=session_name,
                num_speakers=num_speakers,
                gap_seconds=gap,
                turns_per_speaker=turns,
                run_length_range=run_range,
                seed=seed + session_id,
                noise_level=selected_noise,
            )

            wav_path = os.path.join(output_dir, "audio", f"{session_name}.wav")
            sf.write(wav_path, result.audio_data, result.sample_rate)

            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            duration_s = len(result.audio_data) / result.sample_rate
            labels = export_frame_labels(
                result.turns, duration_s, frame_hop_ms=frame_hop_ms, tolerance_ms=label_tolerance_ms
            )
            np.save(os.path.join(output_dir, "labels", f"{session_name}.npy"), labels)

            all_turns.extend(result.turns)
            session_id += 1

    df = pd.DataFrame([t.__dict__ for t in all_turns])
    csv_path = os.path.join(output_dir, "annotations", "manifest.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {session_id} synthetic sessions in '{output_dir}'.")
    print(f"Master CSV written to: {csv_path}")


def load_speaker_clips(data_dir: str) -> Dict[str, List[str]]:
    data_path = Path(data_dir)
    clips_db: Dict[str, List[str]] = {}
    for speaker_dir in sorted(data_path.iterdir()):
        if speaker_dir.is_dir():
            wav_files = sorted(str(p) for p in speaker_dir.glob("*.wav"))
            if wav_files:
                clips_db[speaker_dir.name] = wav_files
    return clips_db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/speaker_tuning/speakers")
    parser.add_argument("--output-dir", type=str, default="data/speaker_tuning/synthetic")
    parser.add_argument("--frame-hop-ms", type=float, default=10.0)
    parser.add_argument("--label-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--mode", choices=["windows", "sessions"], default="windows",
        help="'windows' (default): fixed-duration single-change/no-change clips for "
        "classifier training. 'sessions': legacy multi-turn continuous recordings.",
    )

    # windows mode
    parser.add_argument("--window-duration-s", type=float, default=3.0)
    parser.add_argument("--min-segment-s", type=float, default=1.0)
    parser.add_argument("--change-position-step-s", type=float, default=0.5)
    parser.add_argument("--change-position-jitter-s", type=float, default=0.15)
    parser.add_argument("--change-fraction", type=float, default=0.5)
    parser.add_argument("--num-pairs", type=int, default=300)

    # sessions mode (legacy)
    parser.add_argument("--count", type=int, default=4)

    args = parser.parse_args()
    clips_db = load_speaker_clips(args.data_dir)

    if args.mode == "windows":
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
    else:
        generate_and_save_suite(
            clips_db,
            output_dir=args.output_dir,
            count=args.count,
            seed=args.seed,
            frame_hop_ms=args.frame_hop_ms,
            label_tolerance_ms=args.label_tolerance_ms,
        )
