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


def load_wav_mono16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz in {path}.")
    return audio

def derive_change_points(turns: List[GroundTruthTurn]) -> List[float]:
    """A change point is any turn whose speaker differs from the previous
    turn, sorted by start_s — this already covers overlap onsets, since an
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
    Generate per-frame label array at `frame_hop_ms` resolution.
    
    Parameters
    ----------
    turns : List[GroundTruthTurn]
        List of turn objects to derive change points from.
    duration_s : float
        Total duration of the audio segment in seconds.
    frame_hop_ms : float, optional
        Frame hop time in milliseconds. Default is 10.0 ms.
    tolerance_ms : float, optional
        Tolerance window (collar) around change points in milliseconds.
    use_soft_labels : bool, optional
        If True, applies Gaussian decay near boundaries. If False, uses hard 0/1 window.
    std_scale : float, optional
        Controls Gaussian width relative to tolerance_ms (e.g., 0.5 sets σ = tolerance_ms / 2).    """
    n_frames = int(np.ceil(duration_s * 1000.0 / frame_hop_ms))
    frame_times_s = np.arange(n_frames) * (frame_hop_ms / 1000.0)
    labels = np.zeros(n_frames, dtype=np.float32)

    change_points = derive_change_points(turns)

    tol_s = tolerance_ms / 1000.0

    if use_soft_labels:
        # Standard deviation sigma in seconds
        sigma = tol_s * std_scale
        
        for cp in change_points:
            # Mask to limit calculation within effective window (±3 sigmas or tolerance)
            dist = np.abs(frame_times_s - cp)
            mask = dist <= tol_s
            
            # Compute Gaussian weights for frames within the collar
            gaussian_vals = np.exp(-0.5 * (dist[mask] / sigma) ** 2)
            
            # Take element-wise max to handle overlapping change-point regions
            labels[mask] = np.maximum(labels[mask], gaussian_vals)
    else:
        for cp in change_points:
            labels[np.abs(frame_times_s - cp) <= tol_s] = 1.0

    return labels



def build_continuous_synthetic_session(
    speaker_clips: Dict[str, List[str]],
    name: str,
    num_speakers: int,
    gap_seconds: float,  # Can be positive (silence), 0.0, or negative (overlap e.g., -0.25, -0.5)
    turns_per_speaker: int = 4,
    run_length_range: Tuple[int, int] = (1, 3),
    seed: int = 0,
    noise_level: str = "zero",  # Options: "zero", "mild", "moderate"
) -> SyntheticSessionResult:
    """
    Constructs a single continuously-mixed WAV track and corresponding ground truth turns.
    Handles gaps, exact boundaries (0s), and negative overlaps by adding floating-point 
    waveforms directly together.
    Applies noise depending on noise_level:
      - "zero": No added background noise
      - "mild": -32 dB RMS ambient room noise
      - "moderate": -10 dB RMS background noise
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
    
    # We will accumulate mixed audio into a list of (start_sample, end_sample, audio_chunk)
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

            # Load and normalize clip volume
            raw_audio = load_wav_mono16k(path)
            norm_audio = normalize_lufs(raw_audio, target_db=-23.0)
            clip_samples = len(norm_audio)

            # Apply overlap or silence offset if not the very first turn
            if total_turns > 0:
                offset_samples = int(gap_seconds * SAMPLE_RATE)
                current_sample_cursor += offset_samples
                # Ensure cursor doesn't drift negative
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
                    notes=f"gap={gap_seconds}s"
                )
            )

            audio_placements.append((start_sample, end_sample, norm_audio))
            
            # Update cursor for the next iteration
            current_sample_cursor = end_sample
            prev_speaker = spk
            total_turns += 1

    # Calculate total duration needed for the final mixed canvas
    max_sample_len = max(end for _, end, _ in audio_placements)
    canvas = np.zeros(max_sample_len, dtype=np.float32)

    # Mix down speech audio
    for start, end, chunk in audio_placements:
        canvas[start:end] += chunk

    # Define noise levels
    noise_db_map = {
        "zero": None,
        "mild": -60.0,      # Mild room tone
        "moderate": -45.0   # Noticeable background noise
    }

    target_db = noise_db_map.get(noise_level)
    if target_db is not None:
        noise = generate_ambient_noise(len(canvas), level_db=target_db)
        canvas += noise

    # Prevent potential clipping
    max_val = np.max(np.abs(canvas))
    if max_val > 0.99:
        canvas = canvas / max_val * 0.99

    return SyntheticSessionResult(
        session_name=name,
        audio_data=canvas,
        sample_rate=SAMPLE_RATE,
        noise_level=noise_level,
        turns=turns
    )


def generate_and_save_suite(
    speaker_clips: Dict[str, List[str]],
    output_dir: str,
    seed: int = 42,
    count: int = 4
):
    """
    Executes a multi-scenario generation run covering gaps, exact boundaries, 
    and negative overlap scenarios (-0.5s, -0.25s, 0.0s, +0.25s, +0.5s).
    """
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    configs = [
        # (num_speakers, turns_per_spk, gap_seconds, count, run_range)
        (2, 6, -1.00, count, (1, 3)),   # Heavy overlap (Scenario 1)
        (2, 6, -0.50, count, (1, 3)),   # Medium overlap (Scenario 1)
        (2, 6, -0.25, count, (1, 3)),   # Slight overlap (Scenario 1)
        (2, 6,  0.00, count, (1, 3)),   # Exact boundary (Scenario 2)
        (2, 6,  0.25, count, (1, 3)),   # Brief silence (Scenario 3)
        (2, 6,  0.50, count, (1, 3)),   # Standard gap (Scenario 3)
        (4, 5, -1.00, count, (1, 3)),   # Multi-speaker overlap
        (4, 5, -0.50, count, (1, 3)),   # Multi-speaker overlap
        (4, 5,  0.25, count, (1, 3)),   # Multi-speaker gap
        (4, 5,  0.00, count, (1, 3)),   # Multi-speaker gap
    ]

    # Set noise categories and probabilities (40% zero, 30% mild, 30% moderate)
    noise_options = ["zero", "mild", "moderate"]
    noise_probs = [0.50, 0.25, 0.25]

    all_turns: List[GroundTruthTurn] = []
    session_id = 0
    rng = random.Random(seed)

    for num_speakers, turns, gap, n, run_range in configs:
        for i in range(n):
            # Sample noise level per session according to 40/30/30 distribution
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

            # Save WAV Audio
            wav_path = os.path.join(output_dir, "audio", f"{session_name}.wav")
            sf.write(wav_path, result.audio_data, result.sample_rate)

            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            duration_s = len(result.audio_data) / result.sample_rate
            labels = export_frame_labels(result.turns, duration_s,
                                        frame_hop_ms=args.frame_hop_ms,
                                        tolerance_ms=args.label_tolerance_ms)
            np.save(os.path.join(output_dir, "labels", f"{session_name}.npy"), labels)

            # Append turns to master annotation list
            all_turns.extend(result.turns)
            session_id += 1

    # Save CSV ground truth dataset
    df = pd.DataFrame([t.__dict__ for t in all_turns])
    csv_path = os.path.join(output_dir, "annotations", "manifest.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {session_id} synthetic sessions in '{output_dir}'.")
    print(f"Master CSV written to: {csv_path}")


# Example usage block (Uncomment to execute against local clip structure)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/speaker_tuning/speakers")
    parser.add_argument("--output-dir", type=str, default="data/speaker_tuning/synthetic")
    parser.add_argument("--frame-hop-ms", type=float, default=10.0)
    parser.add_argument("--label-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--count", type=int, default=4)

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    clips_db = {}
    
    # Iterate through subdirectories in data_dir (sorted for consistent ordering)
    for index, speaker_dir in enumerate(sorted(data_path.iterdir()), start=1):
        if speaker_dir.is_dir():
            # Find all .wav files in the subfolder
            wav_files = sorted(str(p) for p in speaker_dir.glob("*.wav"))
            if wav_files:
                clips_db[speaker_dir.name] = wav_files

    generate_and_save_suite(clips_db, output_dir=args.output_dir, count=args.count)