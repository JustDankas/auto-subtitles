#!/usr/bin/env python3
"""
harness.py -- Algorithm-agnostic streaming simulation harness for speaker-
change detection (Section 0.2 of the evaluation plan).

Feeds each wav in your corpus through whichever candidate algorithm you pick
via --algorithm, one small chunk at a time (matching your Silero VAD hop),
and logs:
    (a) every predicted boundary + confidence         -> predictions_<algo>.csv
    (b) wall-clock compute time per streaming call     -> runtime_stats_<algo>.csv
It also derives ground-truth speaker-change timestamps from your manifest CSV
and writes them once to ground_truth_boundaries.csv, for later use by a
metrics script (Section 0.3).

Example usage
-------------
# 1. Validate the harness itself before any model is wired up:
python harness.py \\
    --algorithm noop \\
    --manifest manifest.csv \\
    --audio-dir ./wavs \\
    --output-dir ./results/noop

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from detectors import ALGORITHM_REGISTRY, SpeakerChangeDetector, build_detector


# ---------------------------------------------------------------------------
# CPU budget -- keep every run comparable to production (CPU-only, pinned)
# ---------------------------------------------------------------------------
def set_cpu_budget(num_threads: int) -> None:
    """
    Pin thread usage so timing numbers reflect the same CPU budget every
    algorithm will actually run under in production. This covers numpy/BLAS
    and torch (if installed); each detector is still responsible for pinning
    its own onnxruntime.SessionOptions().intra_op_num_threads, since sessions
    are created inside detectors.py, not here (see the `cpu_threads` value
    forwarded into every detector's config).
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    try:
        import torch  # noqa: F401
        torch.set_num_threads(num_threads)
    except ImportError:
        pass
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(range(num_threads)))
    except Exception as e:  # pragma: no cover - platform dependent
        print(f"[warn] could not pin CPU affinity: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Manifest / ground truth
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"file_id", "start_s", "end_s", "speaker_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if "has_overlap" in df.columns and df["has_overlap"].dtype == object:
        df["has_overlap"] = df["has_overlap"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    return df


def derive_ground_truth_boundaries(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    A speaker-change boundary is placed at the start of any turn whose
    speaker differs from the immediately preceding turn (per file_id, sorted
    by start_s). `new_start_s`/`prev_end_s` are both kept so a later metrics
    script can widen the match window for overlapping turns (has_overlap) if
    it wants a looser collar there than for clean, gapped turns.
    """
    rows = []
    for file_id, group in manifest.groupby("file_id", sort=False):
        g = group.sort_values("start_s").reset_index(drop=True)
        for i in range(1, len(g)):
            prev, cur = g.iloc[i - 1], g.iloc[i]
            if cur["speaker_id"] != prev["speaker_id"]:
                rows.append(
                    {
                        "file_id": file_id,
                        "boundary_time_s": cur["start_s"],
                        "prev_speaker": prev["speaker_id"],
                        "new_speaker": cur["speaker_id"],
                        "prev_end_s": prev["end_s"],
                        "new_start_s": cur["start_s"],
                        "has_overlap": bool(cur.get("has_overlap", False))
                        or bool(prev.get("has_overlap", False)),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "file_id",
            "boundary_time_s",
            "prev_speaker",
            "new_speaker",
            "prev_end_s",
            "new_start_s",
            "has_overlap",
        ],
    )


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def resample_audio(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def load_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    return audio.astype(np.float32, copy=False), sr


# ---------------------------------------------------------------------------
# Streaming simulation
# ---------------------------------------------------------------------------
def run_file(
    detector: SpeakerChangeDetector, audio: np.ndarray, sr: int, chunk_ms: float
) -> Tuple[List[Dict], Dict]:
    """
    Feeds `audio` to `detector` one chunk at a time, exactly as it would
    arrive in production, timing each call. Returns (predictions, stats).

    Each prediction records two timestamps:
      - boundary_time_s: the algorithm's own estimate of when the change
        happened (some algorithms look slightly backward/forward).
      - emitted_at_s: how much audio had actually been streamed in when the
        detector produced that prediction -- this is what the person watching
        captions actually experiences as "the delay", and is what later
        latency metrics should be computed from.
    """
    detector.reset()
    chunk_samples = max(1, int(round(sr * chunk_ms / 1000.0)))
    n = len(audio)

    predictions: List[Dict] = []
    compute_times_s: List[float] = []

    pos = 0
    while pos < n:
        end = min(pos + chunk_samples, n)
        chunk = audio[pos:end]
        chunk_start_s = pos / sr

        t0 = time.perf_counter()
        events = detector.process_chunk(chunk, chunk_start_s)
        t1 = time.perf_counter()
        compute_times_s.append(t1 - t0)

        emitted_at_s = end / sr
        for ev in events:
            predictions.append(
                {
                    "emitted_at_s": emitted_at_s,
                    "boundary_time_s": ev.boundary_time_s,
                    "confidence": ev.confidence,
                }
            )
        pos = end

    stream_end_s = n / sr
    t0 = time.perf_counter()
    final_events = detector.finalize(stream_end_s)
    t1 = time.perf_counter()
    compute_times_s.append(t1 - t0)
    for ev in final_events:
        predictions.append(
            {
                "emitted_at_s": stream_end_s,
                "boundary_time_s": ev.boundary_time_s,
                "confidence": ev.confidence,
            }
        )

    ct = np.array(compute_times_s, dtype=np.float64)
    stats = {
        "audio_duration_s": stream_end_s,
        "num_chunks": int(len(ct)),
        "total_compute_s": float(ct.sum()),
        "mean_chunk_ms": float(ct.mean() * 1000) if len(ct) else 0.0,
        "p95_chunk_ms": float(np.percentile(ct, 95) * 1000) if len(ct) else 0.0,
        "max_chunk_ms": float(ct.max() * 1000) if len(ct) else 0.0,
        "rtf": float(ct.sum() / stream_end_s) if stream_end_s > 0 else float("nan"),
    }
    return predictions, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Algorithm-agnostic streaming evaluation harness for speaker-change detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--algorithm", default="scd_classifier", choices=sorted(ALGORITHM_REGISTRY))
    p.add_argument("--manifest", default="../data/speaker_tuning/synthetic/annotations/manifest.csv", type=Path)
    p.add_argument("--audio-dir", default="../data/speaker_tuning/synthetic/audio", type=Path)
    p.add_argument("--output-dir", default="results", type=Path)
    p.add_argument(
        "--chunk-ms",
        type=float,
        default=32.0,
        help="Streaming chunk size in ms. Match your Silero VAD hop "
        "(512 samples @16kHz = 32ms is Silero's default). Default: 32",
    )
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="CPU threads to pin the process to, matching production (default: 1)",
    )
    p.add_argument(
        "--config",
        type=str,
        default="{}",
        help="JSON string of algorithm-specific hyperparameters, e.g. "
        '\'{"left_ms": 500, "right_ms": 500, "threshold": 0.3}\'',
    )
    p.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to a JSON file of hyperparameters (overrides --config if both given)",
    )
    p.add_argument(
        "--file-id",
        type=str,
        default=None,
        help="Restrict the run to a single file_id from the manifest, for debugging",
    )
    p.add_argument("--limit", type=int, default=None, help="Only process the first N files")
    p.add_argument("--audio-ext", type=str, default=".wav")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_cpu_budget(args.num_threads)

    if args.config_file is not None:
        config = json.loads(Path(args.config_file).read_text())
    else:
        config = json.loads(args.config)
    config.setdefault("cpu_threads", args.num_threads)

    manifest = load_manifest(args.manifest)
    ground_truth = derive_ground_truth_boundaries(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = args.output_dir / "ground_truth_boundaries.csv"
    ground_truth.to_csv(gt_path, index=False)
    print(f"Derived {len(ground_truth)} ground-truth boundaries -> {gt_path}")

    file_ids = list(dict.fromkeys(manifest["file_id"]))  # preserve order, de-dupe
    if args.file_id:
        file_ids = [f for f in file_ids if f == args.file_id]
    if args.limit:
        file_ids = file_ids[: args.limit]
    if not file_ids:
        print("[warn] no files selected -- check --file-id / --limit / manifest contents", file=sys.stderr)

    detector = build_detector(args.algorithm, sample_rate=args.sample_rate, **config)

    all_predictions: List[Dict] = []
    all_stats: List[Dict] = []
    for file_id in file_ids:
        wav_path = args.audio_dir / f"{file_id}{args.audio_ext}"
        if not wav_path.exists():
            print(f"[warn] missing audio for {file_id}: {wav_path}", file=sys.stderr)
            continue

        audio, sr = load_audio(wav_path, args.sample_rate)
        try:
            predictions, stats = run_file(detector, audio, sr, args.chunk_ms)
        except NotImplementedError as e:
            print(f"[error] '{args.algorithm}' is not implemented yet: {e}", file=sys.stderr)
            sys.exit(1)

        for pred in predictions:
            pred["file_id"] = file_id
            pred["algorithm"] = args.algorithm
        stats["file_id"] = file_id
        stats["algorithm"] = args.algorithm

        all_predictions.extend(predictions)
        all_stats.append(stats)
        print(
            f"[{file_id}] {len(predictions)} predicted boundaries, "
            f"RTF={stats['rtf']:.3f}, mean_chunk_ms={stats['mean_chunk_ms']:.2f}, "
            f"p95_chunk_ms={stats['p95_chunk_ms']:.2f}"
        )

    pred_df = pd.DataFrame(
        all_predictions,
        columns=["algorithm", "file_id", "emitted_at_s", "boundary_time_s", "confidence"],
    )
    stats_df = pd.DataFrame(all_stats)

    pred_path = args.output_dir / f"predictions_{args.algorithm}.csv"
    stats_path = args.output_dir / f"runtime_stats_{args.algorithm}.csv"
    pred_df.to_csv(pred_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"\nWrote {len(pred_df)} predictions      -> {pred_path}")
    print(f"Wrote runtime stats for {len(stats_df)} files -> {stats_path}")
    if len(stats_df):
        print(
            f"Aggregate: mean RTF={stats_df['rtf'].mean():.3f}  "
            f"mean p95_chunk_ms={stats_df['p95_chunk_ms'].mean():.2f}"
        )


if __name__ == "__main__":
    main()