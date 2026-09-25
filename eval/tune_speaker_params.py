#!/usr/bin/env python3
"""
tune_speaker_params.py

Grid-searches InstantChangeDetector parameters (speaker-k, std-floor,
min-window, max-window) against synthetic multi-speaker sessions built
from a folder of single-speaker clips.

See MANUAL.md for the dataset layout this expects and for how to
interpret the results.

Usage:
    python tune_speaker_params.py \
        --dataset-dir data/speaker_tuning/speakers \
        --embedding-model models/wespeaker_en_voxceleb_CAM++.onnx \
        --out-dir results/

    # Re-use cached embeddings on a later run (skip re-running the model):
    python tune_speaker_params.py \
        --dataset-dir data/speaker_tuning/speakers \
        --embedding-cache results/embeddings_cache.npz \
        --out-dir results/
"""

import argparse
import json
import os
import sys

from speaker_tuning_lib import (
    EmbeddingExtractor,
    build_session_suite,
    discover_speaker_clips,
    embed_and_cache_suite,
    grid_search,
    load_cached_embeddings,
    results_table,
    save_results_json,
    train_val_split,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=str, default=None,
                    help="Folder of <speaker_id>/*.wav clips. Required unless --embedding-cache is given.")
    p.add_argument("--embedding-model", type=str, default=None,
                    help="Path to sherpa-onnx speaker embedding .onnx model. "
                         "Required unless --embedding-cache is given.")
    p.add_argument("--embedding-cache", type=str, default=None,
                    help="Path to a .npz cache written by a previous run. If given and it "
                         "exists, embedding extraction is skipped entirely.")
    p.add_argument("--out-dir", type=str, default="results",
                    help="Where to write embeddings_cache.npz, grid_results.json, and best_params.json")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.3,
                    help="Fraction of sessions held out for validation (not used in grid search).")
    p.add_argument("--beta", type=float, default=1.5,
                    help="F-beta weight. >1 favors recall (fewer missed speaker changes) "
                         "over precision (fewer spurious line splits).")

    p.add_argument("--k-grid", type=float, nargs="+",
                    default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                    help="Candidate values for --speaker-k")
    p.add_argument("--std-floor-grid", type=float, nargs="+",
                    default=[0.02, 0.05, 0.08, 0.12],
                    help="Candidate values for --std-floor")
    p.add_argument("--window-grid", type=str, nargs="+",
                    default=["2:15", "3:20", "4:25"],
                    help="Candidate (min_window:max_window) pairs, e.g. 3:20")
    p.add_argument("--min-active-seconds-grid", type=float, nargs="+",
                    default=[0.0],
                    help="Candidate values for a duration-based gate: require at least "
                         "this many seconds of audio in the window (on top of min_window) "
                         "before a change can be detected. 0.0 disables it. Try e.g. "
                         "1.0 1.5 2.0 to match 'only decide after the preceding speaker "
                         "has talked for 1-2 seconds'.")

    p.add_argument("--include-gap-zero", action="store_true",
                    help="Include zero-gap (no-pause) sessions in the score used to pick "
                         "the winner. Off by default -- see MANUAL.md 'Known limitation'.")
    p.add_argument("--top-n", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    window_grid = []
    for pair in args.window_grid:
        min_w, max_w = pair.split(":")
        window_grid.append((int(min_w), int(max_w)))

    # 1. Get embeddings, either from cache or by building sessions + running the extractor.
    cache_path = args.embedding_cache or os.path.join(args.out_dir, "embeddings_cache.npz")

    if args.embedding_cache and os.path.exists(args.embedding_cache):
        print(f"Loading cached embeddings from {args.embedding_cache}")
        embeddings_by_session = load_cached_embeddings(args.embedding_cache)
        if not args.dataset_dir:
            print("ERROR: --dataset-dir is still required to rebuild session metadata "
                  "(true speaker-change labels) even when using a cache.", file=sys.stderr)
            sys.exit(1)
        speaker_clips = discover_speaker_clips(args.dataset_dir)
        sessions = build_session_suite(speaker_clips, seed=args.seed)
        # sanity check: cache must contain every session we just built
        missing = [s.name for s in sessions if s.name not in embeddings_by_session]
        if missing:
            print(f"ERROR: cache is missing sessions {missing}. Delete the cache and rerun "
                  "without --embedding-cache, or check --seed matches the run that built it.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        if not args.dataset_dir or not args.embedding_model:
            print("ERROR: --dataset-dir and --embedding-model are required "
                  "when not using an existing --embedding-cache.", file=sys.stderr)
            sys.exit(1)
        print(f"Scanning dataset: {args.dataset_dir}")
        speaker_clips = discover_speaker_clips(args.dataset_dir)
        n_speakers = len(speaker_clips)
        n_clips = sum(len(v) for v in speaker_clips.values())
        print(f"Found {n_speakers} speakers, {n_clips} clips total.")
        if n_speakers < 6:
            print("WARNING: fewer than 6 speakers found. MANUAL.md recommends >= 8 for a "
                  "usable train/val split. Results below may be unstable.", file=sys.stderr)

        print("Building synthetic sessions...")
        sessions = build_session_suite(speaker_clips, seed=args.seed)
        print(f"Built {len(sessions)} sessions.")

        print(f"Loading embedding model: {args.embedding_model}")
        extractor = EmbeddingExtractor(args.embedding_model, num_threads=args.num_threads,
                                        provider=args.provider)
        print("Computing embeddings for every session (this is the only slow step)...")
        embeddings_by_session = embed_and_cache_suite(sessions, extractor, cache_path=cache_path)
        print(f"Cached embeddings to {cache_path}")

    # 2. Train/val split.
    train_sessions, val_sessions = train_val_split(sessions, val_fraction=args.val_fraction,
                                                     seed=args.seed)
    print(f"\nTrain sessions: {[s.name for s in train_sessions]}")
    print(f"Val sessions:   {[s.name for s in val_sessions]}")

    # 3. Grid search on train sessions only.
    print(f"\nGrid: {len(args.k_grid)} k values x {len(args.std_floor_grid)} std_floor values "
          f"x {len(window_grid)} window pairs = "
          f"{len(args.k_grid) * len(args.std_floor_grid) * len(window_grid)} points")
    results = grid_search(train_sessions, embeddings_by_session,
                           args.k_grid, args.std_floor_grid, window_grid,
                           min_active_seconds_grid=args.min_active_seconds_grid, beta=args.beta)

    print(f"\n=== Top {args.top_n} on TRAIN sessions (F{args.beta}) ===")
    print(results_table(results, top_n=args.top_n))

    save_results_json(results, os.path.join(args.out_dir, "grid_results_train.json"))

    # 4. Evaluate the winner (and runner-ups) on held-out validation sessions.
    from speaker_tuning_lib import evaluate_params
    print(f"\n=== Same top {min(5, args.top_n)} points, scored on VAL sessions ===")
    val_scored = []
    for gp in results[:5]:
        val_gp = evaluate_params(val_sessions, embeddings_by_session,
                                  gp.k, gp.std_floor, gp.min_window, gp.max_window,
                                  min_active_seconds=gp.min_active_seconds,
                                  beta=args.beta, include_gap_zero=args.include_gap_zero)
        val_scored.append(val_gp)
    print(results_table(val_scored, top_n=5))

    best_train = results[0]
    best_val = max(val_scored, key=lambda gp: gp.mean_fbeta)

    print("\n=== Recommendation ===")
    print(f"Best on train:      k={best_train.k}, std_floor={best_train.std_floor}, "
          f"min_window={best_train.min_window}, max_window={best_train.max_window} "
          f"(train F={best_train.mean_fbeta:.3f})")
    print(f"Best of those on val: k={best_val.k}, std_floor={best_val.std_floor}, "
          f"min_window={best_val.min_window}, max_window={best_val.max_window} "
          f"(val F={best_val.mean_fbeta:.3f})")
    if best_train.k != best_val.k or best_train.std_floor != best_val.std_floor:
        print("NOTE: train and val disagree on the best point. Prefer the val-best "
              "params below, and consider adding more speakers/sessions -- disagreement "
              "usually means the grid is overfitting to train-session quirks.")

    with open(os.path.join(args.out_dir, "best_params.json"), "w") as f:
        json.dump({
            "speaker_k": best_val.k,
            "std_floor": best_val.std_floor,
            "speaker_min_window": best_val.min_window,
            "speaker_max_window": best_val.max_window,
            "min_active_seconds": best_val.min_active_seconds,
            "val_mean_fbeta": best_val.mean_fbeta,
            "val_mean_precision": best_val.mean_precision,
            "val_mean_recall": best_val.mean_recall,
            "train_mean_fbeta": best_train.mean_fbeta,
        }, f, indent=2)
    print(f"\nWrote best_params.json to {args.out_dir}/")

    # 5. Zero-gap stress test, reported separately, never used to pick params.
    gap_zero_sessions = [s for s in sessions if s.gap_seconds == 0.0]
    if gap_zero_sessions and not args.include_gap_zero:
        stress_gp = evaluate_params(gap_zero_sessions, embeddings_by_session,
                                     best_val.k, best_val.std_floor,
                                     best_val.min_window, best_val.max_window,
                                     min_active_seconds=best_val.min_active_seconds,
                                     beta=args.beta, include_gap_zero=True)
        print(f"\n=== Zero-gap stress test (winner params, NOT used for selection) ===")
        print(f"mean F{args.beta}={stress_gp.mean_fbeta:.3f}  "
              f"precision={stress_gp.mean_precision:.3f}  recall={stress_gp.mean_recall:.3f}")
        print("Low recall here is expected -- see MANUAL.md 'Known limitation'. "
              "This is what Phase 9 (sliding-window mid-segment detection) is for.")


if __name__ == "__main__":
    main()
