#!/usr/bin/env python3
"""
eval_metrics.py -- Collar-based evaluation and threshold sweeping script
for Speaker Change Detection.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_predictions(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    threshold: float,
    collar_s: float = 0.25,
) -> dict:
    """
    Matches predictions with ground truth boundaries within a tolerance collar.
    """
    filtered_preds = pred_df[pred_df["confidence"] >= threshold]

    tp, fp, fn = 0, 0, 0
    total_frames_evaluated = 0

    file_ids = set(gt_df["file_id"]).union(set(pred_df["file_id"]))

    for fid in file_ids:
        gt_times = gt_df[gt_df["file_id"] == fid]["boundary_time_s"].values
        p_times = filtered_preds[filtered_preds["file_id"] == fid]["boundary_time_s"].values

        matched_gt = set()
        for p in p_times:
            # Find matching ground truth within collar
            matches = [
                idx for idx, g in enumerate(gt_times)
                if abs(p - g) <= collar_s and idx not in matched_gt
            ]
            if matches:
                matched_gt.add(matches[0])
                tp += 1
            else:
                fp += 1

        fn += len(gt_times) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SCD model predictions.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing CSV results from harness")
    parser.add_argument("--algorithm", type=str, default="scd_classifier", help="Algorithm name prefix")
    parser.add_argument("--collar-s", type=float, default=0.25, help="Tolerance collar in seconds (default: 0.25s)")
    args = parser.parse_args()

    gt_path = args.results_dir / "ground_truth_boundaries.csv"
    pred_path = args.results_dir / f"predictions_{args.algorithm}.csv"

    if not gt_path.exists() or not pred_path.exists():
        raise FileNotFoundError("Results files missing. Run harness.py first.")

    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    thresholds = np.linspace(0.1, 0.9, 17)
    results = []

    for t in thresholds:
        res = evaluate_predictions(gt_df, pred_df, threshold=t, collar_s=args.collar_s)
        results.append(res)

    res_df = pd.DataFrame(results)

    print("\n--- Threshold Sweep Metrics ---")
    print(res_df[["threshold", "precision", "recall", "f1", "accuracy", "TP", "FP", "FN"]].to_string(index=False))

    best_idx = res_df["f1"].idxmax()
    best_res = res_df.iloc[best_idx]

    print("\n--- Optimal Threshold Strategy ---")
    print(f"Optimal Threshold (Max F1): {best_res['threshold']:.2f}")
    print(f"Precision : {best_res['precision']:.4f}")
    print(f"Recall    : {best_res['recall']:.4f}")
    print(f"F1-Score  : {best_res['f1']:.4f}")
    print(f"Accuracy  : {best_res['accuracy']:.4f}")


if __name__ == "__main__":
    main()