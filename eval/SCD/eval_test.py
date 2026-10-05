"""
Run:
python eval_test.py 
  --checkpoint checkpoints/scdnet_best.pt 
  --test_pool_dir pools/test 
  --test_fixed_path test_fixed.npz 
  --test_size 3000
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from model import SCDNet
from torch.utils.data import DataLoader
from train import FixedWindowSet, run_validation
from window_sampler import Pool, SamplerConfig, WindowSampler, make_fixed_set


def evaluate_test_set(
    checkpoint_path: str,
    test_pool_dir: str,
    test_fixed_path: str = "test_fixed.npz",
    test_size: int = 3000,
    batch_size: int = 64,
    aux_weight: float = 0.5,
    causal: bool = True,
    device: str = None,
    results_dir="results",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Create a frozen test dataset if it doesn't already exist
    if not os.path.exists(test_fixed_path):
        print(f"Generating frozen test set ({test_size} windows) -> {test_fixed_path}...")
        test_pool = Pool(f"{test_pool_dir}/pool.npy", f"{test_pool_dir}/index.npz")
        # Ensure speed perturbations are off for evaluation
        test_sampler = WindowSampler(test_pool, SamplerConfig(p_speed=0.0))
        make_fixed_set(test_sampler, n=test_size, seed=42, path=test_fixed_path)
    else:
        print(f"Loading existing frozen test set: {test_fixed_path}")

    # 2. DataLoader
    test_ds = FixedWindowSet(test_fixed_path)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Load Model Checkpoint
    model = SCDNet(
        n_mels=64, ch=64, kernel_size=5, dilations=(1, 2, 4, 8),
        dropout=0.15, causal=causal, augment=False, n_outputs=2
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # 4. Run Evaluation
    loss, report = run_validation(model, test_dl, device, aux_weight)

    # 5. Output Results
    lines = [
        "=" * 40,
        "TEST EVALUATION REPORT",
        "=" * 40,
        f"Checkpoint       : {checkpoint_path}",
        f"Test Loss        : {loss:.4f}",
        f"Test AP          : {report['AP']:.4f}",
        f"Test AUROC       : {report['AUROC']:.4f}",
        f"Event F1 @ 200ms : {report['event@200ms']['f1']:.4f}",
        f"Precision @200ms : {report['event@200ms']['precision']:.4f}",
        f"Recall @200ms    : {report['event@200ms']['recall']:.4f}",
    ]

    if "median_ms" in report.get("lag", {}):
        lines.append(f"Median Lag       : {report['lag']['median_ms']:+.1f} ms")

    lines.append("=" * 40)

    report_str = "\n".join(lines)

    # Print to console
    print("\n" + report_str)

    # Save to .txt file in results_dir named after the checkpoint .pt filename
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        checkpoint_stem = Path(checkpoint_path).stem  # Extracts filename without extension
        txt_filename = f"{checkpoint_stem}.txt"
        output_filepath = os.path.join(results_dir, txt_filename)

        with open(output_filepath, "w") as f:
            f.write(report_str + "\n")

        print(f"Report successfully saved to: {output_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SCDNet on a test pool.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--test_pool_dir", type=str, required=True, help="Directory containing test pool.npy and index.npz")
    parser.add_argument("--test_fixed_path", type=str, default="test_fixed.npz", help="Path to save/read frozen test .npz")
    parser.add_argument("--test_size", type=int, default=3000, help="Number of test windows to generate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--non-causal", action="store_true", default=False)
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save evaluation .txt reports")
    args = parser.parse_args()

    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        test_pool_dir=args.test_pool_dir,
        test_fixed_path=args.test_fixed_path,
        test_size=args.test_size,
        batch_size=args.batch_size,
        device=args.device,
        causal=not args.non_causal,
        results_dir=args.results_dir,
    )

