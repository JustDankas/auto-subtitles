import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import SCDDataset
from model import SCDModel


def plot_val_session_predictions(
    work_dir: str,
    backbone: str = "gru",
    num_layers: int = 2,
    num_sessions: int = 4,
    frame_hop_ms: float = 10.0,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Pick validation sessions, run inference, and plot predicted probabilities

    over time along with true speaker boundary timestamps.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load validation dataset and model
    val_ds = SCDDataset(f"{work_dir}/data/synth_val")
    model = SCDModel(input_dim=64, hidden_dim=64, num_layers=num_layers, backbone=backbone).to(device)

    checkpoint_path = f"{work_dir}/checkpoints/best_{backbone}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Select 3-4 sessions
    num_samples = min(num_sessions, len(val_ds))
    indices = list(range(num_samples))

    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3 * num_samples), sharex=False)
    if num_samples == 1:
        axes = [axes]

    # 3. Infer and plot each session
    with torch.no_grad():
        for i, idx in enumerate(indices):
            session_id = val_ds.session_ids[idx]
            feats, labels = val_ds[idx]

            # Forward pass to get sigmoid probabilities
            logits = model(feats.unsqueeze(0).to(device)).squeeze(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.numpy()

            # Time axis in seconds
            num_frames = len(probs)
            time_axis = np.arange(num_frames) * (frame_hop_ms / 1000.0)

            # Find true boundary times from labels (1-to-0 or 0-to-1 transitions in target binary mask/label)
            # Or positive label locations where positive label indicates boundary region
            true_boundary_indices = np.where(labels_np > 0.5)[0]

            ax = axes[i]
            # Plot continuous predicted probability curve
            ax.plot(time_axis, probs, label="Predicted P(change)", color="tab:blue", linewidth=1.5)

            # Mark decision threshold line (0.5)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Threshold 0.5")

            # Draw vertical lines for true boundary regions
            if len(true_boundary_indices) > 0:
                # Group contiguous boundary indices to plot key boundary regions cleanly
                boundary_times = time_axis[true_boundary_indices]
                for b_idx, b_time in enumerate(boundary_times):
                    ax.axvline(
                        x=b_time,
                        color="tab:red",
                        alpha=0.35,
                        linestyle="-",
                        label="True Boundary" if (b_idx == 0 and i == 0) else "",
                    )

            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Probability")
            ax.set_title(f"Val Session {idx + 1}: {session_id}")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved prediction plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot model prediction probability curve over time.")
    parser.add_argument("--work_dir", type=str, required=True, help="Path to working directory")
    parser.add_argument("--backbone", type=str, default="gru", choices=["gru", "lstm", "cnn"])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_sessions", type=int, default=4)
    parser.add_argument("--save_path", type=str, default=None, help="Optional output image path")

    args = parser.parse_args()
    plot_val_session_predictions(
        work_dir=args.work_dir,
        backbone=args.backbone,
        num_layers=args.num_layers,
        num_sessions=args.num_sessions,
        save_path=args.save_path,
    )