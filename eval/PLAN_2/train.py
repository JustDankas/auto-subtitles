import argparse
import os
from datetime import datetime
from typing import Literal, Optional

import matplotlib.pyplot as plt
import torch
from dataset import SCDDataset, collate_fn
from model import SCDModel
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    work_dir: str,
    backbone: Literal["gru", "lstm", "cnn"],
    window_duration_s: float = 3.0,
    frame_hop_ms: float = 10.0,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: Optional[int] = None,
    kernel_size: int = 7,
    patience: int = 5,
    batch_size: int = 32,
    device: Optional[str] = None,
    num_workers: int = 0
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if num_workers == 0:
        num_workers = min(4, os.cpu_count())

    train_ds = SCDDataset(f"{work_dir}/data/synth_train")
    val_ds = SCDDataset(f"{work_dir}/data/synth_val")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=num_workers,
                          pin_memory=True,  # Speeds up CPU-to-GPU data transfer
                          persistent_workers=True if num_workers > 0 else False,  # Prevents worker recreation cost every epoch
                          prefetch_factor=4 if num_workers > 0 else None,  # Prefetches 4 batches per worker
                        )
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn,
                          num_workers=num_workers,
                          pin_memory=True,  # Speeds up CPU-to-GPU data transfer
                          persistent_workers=True if num_workers > 0 else False,  # Prevents worker recreation cost every epoch
                          prefetch_factor=4 if num_workers > 0 else None,  # Prefetches 4 batches per worker
)

    # Class imbalance weight, computed on the *actual* training distribution
    # (roughly change_fraction of windows have a soft-labeled bump; the rest
    # are all-zero) -- recompute this any time change_fraction/tolerance_ms
    # changes, don't reuse a stale value from a previous dataset version.
    # pos = sum(l.sum().item() for _, l in train_ds)
    # neg = sum(l.numel() - l.sum().item() for _, l in train_ds)
    # pos_weight = torch.tensor(neg / max(pos, 1.0)).to(device)

    model = SCDModel(
        input_dim=64,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        backbone=backbone,
        window_duration_s=window_duration_s,
        frame_hop_ms=frame_hop_ms,
        kernel_size=kernel_size,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss(
        # pos_weight=pos_weight, reduction="none"
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # prior_prob = pos / (pos + neg)
    # init_bias = torch.log(torch.tensor(prior_prob / (1 - prior_prob)))
    # torch.nn.init.constant_(model.head.bias, init_bias)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_val_f1, bad_epochs = 0.0, 0

    # Lists to store metrics for plotting
    history_train_loss = []
    history_val_loss = []
    history_train_f1 = []
    history_val_f1 = []

    print("Training on device:", device)
    for epoch in range(50):
        model.train()
        train_tp = train_fp = train_fn = 0
        running_train_loss = 0.0
        # Wrap train_dl with tqdm and include descriptive text
        train_pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/50 [Train]", leave=False)
        for feats, labels, mask, _ in train_pbar:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(feats)
                loss = (criterion(logits, labels) * mask).sum() / mask.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            last_loss = loss.item()

            running_train_loss += loss.item()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5) & mask
                gt = (labels > 0.5) & mask
                train_tp += (preds & gt).sum().item()
                train_fp += (preds & ~gt).sum().item()
                train_fn += (~preds & gt).sum().item()

        epoch_train_loss = running_train_loss / len(train_dl)
        train_p = train_tp / max(train_tp + train_fp, 1)
        train_r = train_tp / max(train_tp + train_fn, 1)
        train_f1 = 2 * train_p * train_r / max(train_p + train_r, 1e-9)

        model.eval()
        val_tp = val_fp = val_fn = 0
        running_val_loss = 0.0

        # Wrap val_dl with tqdm for validation monitoring
        val_pbar = tqdm(val_dl, desc=f"Epoch {epoch+1}/50 [Val]", leave=False)
        with torch.no_grad():
            for feats, labels, mask, _ in val_pbar:
                feats = feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(feats)
                    val_loss = (criterion(logits, labels) * mask).sum() / mask.sum()

                running_val_loss += val_loss.item()
                preds = (torch.sigmoid(logits) > 0.5) & mask
                gt = (labels > 0.5) & mask
                val_tp += (preds & gt).sum().item()
                val_fp += (preds & ~gt).sum().item()
                val_fn += (~preds & gt).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_dl)

        val_p = val_tp / max(val_tp + val_fp, 1)
        val_r = val_tp / max(val_tp + val_fn, 1)
        val_f1 = 2 * val_p * val_r / max(val_p + val_r, 1e-9)

        # Store epoch metrics
        history_train_loss.append(epoch_train_loss)
        history_val_loss.append(epoch_val_loss)
        history_train_f1.append(train_f1)
        history_val_f1.append(val_f1)
        
        print(
            f"epoch {epoch}: train loss={epoch_train_loss:.4f} val loss={epoch_val_loss:.4f}"
            f"train_frame_f1={train_f1:.4f} val_frame_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1, bad_epochs = val_f1, 0
            checkpoint_filename = f"best_{backbone}_{run_id}.pt"
            os.makedirs(os.path.join(work_dir, "checkpoints"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(work_dir, "checkpoints", checkpoint_filename))
            print(f"checkpoint saved")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("early stopping")
                plot_metrics(history_train_loss, history_val_loss, history_train_f1, history_val_f1, work_dir)
                break
        if epoch % 5 == 0:
            plot_metrics(history_train_loss, history_val_loss, history_train_f1, history_val_f1, work_dir)

    plot_metrics(history_train_loss, history_val_loss, history_train_f1, history_val_f1, work_dir)

def plot_metrics(history_train_loss, history_val_loss, history_train_f1, history_val_f1, work_dir):
# Plot loss and F1 scores
    epochs_range = range(1, len(history_train_loss) + 1)
    
    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss History
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_train_loss, label="Train Loss", color="tab:red")
    plt.plot(epochs_range, history_val_loss, label="Val Loss", color="tab:orange")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    # Subplot 2: F1 Score History
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_train_f1, label="Train F1", color="tab:blue")
    plt.plot(epochs_range, history_val_f1, label="Val F1", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Train vs Validation F1 Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(work_dir, "loss_history.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved metrics plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SCD model on fixed-duration windows.")
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="cnn", choices=["gru", "lstm", "cnn"])
    parser.add_argument("--window_duration_s", type=float, default=3.0)
    parser.add_argument("--frame_hop_ms", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--num_layers", type=int, default=None,
        help="For cnn: leave unset to auto-size depth to the window (recommended). "
        "For gru/lstm: defaults to 2 if unset.",
    )
    parser.add_argument("--kernel_size", type=int, default=7, help="cnn backbone only")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    train_model(
        work_dir=args.work_dir,
        backbone=args.backbone,
        window_duration_s=args.window_duration_s,
        frame_hop_ms=args.frame_hop_ms,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
