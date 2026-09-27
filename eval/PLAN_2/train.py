from typing import Literal, Optional

import torch
from dataset import SCDDataset, collate_fn
from model import SCDModel
from torch.utils.data import DataLoader


def train_model(
    work_dir: str, 
    backbone: Literal["gru", "lstm", "cnn"], 
    lr: float = 1e-3,
    num_layers: int = 2,
    patience: int = 5,
    device: Optional[str] = None
    ):    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SCDDataset(f"{work_dir}/data/synth_train")
    val_ds = SCDDataset(f"{work_dir}/data/synth_val")
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Let model overfit to 3 sessions
    # train_ds.session_ids = train_ds.session_ids[:3]
    # val_ds.session_ids = val_ds.session_ids[:3]

    # class imbalance: positives are a thin band around each boundary
    pos = sum(l.sum().item() for _, l in train_ds)
    neg = sum(l.numel() - l.sum().item() for _, l in train_ds)
    pos_weight = torch.tensor(neg / max(pos, 1.0)).to(device)

    model = SCDModel(input_dim=64, hidden_dim=64, num_layers=num_layers, backbone=backbone).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # Fix 3: initialize bias
    prior_prob = pos / (pos + neg)
    init_bias = torch.log(torch.tensor(prior_prob / (1 - prior_prob)))
    torch.nn.init.constant_(model.head.bias, init_bias)
    # Fix 3: initialize bias

    best_val_f1, bad_epochs = 0.0, 0
    for epoch in range(50):
        model.train()
        tp = fp = fn = 0
        for feats, labels, mask, _ in train_dl:
            feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(feats)
                loss = (criterion(logits, labels) * mask).sum() / mask.sum()
            scaler.scale(loss).backward()
            ## Fix 2: gradient clipping
            scaler.unscale_(opt)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            ## Fix 2: gradient clipping
            scaler.step(opt)
            scaler.update()

            # Track train metrics
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5) & mask
                gt = (labels > 0.5) & mask
                tp += (preds & gt).sum().item()
                fp += (preds & ~gt).sum().item()
                fn += (~preds & gt).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        train_f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        # --- validation: frame-level F1 as a fast proxy; run boundary-F1 (Sec. 7) at the end ---
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for feats, labels, mask, _ in val_dl:
                feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
                preds = (torch.sigmoid(model(feats)) > 0.5) & mask
                gt = (labels > 0.5) & mask
                tp += (preds & gt).sum().item()
                fp += (preds & ~gt).sum().item()
                fn += (~preds & gt).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(f"epoch {epoch}: loss={loss.item():.4f} train_frame_f1={train_f1:.4f} val_frame_f1={f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1, bad_epochs = f1, 0
            torch.save(model.state_dict(), f"{work_dir}/checkpoints/best_gru.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("early stopping")
                break
