"""
train.py -- Step 6 training recipe implementation.
Features: Step-based schedule, vectorised GPU augmentation, curriculum warmup, 
AP-based selection, per-kind diagnostic logging, and checkpointing.
"""
import argparse
import copy
import csv
import math
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from aug_assets import AugmentAssets
from eval_utils import false_alarm_by_kind, full_report
from gpu_aug import GpuAugmentPipeline
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from window_sampler import Pool, SamplerConfig, WindowSampler, make_fixed_set

from scd_model import SCDNet


# ---------------------------------------------------------------------------
# Worker Initialization & Datasets
# ---------------------------------------------------------------------------
def worker_init(worker_id: int):
    """Worker initialization function with start_step offset to guarantee non-repeating streams on resume."""
    torch.set_num_threads(1)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        ds = worker_info.dataset
        start_step = getattr(ds, "start_step", 0)
        seed = (ds.seed + 1000 * start_step + worker_id) & 0xFFFFFFFF
        ds.setup(np.random.default_rng(seed))


class OnTheFlyWindows(Dataset):
    """
    Generates raw un-convolved speaker segments and metadata on every __getitem__ call.
    Pool and WindowSampler are constructed per worker in setup().
    """

    def __init__(
        self,
        pool_dir: str,
        cfg: SamplerConfig,
        bank_dir: Optional[str] = None,
        split: str = "train",
        seed: int = 0,
        no_wave_aug: bool = False,
        start_step: int = 0,
    ):
        self.pool_dir = pool_dir
        self.cfg = cfg
        self.bank_dir = bank_dir
        self.split = split
        self.seed = seed
        self.no_wave_aug = no_wave_aug
        self.start_step = start_step

        self.pool = None
        self.sampler = None
        self.rng = None

    def setup(self, rng: np.random.Generator):
        """Called by worker_init in worker process."""
        self.rng = rng
        self.pool = Pool(f"{self.pool_dir}/pool.npy", f"{self.pool_dir}/index.npz", mmap=True)

        if not self.no_wave_aug:
            bank_path = self.bank_dir if self.bank_dir is not None else "./data/aug_banks"
            assets = AugmentAssets(bank_dir=bank_path, split=self.split)
        else:
            assets = None

        self.sampler = WindowSampler(self.pool, self.cfg, assets=assets)

    def __len__(self):
        return 10**9  # Infinite dataset stream for step-based loop

    def __getitem__(self, idx):
        if self.sampler is None:
            seed = (self.seed + 1000 * self.start_step) & 0xFFFFFFFF
            self.setup(np.random.default_rng(seed))

        s = self.sampler.sample(self.rng, raw=True)
        return s


def raw_collate_fn(batch):
    """Custom collate function to construct PyTorch batch tensors from raw sampler dicts."""
    res = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], np.ndarray):
            res[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
        elif isinstance(batch[0][key], (int, float, bool)):
            res[key] = torch.tensor([b[key] for b in batch])
        else:
            res[key] = [b[key] for b in batch]
    return res


class FixedWindowSet(Dataset):
    """
    Reads a frozen .npz written by window_sampler.make_fixed_set.
    Returns wave, labels, step, kind, rir, snr_db.
    """

    def __init__(self, path: str):
        d = np.load(path)
        self.waves = d["waves"]
        self.labels = d["labels"]
        self.step = d["step"]
        self.kind = d["kind"] if "kind" in d else np.array(["unknown"] * len(self.waves))
        self.rir = d["rir"] if "rir" in d else np.zeros(len(self.waves), dtype=bool)
        self.snr_db = d["snr_db"] if "snr_db" in d else np.full(len(self.waves), np.nan)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        wave = self.waves[idx].astype(np.float32) / 32768.0
        return (
            torch.from_numpy(wave),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.step[idx]),
            str(self.kind[idx]),
            bool(self.rir[idx]),
            float(self.snr_db[idx]),
        )


class EMA:
    """Exponential Moving Average wrapper for model parameters with warmup and BN synchronization."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))  # Warmup
        ep = [p for p in self.model.parameters()]
        mp = [p for p in model.parameters()]
        torch._foreach_lerp_(ep, mp, 1.0 - d)
        for be, bm in zip(self.model.buffers(), model.buffers()):
            be.copy_(bm)

    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(self.model.state_dict())

    def state_dict(self):
        return {"model": self.model.state_dict(), "n": self.n}

    def load_state_dict(self, state_dict):
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            self.n = state_dict.get("n", 0)
        else:
            for name, param in self.model.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name])


def analytic_pos_weight(cfg: SamplerConfig, mode: str) -> float:
    """Analytically computes the frame-wise positive label ratio based on SamplerConfig parameters."""
    n_frames = int(cfg.window_s * 1000 / cfg.frame_hop_ms)
    ft = np.arange(n_frames) * cfg.frame_hop_ms / 1000.0
    tol = cfg.tol_ms / 1000.0

    tc_samples = np.linspace(cfg.min_segment_s, cfg.window_s - cfg.min_segment_s, 2001)

    def integrate_target(tc):
        diff = np.abs(ft - tc)
        mask = diff <= tol
        vals = np.zeros_like(ft)
        vals[mask] = np.exp(-0.5 * ((diff[mask]) / (tol * cfg.std_scale)) ** 2)
        return vals.sum()

    pos_one = np.mean([integrate_target(tc) for tc in tc_samples])
    pos = cfg.p_change * pos_one
    neg = n_frames - pos
    ratio = neg / max(pos, 1e-9)

    return {"sqrt": math.sqrt(ratio), "one": 1.0, "full": ratio}[mode]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def parse_sampler_args(sampler_args: Optional[list]) -> Dict[str, Any]:
    """Parses repeatable --sampler key=value arguments into a dict."""
    overrides = {}
    if not sampler_args:
        return overrides
    for arg in sampler_args:
        if "=" not in arg:
            raise ValueError(f"Invalid --sampler format '{arg}'. Must be key=value.")
        k, v = arg.split("=", 1)
        try:
            if "." in v:
                v_cast = float(v)
            else:
                v_cast = int(v)
        except ValueError:
            if v.lower() in ("true", "false"):
                v_cast = v.lower() == "true"
            else:
                v_cast = v
        overrides[k] = v_cast
    return overrides


def plot_metrics(history: Dict[str, list], work_dir: str, save_name: str, title_suffix: str):
    steps = history["step"]
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(steps, history["train_loss"], label="Train Loss", color="tab:red")
    plt.plot(steps, history["val_aug_loss"], label="Val Aug Loss", color="tab:orange")
    plt.plot(steps, history["val_clean_loss"], label="Val Clean Loss", color="tab:grey", linestyle="--")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.grid(True); plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(steps, history["train_frame_f1"], label="Train frame-F1 (cheap)", color="tab:blue")
    plt.plot(steps, history["val_aug_event_f1_200"], label="Val Aug event-F1 @200ms", color="tab:orange")
    plt.plot(steps, history["val_clean_event_f1_200"], label="Val Clean event-F1 @200ms", color="tab:grey", linestyle="--")
    plt.xlabel("Step"); plt.ylabel("F1"); plt.title("Train frame-F1 vs Val event-F1")
    plt.grid(True); plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(steps, history["val_aug_AP"], label="Val Aug AP", color="tab:green")
    plt.plot(steps, history["val_aug_AUROC"], label="Val Aug AUROC", color="tab:purple")
    plt.xlabel("Step"); plt.ylabel("Score"); plt.title("Val AP / AUROC (Aug)")
    plt.grid(True); plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(steps, history["val_aug_lag_ms"], label="Median lag (ms)", color="tab:brown")
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.xlabel("Step"); plt.ylabel("ms"); plt.title("Val Lag (Aug)")
    plt.grid(True); plt.legend()

    plt.suptitle(f"SCDNet | {title_suffix}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.join(work_dir, "checkpoints"), exist_ok=True)
    path = os.path.join(work_dir, "checkpoints", f"{save_name}.png")
    plt.savefig(path)
    plt.close()


def save_history_csv(history: Dict[str, list], csv_path: str):
    fieldnames = list(history.keys())
    n_rows = len(history["step"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for idx in range(n_rows):
            writer.writerow([history[k][idx] for k in fieldnames])


@torch.no_grad()
def run_validation(model, val_dl, device, aux_weight, pos_weight, use_amp):
    model.eval()
    probs_all, labels_all, loss_sum, n_batches = [], [], 0.0, 0
    kinds_all = []

    for wav, labels, step, kind, rir, snr_db in val_dl:
        wav, labels, step = wav.to(device), labels.to(device), step.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(wav)
            T = min(out.shape[-1], labels.shape[-1])
            out, labels_t, step_t = out[:, :, :T], labels[:, :T], step[:, :T]
            loss = F.binary_cross_entropy_with_logits(out[:, 0], labels_t, pos_weight=pos_weight) + \
                aux_weight * F.binary_cross_entropy_with_logits(out[:, 1], step_t)

        loss_sum += loss.item()
        n_batches += 1

        probs = torch.sigmoid(out[:, 0].float()).cpu().numpy()
        probs_all.append(probs)
        labels_all.append(labels_t.cpu().numpy())
        kinds_all.extend(kind)

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    kinds = np.array(kinds_all)

    report = full_report(probs, labels)

    # Compute per-kind mean probability diagnostic
    mean_probs = {}
    for k in np.unique(kinds):
        mask = (kinds == k)
        mean_probs[str(k)] = float(probs[mask].mean()) if mask.any() else 0.0
    report["mean_probs_by_kind"] = mean_probs

    # Sweep thresholds to select the optimal decision boundary
    best_f1, best_thr = -1.0, 0.5
    best_fa = {}

    for thr in np.arange(0.05, 0.96, 0.05):
        fa_rates = false_alarm_by_kind(probs, kinds, thr=thr)
        cur_f1 = report["event@200ms"]["f1"]
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_thr = float(thr)
            best_fa = fa_rates

    report["fa_by_kind"] = best_fa
    report["best_threshold"] = best_thr

    nochange_count = sum(1 for k in kinds_all if k == "nochange")
    if nochange_count > 0:
        total_nochange_mins = (nochange_count * 3.0) / 60.0
        report["fa_per_min_nochange"] = best_fa.get("nochange", 0) / total_nochange_mins
    else:
        report["fa_per_min_nochange"] = 0.0

    return loss_sum / max(n_batches, 1), report


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_model(
    work_dir: str,
    train_pool_dir: str,
    val_pool_dir: Optional[str] = None,
    val_clean_path: Optional[str] = None,
    val_aug_path: Optional[str] = None,
    bank_dir: Optional[str] = None,
    val_size: int = 3000,
    select_on: str = "aug",
    total_steps: int = 100_000,
    eval_every: int = 2000,
    log_every: int = 50,
    batch_size: int = 64,
    max_lr: float = 2e-3,
    weight_decay: float = 1e-2,
    ch: int = 64,
    kernel_size: int = 5,
    dilations=(1, 2, 4, 8, 16),
    dropout: float = 0.15,
    causal: bool = False,
    dense: bool = False,
    no_specaug: bool = False,
    no_wave_aug: bool = False,
    aux_weight: float = 0.5,
    pos_weight_mode: str = "one",
    patience: int = 8,
    ema_decay: float = 0.999,
    resume_path: Optional[str] = None,
    num_workers: Optional[int] = None,
    prefetch_factor: int = 4,
    device: Optional[str] = None,
    amp_mode: str = "auto",
    seed: int = 0,
    sampler_overrides: Optional[Dict[str, Any]] = None,
):
    # Set framework seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Hardware & Runtime setup
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    if amp_mode == "auto":
        use_amp = (device == "cuda") and (torch.cuda.get_device_capability() >= (7, 0))
    elif amp_mode == "on":
        use_amp = (device == "cuda")
    else:
        use_amp = False
    print(f"Using AMP: {use_amp} | Device: {device} | Seed: {seed}")

    # Worker count clamp to prevent page cache starvation
    if num_workers is None:
        num_workers = min(6, max(1, (os.cpu_count() or 2) - 1))

    pin_memory = (device == "cuda")

    # Sampler Configuration
    sampler_cfg = SamplerConfig()
    if sampler_overrides:
        for k, v in sampler_overrides.items():
            if hasattr(sampler_cfg, k):
                setattr(sampler_cfg, k, v)
            else:
                print(f"Warning: Unknown SamplerConfig key '{k}', skipping.")

    train_ds = OnTheFlyWindows(
        pool_dir=train_pool_dir,
        cfg=sampler_cfg,
        bank_dir=bank_dir,
        split="train",
        seed=seed,
        no_wave_aug=no_wave_aug,
        start_step=0,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_init,
        collate_fn=raw_collate_fn,
    )

    # Vectorised GPU Augmenter
    gpu_aug = GpuAugmentPipeline(
        bank_dir=bank_dir or "./data/aug_banks",
        split="train",
        device=device,
    ) if not no_wave_aug else None

    # Setup Dual Frozen Validation Sets
    val_clean_path = val_clean_path or f"{work_dir}/val_clean.npz"
    val_aug_path = val_aug_path or f"{work_dir}/val_aug.npz"

    if not os.path.exists(val_clean_path) or not os.path.exists(val_aug_path):
        if val_pool_dir is None:
            raise ValueError("Need --val_pool_dir to build val_clean.npz and val_aug.npz validation files.")
        val_pool = Pool(f"{val_pool_dir}/pool.npy", f"{val_pool_dir}/index.npz")

        if not os.path.exists(val_clean_path):
            print(f"Building clean frozen val set: {val_clean_path}")
            val_clean_sampler = WindowSampler(val_pool, SamplerConfig(p_speed=0.0), assets=None)
            make_fixed_set(val_clean_sampler, n=val_size, seed=123, path=val_clean_path)

        if not os.path.exists(val_aug_path):
            print(f"Building augmented frozen val set: {val_aug_path}")
            val_bank_dir = bank_dir or "./data/aug_banks"
            val_assets = AugmentAssets(bank_dir=val_bank_dir, split="val")
            val_aug_sampler = WindowSampler(val_pool, SamplerConfig(p_speed=0.0), assets=val_assets)
            make_fixed_set(val_aug_sampler, n=val_size, seed=123, path=val_aug_path)

    print(f"Loaded frozen validation sets:\n - Clean: {val_clean_path}\n - Aug:   {val_aug_path}")
    val_clean_dl = DataLoader(FixedWindowSet(val_clean_path), batch_size=batch_size * 2, shuffle=False)
    val_aug_dl = DataLoader(FixedWindowSet(val_aug_path), batch_size=batch_size * 2, shuffle=False)

    # Analytic pos_weight
    pos_weight_value = analytic_pos_weight(sampler_cfg, mode=pos_weight_mode)
    print(f"pos_weight ({pos_weight_mode}) = {pos_weight_value:.3f} (analytic)")
    pos_weight = torch.tensor(pos_weight_value, device=device)

    model_config = {
        "n_mels": 64,
        "ch": ch,
        "kernel_size": kernel_size,
        "dilations": dilations,
        "dropout": dropout,
        "causal": causal,
        "dense": dense,
        "augment": not no_specaug, # Ensures frontend.augment is instantiated
        "n_outputs": 2,
    }
    model = SCDNet(**model_config).to(device)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=total_steps, pct_start=0.05, div_factor=20, final_div_factor=100
    )

    start_step = 0
    best_val_ap = -1.0
    bad_evals = 0

    history = {k: [] for k in [
        "step", "train_loss", "train_frame_f1",
        "val_aug_loss", "val_aug_event_f1_200", "val_aug_AP", "val_aug_AUROC", "val_aug_lag_ms",
        "val_clean_loss", "val_clean_event_f1_200", "val_clean_AP", "val_clean_AUROC", "val_clean_lag_ms"
    ]}

    # Resume handling
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        best_val_ap = ckpt["best_val_ap"]
        history = ckpt["history"]
        train_ds.start_step = start_step
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if "rng" in ckpt:
            torch.set_rng_state(ckpt["rng"]["torch"])
            np.random.set_state(ckpt["rng"]["numpy"])
            random.set_state(ckpt["rng"]["python"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"scdnet_{run_id}_blocks{len(dilations)}_k{kernel_size}_causal{causal}_dense{dense}"
    title_suffix = f"blocks={len(dilations)} kernel={kernel_size} causal={causal} dense={dense}"

    ckpt_dir = os.path.join(work_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(ckpt_dir, "history.csv")

    print(f"Training on {device} (AMP={use_amp}) | workers={num_workers} | select_on={select_on} | "
          f"batch_size={batch_size} | total_steps={total_steps} | eval_every={eval_every}")

    train_iter = iter(train_dl)
    model.train()

    # Store initial specaug instance reference if enabled
    specaug_module = model.frontend.augment if not no_specaug else None

    loss_acc = torch.tensor(0.0, device=device)
    tp_acc = torch.tensor(0, dtype=torch.int64, device=device)
    fp_acc = torch.tensor(0, dtype=torch.int64, device=device)
    fn_acc = torch.tensor(0, dtype=torch.int64, device=device)
    acc_steps = 0

    t_data_wait = 0.0
    t_step_proc = 0.0
    t0 = time.perf_counter()

    pbar = tqdm(range(start_step, total_steps), desc="Training", initial=start_step, total=total_steps)
    for step in pbar:
        current_step = step + 1

# Curriculum Warmup: Disable SpecAugment for the first 5000 steps
        if not no_specaug:
            if current_step <= 5000:
                model.frontend.augment = None
            else:
                model.frontend.augment = specaug_module

        batch_raw = next(train_iter)
        t1 = time.perf_counter()
        t_data_wait += (t1 - t0)

        labels = batch_raw["labels"].to(device, non_blocking=pin_memory)
        step_gt = batch_raw["step"].to(device, non_blocking=pin_memory)

        if gpu_aug is not None:
            wav = gpu_aug(batch_raw)
        else:
            wav = (batch_raw["seg_a"] + batch_raw["seg_b"]).to(device, non_blocking=pin_memory)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(wav)
            T = min(out.shape[-1], labels.shape[-1])
            out, labels_t, step_t = out[:, :, :T], labels[:, :T], step_gt[:, :T]
            loss = F.binary_cross_entropy_with_logits(out[:, 0], labels_t, pos_weight=pos_weight) + \
                aux_weight * F.binary_cross_entropy_with_logits(out[:, 1], step_t)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if ema:
            ema.update(model)

        with torch.no_grad():
            loss_acc += loss.detach()
            preds = (out[:, 0] > 0.0)
            gt = (labels_t > 0.5)
            tp_acc += (preds & gt).sum()
            fp_acc += (preds & ~gt).sum()
            fn_acc += (~preds & gt).sum()
            acc_steps += 1

        t2 = time.perf_counter()
        t_step_proc += (t2 - t1)
        t0 = t2

        # Periodic Logging
        if current_step % log_every == 0:
            loss_val = (loss_acc / acc_steps).item()
            tp = tp_acc.item()
            fp = fp_acc.item()
            fn = fn_acc.item()

            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)

            total_time = t_data_wait + t_step_proc
            data_wait_pct = (t_data_wait / max(total_time, 1e-6)) * 100

            pbar.set_postfix(
                loss=f"{loss_val:.3f}",
                f1=f"{f1:.3f}",
                wait=f"{data_wait_pct:.1f}%",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

            current_train_loss = loss_val
            current_train_f1 = f1

            loss_acc.zero_()
            tp_acc.zero_()
            fp_acc.zero_()
            fn_acc.zero_()
            acc_steps = 0
            t_data_wait = 0.0
            t_step_proc = 0.0

        # Periodic Evaluation
        if current_step % eval_every == 0 or current_step == total_steps:
            eval_model = ema.model if ema else model
            eval_model.eval()

            val_aug_loss, aug_report = run_validation(eval_model, val_aug_dl, device, aux_weight, pos_weight, use_amp)
            val_clean_loss, clean_report = run_validation(eval_model, val_clean_dl, device, aux_weight, pos_weight, use_amp)

            history["step"].append(current_step)
            history["train_loss"].append(current_train_loss if 'current_train_loss' in locals() else 0.0)
            history["train_frame_f1"].append(current_train_f1 if 'current_train_f1' in locals() else 0.0)

            history["val_aug_loss"].append(val_aug_loss)
            history["val_aug_event_f1_200"].append(aug_report["event@200ms"]["f1"])
            history["val_aug_AP"].append(aug_report["AP"])
            history["val_aug_AUROC"].append(aug_report["AUROC"])
            history["val_aug_lag_ms"].append(aug_report["lag"].get("median_ms", float("nan")))

            history["val_clean_loss"].append(val_clean_loss)
            history["val_clean_event_f1_200"].append(clean_report["event@200ms"]["f1"])
            history["val_clean_AP"].append(clean_report["AP"])
            history["val_clean_AUROC"].append(clean_report["AUROC"])
            history["val_clean_lag_ms"].append(clean_report["lag"].get("median_ms", float("nan")))

            save_history_csv(history, csv_path)
            plot_metrics(history, work_dir, tag, title_suffix)

            aug_fa = aug_report.get("fa_by_kind", {})
            clean_fa = clean_report.get("fa_by_kind", {})
            aug_probs = aug_report.get("mean_probs_by_kind", {})
            print(
                f"\n[Step {current_step}/{total_steps}]\n"
                f"  [val_aug]   loss={val_aug_loss:.4f} event_f1@200ms={aug_report['event@200ms']['f1']:.4f} AP={aug_report['AP']:.4f} "
                f"FA={aug_fa} mean_probs={aug_probs}\n"
                f"  [val_clean] loss={val_clean_loss:.4f} event_f1@200ms={clean_report['event@200ms']['f1']:.4f} AP={clean_report['AP']:.4f} FA={clean_fa}"
            )

            # Checkpoint: Last State
            last_ckpt = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": current_step,
                "history": history,
                "best_val_ap": best_val_ap,
                "config": model_config,
                "sampler_config": asdict(sampler_cfg),
                "rng": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                }
            }
            if ema:
                last_ckpt["ema"] = ema.state_dict()
            torch.save(last_ckpt, os.path.join(ckpt_dir, "last.pt"))

            # Checkpoint: Best Model (selected via AP)
            selected_val_metric = aug_report["AP"] if select_on == "aug" else clean_report["AP"]

            if selected_val_metric > best_val_ap:
                best_val_ap = selected_val_metric
                bad_evals = 0
                best_ckpt = {
                    "model": ema.model.state_dict() if ema else model.state_dict(),
                    "config": model_config,
                    "sampler_config": asdict(sampler_cfg),
                    "step": current_step,
                    "metrics": aug_report if select_on == "aug" else clean_report,
                    "select_on": select_on,
                    "threshold": aug_report.get("best_threshold", 0.5)
                }
                torch.save(best_ckpt, os.path.join(ckpt_dir, "best.pt"))
                print(f" Saved new best model checkpoint to best.pt (val_{select_on}_AP={best_val_ap:.4f})")
            else:
                bad_evals += 1
                if current_step >= 0.40 * total_steps and bad_evals >= patience:
                    plot_metrics(history, work_dir, tag, title_suffix)
                    print(f"\nEarly stopping triggered at step {current_step} (bad_evals={bad_evals})")
                    break

            model.train()
            t0 = time.perf_counter()

    return os.path.join(ckpt_dir, "best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train SCDNet with step-based schedule and EMA.")
    p.add_argument("--work_dir", type=str, required=True)
    p.add_argument("--train_pool_dir", type=str, default="C:/src/data/pools/train", help="dir with pool.npy + index.npz")
    p.add_argument("--val_pool_dir", type=str, default="C:/src/data/pools/val", help="used only if val npz files don't exist yet")
    p.add_argument("--val_clean_path", type=str, default="C:/src/data/pools/val_clean.npz", help="default: <work_dir>/val_clean.npz")
    p.add_argument("--val_aug_path", type=str, default="C:/src/data/pools/val_aug.npz", help="default: <work_dir>/val_aug.npz")
    p.add_argument("--bank_dir", type=str, default="C:/src/data/aug_banks", help="Directory containing bank folders (e.g. ./data/aug_banks)")
    p.add_argument("--val_size", type=int, default=3000)
    p.add_argument("--select_on", type=str, choices=["aug", "clean"], default="aug", help="Selection metric set")
    p.add_argument("--total_steps", type=int, default=100_000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--dilations", type=str, default="1,2,4,8,16")
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--dense", action="store_true", help="Use standard 1D convs instead of depthwise separable convs")
    p.add_argument("--no_specaug", action="store_true", help="Disable GPU SpecAugment")
    p.add_argument("--no_wave_aug", action="store_true", help="Disable waveform augmentations (reverb + noise)")
    p.add_argument("--aux_weight", type=float, default=0.5)
    p.add_argument("--pos_weight_mode", type=str, default="one", choices=["sqrt", "one", "full"])
    p.add_argument("--patience", type=int, default=8, help="Patience count in evals")
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay rate (0.0 to disable)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint (e.g., last.pt) to resume training")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--amp", type=str, choices=["auto", "on", "off"], default="auto")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", action="append", help="Override SamplerConfig parameters (e.g. --sampler p_rir=0.8)")

    args = p.parse_args()

    sampler_overrides = parse_sampler_args(args.sampler)

    train_model(
        work_dir=args.work_dir,
        train_pool_dir=args.train_pool_dir,
        val_pool_dir=args.val_pool_dir,
        val_clean_path=args.val_clean_path,
        val_aug_path=args.val_aug_path,
        bank_dir=args.bank_dir,
        val_size=args.val_size,
        select_on=args.select_on,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        ch=args.ch,
        kernel_size=args.kernel_size,
        dilations=tuple(int(x) for x in args.dilations.split(",")),
        dropout=args.dropout,
        causal=args.causal,
        dense=args.dense,
        no_specaug=args.no_specaug,
        no_wave_aug=args.no_wave_aug,
        aux_weight=args.aux_weight,
        pos_weight_mode=args.pos_weight_mode,
        patience=args.patience,
        ema_decay=args.ema,
        resume_path=args.resume,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        device=args.device,
        amp_mode=args.amp,
        seed=args.seed,
        sampler_overrides=sampler_overrides,
    )