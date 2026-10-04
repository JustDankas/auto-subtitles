"""
bench_pipeline.py -- Phase 0 throughput and step-time benchmark tool.

Usage:
  python bench_pipeline.py --mode sampler --train_pool_dir ./data/train_pool
  python bench_pipeline.py --mode loader --train_pool_dir ./data/train_pool --workers 4 6 8 10
  python bench_pipeline.py --mode gpu --batch_size 64 128
  python bench_pipeline.py --mode compare --train_pool_dir ./data/train_pool
"""

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SCDNet
from torch.utils.data import DataLoader
from train import OnTheFlyWindows
from window_sampler import Pool, SamplerConfig, WindowSampler

try:
    import psutil
except ImportError:
    psutil = None


def bench_sampler(train_pool_dir, n_samples=200):
    print("\n=== Stage Timing & Single-Process Sampler Benchmark ===")
    pool = Pool(f"{train_pool_dir}/pool.npy", f"{train_pool_dir}/index.npz", mmap=True)
    rng = np.random.default_rng(42)

    configs = {
        "no-aug": SamplerConfig(p_rir=0.0, p_musan=0.0, p_speed=0.0),
        "rir-only": SamplerConfig(p_rir=1.0, p_musan=0.0, p_speed=0.0),
        "musan-only": SamplerConfig(p_rir=0.0, p_musan=1.0, p_speed=0.0),
        "full-aug": SamplerConfig(p_rir=0.5, p_musan=0.5, p_speed=0.5),
    }

    for name, cfg in configs.items():
        sampler = WindowSampler(pool, cfg)
        t0 = time.perf_counter()
        for _ in range(n_samples):
            _ = sampler.sample(rng)
        elapsed = time.perf_counter() - t0
        wps = n_samples / elapsed
        print(f"  [{name:10s}] {n_samples} windows in {elapsed:.2f}s -> {wps:.2f} windows/s")


def bench_loader(train_pool_dir, bank_dir, workers_list, batch_size=64, warmup_batches=20, timed_batches=100):
    print("\n=== Multi-Worker DataLoader Throughput Benchmark ===")
    total_samples = (warmup_batches + timed_batches) * batch_size

    for num_workers in workers_list:
        train_ds = OnTheFlyWindows(cfg=SamplerConfig(), pool_dir=train_pool_dir, bank_dir=bank_dir, split="train")
        dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        iterator = iter(dl)

        # Warmup phase
        for _ in range(warmup_batches):
            _ = next(iterator)

        # Timed phase
        t0 = time.perf_counter()
        for _ in range(timed_batches):
            _ = next(iterator)
        elapsed = time.perf_counter() - t0

        windows_processed = timed_batches * batch_size
        wps = windows_processed / elapsed

        rss_str = "N/A"
        if psutil is not None and num_workers > 0:
            proc = psutil.Process(os.getpid())
            children = proc.children(recursive=True)
            rss_mb = sum(c.memory_info().rss for c in children) / (1024 * 1024)
            rss_str = f"{rss_mb:.1f} MB total worker RSS"

        print(f"  workers={num_workers:2d} | {timed_batches} batches ({windows_processed} windows) in {elapsed:.2f}s -> {wps:.2f} windows/s | RSS: {rss_str}")


def bench_gpu(batch_sizes, n_steps=100):
    print("\n=== GPU Step Time, Precision & Model Architecture Benchmark ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("  CUDA is not available. Skipping GPU benchmark.")
        return

    arch_list = torch.cuda.get_arch_list()
    cap = torch.cuda.get_device_capability()
    print(f"  Device: {torch.cuda.get_device_name(0)} (Capability {cap[0]}.{cap[1]})")
    print(f"  Supported Architectures: {arch_list}")

    # Variations to compare
    model_configs = [
        ("Depthwise (Non-Causal)", False, False),  # dense=False, causal=False
        ("Dense (Non-Causal)",     True,  False),  # dense=True,  causal=False
        ("Depthwise (Causal)",     False, True),   # dense=False, causal=True
        ("Dense (Causal)",         True,  True),   # dense=True,  causal=True
    ]

    for batch_size in batch_sizes:
        for mode_name, use_amp in [("FP32", False), ("AMP (FP16)", True)]:
            for config_label, is_dense, is_causal in model_configs:
                model = SCDNet(
                    n_mels=64,
                    ch=64,
                    kernel_size=5,
                    dilations=(1, 2, 4, 8, 16),
                    n_outputs=2,
                    dense=is_dense,
                    causal=is_causal,
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

                # Dummy input matching waveform dimensions: (B, 48000)
                wav_input = torch.randn(batch_size, 48000, device=device)
                labels = torch.randint(0, 2, (batch_size, 301), device=device, dtype=torch.float32)
                step_target = torch.randint(0, 2, (batch_size, 301), device=device, dtype=torch.float32)

                # Warmup
                for _ in range(10):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out = model(wav_input)
                        T_dim = min(out.shape[-1], labels.shape[-1])
                        loss = F.binary_cross_entropy_with_logits(out[:, 0, :T_dim], labels[:, :T_dim]) + \
                               0.5 * F.binary_cross_entropy_with_logits(out[:, 1, :T_dim], step_target[:, :T_dim])
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                for _ in range(n_steps):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out = model(wav_input)
                        T_dim = min(out.shape[-1], labels.shape[-1])
                        loss = F.binary_cross_entropy_with_logits(out[:, 0, :T_dim], labels[:, :T_dim]) + \
                               0.5 * F.binary_cross_entropy_with_logits(out[:, 1, :T_dim], step_target[:, :T_dim])
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                ms_per_step = (elapsed / n_steps) * 1000.0
                steps_per_sec = n_steps / elapsed
                consumed_wps = batch_size * steps_per_sec

                print(
                    f"  Batch={batch_size:3d} | Precision={mode_name:10s} | "
                    f"Arch={config_label:23s} | {ms_per_step:6.2f} ms/step "
                    f"({steps_per_sec:5.2f} steps/s) -> GPU consumes {consumed_wps:.1f} windows/s"
                )

def compare_pipeline(train_pool_dir, bank_dir, batch_size=64):
    print("\n=== Pipeline Bottleneck Comparison ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("  CUDA required for comparison benchmark.")
        return

    # Benchmark GPU speed (FP32 baseline)
    model = SCDNet(n_mels=64, ch=64, kernel_size=5, dilations=(1, 2, 4, 8, 16), n_outputs=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    wav_input = torch.randn(batch_size, 48000, device=device)
    labels = torch.randint(0, 2, (batch_size, 301), device=device, dtype=torch.float32)

    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        out = model(wav_input)
        loss = F.binary_cross_entropy_with_logits(out[:, 0, :301], labels)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_steps = 100
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(wav_input)
        loss = F.binary_cross_entropy_with_logits(out[:, 0, :301], labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    gpu_elapsed = time.perf_counter() - t0
    gpu_wps = (n_steps * batch_size) / gpu_elapsed

    # Benchmark loader speed (6 workers)
    num_workers = min(6, os.cpu_count() or 1)
    train_ds = OnTheFlyWindows(cfg=SamplerConfig(), pool_dir=train_pool_dir, bank_dir=bank_dir, split="train")
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, persistent_workers=True, prefetch_factor=2)
    iterator = iter(dl)
    for _ in range(10):
        _ = next(iterator)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = next(iterator)
    loader_elapsed = time.perf_counter() - t0
    loader_wps = (50 * batch_size) / loader_elapsed

    print(f"  GPU Max Consumption Rate (FP32, BS={batch_size}): {gpu_wps:.1f} windows/s")
    print(f"  DataLoader Throughput ({num_workers} workers, BS={batch_size}): {loader_wps:.1f} windows/s")

    ratio = loader_wps / gpu_wps
    if ratio >= 1.2:
        verdict = "GPU-bound (DataLoader feeds GPU sufficiently)"
    else:
        verdict = f"Loader-bound (DataLoader throughput is {ratio:.2f}x of GPU demand; target target ≥ 1.2x)"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0 Pipeline Benchmarking")
    parser.add_argument("--mode", choices=["sampler", "loader", "gpu", "compare", "all"], default="all")
    parser.add_argument("--train_pool_dir", type=str, default="./data/pools/train")
    parser.add_argument("--bank_dir", type=str, default="./data/aug_banks")
    parser.add_argument("--workers", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--batch_size", nargs="+", type=int, default=[64, 128])
    args = parser.parse_args()

    if args.mode in ["sampler", "all"]:
        if os.path.exists(f"{args.train_pool_dir}/pool.npy"):
            bench_sampler(args.train_pool_dir)
        else:
            print(f"Skipping sampler benchmark: {args.train_pool_dir}/pool.npy not found.")

    if args.mode in ["loader", "all"]:
        if os.path.exists(f"{args.train_pool_dir}/pool.npy"):
            bench_loader(args.train_pool_dir, args.bank_dir, args.workers)
        else:
            print(f"Skipping loader benchmark: {args.train_pool_dir}/pool.npy not found.")

    if args.mode in ["gpu", "all"]:
        bench_gpu(args.batch_size)

    if args.mode in ["compare", "all"]:
        if os.path.exists(f"{args.train_pool_dir}/pool.npy"):
            compare_pipeline(args.train_pool_dir, args.bank_dir, batch_size=args.batch_size[0])
        else:
            print(f"Skipping compare benchmark: {args.train_pool_dir}/pool.npy not found.")