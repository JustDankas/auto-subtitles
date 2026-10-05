"""
test_sampler.py -- Generates sample WAVs to verify WindowSampler and AugmentAssets pipeline.
"""
import argparse
import wave
from pathlib import Path

import numpy as np
from aug_assets import AugmentAssets
from window_sampler import Pool, SamplerConfig, WindowSampler


def save_wav(path: str, signal: np.ndarray, sample_rate: int = 16000):
    """Saves a 1D float32 numpy array (-1.0 to 1.0) as a 16-bit WAV file."""
    signal_clipped = np.clip(signal, -0.99, 0.99)
    pcm16 = (signal_clipped * 32767.0).astype(np.int16)
    
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)        # 16-bit (2 bytes)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def generate_audio_samples(
    train_pool_dir: str,
    bank_dir: str,
    output_dir: str = "./test_audio_outputs",
    split: str = "train",
    num_samples: int = 10,
    seed: int = 42,
    rir_max_ms: float = 400.0,
    rir_wet_range: tuple[float, float] = (0.2, 0.6),
    musan_snr_db_range: tuple[float, float] = (10.0, 28.0),
):
    """Generates and saves test WAV files to inspect window sampler and augmentations."""
    out_path_dir = Path(output_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("Loading pools and mmap assets...")
    pool = Pool(f"{train_pool_dir}/pool.npy", f"{train_pool_dir}/index.npz")
    assets = AugmentAssets(bank_dir=bank_dir, split=split)

    # Initialize sampler with updated SamplerConfig parameters
    cfg = SamplerConfig(
        p_rir=0.5,
        rir_max_ms=rir_max_ms,
        rir_wet_range=rir_wet_range,
        p_musan=0.5,
        musan_snr_db_range=musan_snr_db_range,
    )
    sampler = WindowSampler(pool, cfg, assets=assets)

    print(f"\nGenerating {num_samples} test audio clips to '{output_dir}'...\n")
    print(
        f"{'Filename':<22} | {'Kind':<8} | {'t_change':<8} | {'RIR':<5} | "
        f"{'SameRIR':<7} | {'NoiseCat':<8} | {'SNR (dB)':<8} | {'RMS (dB)':<8}"
    )
    print("-" * 95)

    for i in range(num_samples):
        sample = sampler.sample(rng)
        audio = sample["wave"]
        kind = sample["kind"]
        t_change = sample["t_change"]
        rir_applied = sample["rir"]
        rir_same = sample["rir_same"]
        noise_cat = sample["noise_cat"]
        snr_db = sample["snr_db"]

        # Calculate signal statistics
        rms_val = np.sqrt(np.mean(audio ** 2)) + 1e-9
        rms_db = 20 * np.log10(rms_val)

        tc_str = f"{t_change:.2f}s" if not np.isnan(t_change) else "N/A"
        snr_str = f"{snr_db:.1f}" if not np.isnan(snr_db) else "N/A"
        
        filename = f"sample_{i+1:02d}_{kind}.wav"
        out_file_path = out_path_dir / filename

        save_wav(str(out_file_path), audio, sample_rate=16000)

        print(
            f"{filename:<22} | {kind:<8} | {tc_str:<8} | {str(rir_applied):<5} | "
            f"{str(rir_same):<7} | {noise_cat:<8} | {snr_str:<8} | {rms_db:<8.1f}"
        )

    print("\nGeneration complete. Audio files are ready for manual inspection.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test window_sampler.py outputs with mmap augmentations.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--num-samples", type=int, default=10, help="Number of audio samples to generate.")
    p.add_argument("--pool_dir", type=str, default="./data/pools/val", help="Path to pool directory.")
    p.add_argument("--bank_dir", type=str, default="./data/aug_banks", help="Path to bank directory.")
    p.add_argument("--split", type=str, default="train", help="Augmentation split ('train' or 'val').")
    p.add_argument("--musan_snr_db_range", nargs=2, type=float, default=(10.0, 28.0), help="SNR range for MUSAN noise.")
    p.add_argument("--rir_max_ms", type=float, default=400.0, help="Maximum RIR duration in milliseconds.")
    p.add_argument("--rir_wet_range", nargs=2, type=float, default=(0.2, 0.6), help="Wet ratio range for RIR augmentation.")

    args = p.parse_args()

    generate_audio_samples(
        train_pool_dir=args.pool_dir,
        bank_dir=args.bank_dir,
        output_dir="./test_audio_outputs",
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        rir_max_ms=args.rir_max_ms,
        rir_wet_range=tuple(args.rir_wet_range),
        musan_snr_db_range=tuple(args.musan_snr_db_range),
    )