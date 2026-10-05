"""
aug_assets.py -- High-performance mmap assets and active-channel reverberation.
"""
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve


class AugmentAssets:
    def __init__(self, bank_dir="./data/aug_banks", split="train"):
        d = Path(bank_dir)
        self.rirs = np.load(d / f"rirs_{split}.npy", mmap_mode="r")
        self.noise = {}
        for c in ("music", "noise"):
            idx = np.load(d / f"{c}_{split}_idx.npz")
            # Unpack NPZ arrays into RAM to avoid dictionary/file overhead during sampling
            self.noise[c] = (
                np.load(d / f"{c}_{split}.npy", mmap_mode="r"),
                idx["start"],
                idx["len"]
            )

    def sample_rir_id(self, rng, max_ms=400.0):
        """Samples and returns the integer RIR index for GPU-side gather."""
        return int(rng.integers(len(self.rirs)))

    def get_rir_by_id(self, rir_id, max_ms=400.0):
        """Retrieves a specific RIR by index."""
        n = int(16 * max_ms)
        return np.array(self.rirs[rir_id, :n], dtype=np.float32)

    def sample_rir(self, rng, max_ms=400.0):
        """Samples and returns an RIR array (CPU path)."""
        i = self.sample_rir_id(rng, max_ms=max_ms)
        return self.get_rir_by_id(i, max_ms=max_ms)

    def sample_noise(self, cat, n, rng, tries=3):
        arr, idx_start, idx_len = self.noise[cat]
        n_items = len(idx_len)
        if n_items == 0:
            return np.zeros(n, dtype=np.float32)

        inv_scale = np.float32(1.0 / 32768.0)
        for _ in range(tries):
            k = int(rng.integers(n_items))
            s, L = int(idx_start[k]), int(idx_len[k])
            if L >= n:
                a = s + int(rng.integers(0, L - n + 1))
                x = arr[a:a + n].astype(np.float32) * inv_scale
            else:
                x = np.resize(arr[s:s + L], n).astype(np.float32) * inv_scale
            
            # Avoid np.sqrt by checking mean squared power against (1e-4)^2
            if np.mean(x * x) > 1e-8:
                return x
        return x


def apply_reverb(channel, rir, active_range=None, wet_ratio=0.3):
    rir_copy = rir.copy()
    fade_len = min(160, len(rir_copy))
    if fade_len > 0:
        rir_copy[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        
    wet = fftconvolve(channel, rir_copy)[:len(channel)]
    
    if active_range is not None:
        i0, i1 = active_range
        i1 = min(i1, len(channel))
        dry_rms = np.sqrt(np.mean(channel[i0:i1] ** 2) + 1e-9)
        wet_rms = np.sqrt(np.mean(wet[i0:i1] ** 2) + 1e-9)
    else:
        dry_rms = np.sqrt(np.mean(channel ** 2) + 1e-9)
        wet_rms = np.sqrt(np.mean(wet ** 2) + 1e-9)
        
    wet = wet * (dry_rms / wet_rms)
    return (1.0 - wet_ratio) * channel + wet_ratio * wet


def mix_background_noise(signal, noise, snr_db=20.0):
    sig_pwr = np.mean(signal ** 2)
    noise_pwr = np.mean(noise ** 2)
    
    if sig_pwr < 1e-9 or noise_pwr < 1e-9:
        return signal
        
    target_noise_pwr = sig_pwr / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_pwr / noise_pwr)
    return signal + noise * scale