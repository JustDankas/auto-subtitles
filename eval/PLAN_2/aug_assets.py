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
            self.noise[c] = (
                np.load(d / f"{c}_{split}.npy", mmap_mode="r"),
                np.load(d / f"{c}_{split}_idx.npz")
            )

    def sample_rir(self, rng, max_ms=400.0):
        i = int(rng.integers(len(self.rirs)))
        n = int(16 * max_ms)
        return np.array(self.rirs[i, :n], dtype=np.float32)

    def sample_noise(self, cat, n, rng, tries=3):
        arr, idx = self.noise[cat]
        if len(idx["len"]) == 0:
            return np.zeros(n, dtype=np.float32)
            
        for _ in range(tries):
            k = int(rng.integers(len(idx["len"])))
            s, L = int(idx["start"][k]), int(idx["len"][k])
            if L >= n:
                a = s + int(rng.integers(0, L - n + 1))
                x = arr[a:a + n]
            else:
                x = np.resize(arr[s:s + L], n)
            x = x.astype(np.float32) / 32768.0
            if np.sqrt(np.mean(x * x)) > 1e-4:
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