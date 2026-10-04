"""
window_sampler.py -- On-the-fly synthetic speaker-change windows (numpy/scipy only).

Replaces the fixed pre-generated window files. Every call to `sample(rng)` builds a
brand-new `window_s`-second clip from a memory-mapped speaker pool (see build_pool.py):

  kind "change"   : speaker A, then a DIFFERENT speaker B          -> Gaussian bump at t_change
  kind "fake"     : the SAME speaker on both sides (different utterances), with the same
                    cut / gap / overlap / fade statistics as a real change -> label all 0.
                    These are the hard negatives: they stop the net from learning
                    "splice artifact / pause  ==  change" and force it to use voice identity.
  kind "nochange" : one speaker for the whole window                -> label all 0
"""
import warnings
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from aug_assets import AugmentAssets, apply_reverb, mix_background_noise
from scipy.signal import resample_poly

SR = 16000


class Pool:
    def __init__(self, pool_path, index_path, mmap=True):
        # Load whole 5 GB array into RAM if mmap=False
        if mmap:
            self.audio = np.load(pool_path, mmap_mode="r")
        else:
            self.audio = np.load(pool_path)  # Loads flat array directly into RAM        
        
        idx = np.load(index_path)
        self.speakers = [str(s) for s in idx["speakers"]]
        self.clip_spk = idx["clip_spk"]
        self.clip_start = idx["clip_start"]
        self.clip_len = idx["clip_len"]
        self.spk_clips = [np.flatnonzero(self.clip_spk == i) for i in range(len(self.speakers))]

        # Pool Diagnostics
        min_samples = int(3.0 * SR)
        stitch_fallback_count = 0
        
        for i, spk_name in enumerate(self.speakers):
            cids = self.spk_clips[i]
            lens = self.clip_len[cids]
            
            if not np.any(lens >= min_samples):
                warnings.warn(f"Speaker '{spk_name}' (index {i}) has no clips >= 3.0s!")
                
            if not np.any(lens >= int(3.5 * SR)):
                stitch_fallback_count += 1
                
        if stitch_fallback_count > 0:
            print(f"Pool Diagnostic: {stitch_fallback_count} speaker(s) may trigger stitch-fallback for >3.5s segments.")


@dataclass
class SamplerConfig:
    window_s: float = 3.0
    min_segment_s: float = 0.5
    gap_options: tuple = (-0.2, -0.1, 0.0, 0.0, 0.0, 0.1, 0.2)

    # Reverb Controls
    p_rir: float = 0.5
    p_same_rir: float = 0.5                  # 50% chance to reuse same RIR for both sides
    rir_max_ms: float = 400.0
    rir_wet_range: tuple = (0.2, 0.6)        # Wet ratio: 0.2 to 0.6
    
    # MUSAN Noise Controls
    p_musan: float = 0.5
    musan_types: tuple = ("music", "noise")
    musan_snr_db_range: tuple = (10.0, 28.0) # Intended SNR range (10 to 28 dB)
    
    # Prior probabilities
    p_change: float = 0.4
    p_fake: float = 0.4                      # Rest (0.2) = nochange
    
    speed_factors: tuple = (0.9, 1.1)
    p_speed: float = 0.5           # chance a segment becomes a speed-perturbed pseudo-speaker (0 for val/test)
    gain_jitter_db: float = 4.0
    fade_ms: tuple = (5.0, 25.0)
    frame_hop_ms: float = 10.0
    tol_ms: float = 150.0
    std_scale: float = 0.5
    min_active_ratio: float = 0.6  # re-draw crops that are mostly silence


def _fit(x, n):
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def _active_ratio(x, frame=400):
    n = len(x) // frame
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    return float(np.mean(rms > max(0.1 * np.percentile(rms, 95), 1e-5)))


def _change_speed(x, s):
    f = Fraction(1.0 / s).limit_denominator(20)                 # 0.9 -> 10/9, 1.1 -> 10/11
    return resample_poly(x, f.numerator, f.denominator).astype(np.float32)


def _fade(x, n, out):
    n = min(int(n), len(x))
    if n <= 0:
        return x
    x = x.copy()
    if out:
        x[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    else:
        x[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return x


class WindowSampler:
    def __init__(self, pool, cfg=None, assets: AugmentAssets = None):
        self.pool = pool
        self.cfg = cfg or SamplerConfig()
        self.assets = assets
        c = self.cfg
        self.n_spk = len(pool.spk_clips)
        if self.n_spk < 2:
            raise ValueError("Need at least 2 speakers in the pool")
        if max(abs(g) for g in c.gap_options) > 0.4 * c.min_segment_s:
            raise ValueError("gap_options must stay within 0.4 * min_segment_s")
            
        self.n_samples = int(round(c.window_s * SR))
        self.n_frames = int(np.ceil(c.window_s * 1000 / c.frame_hop_ms))
        self.frame_t = np.arange(self.n_frames) * (c.frame_hop_ms / 1000.0)

    def _crop(self, spk, n, rng, exclude=None):
        p = self.pool
        cids = p.spk_clips[spk]
        ok = cids[p.clip_len[cids] >= n]
        if exclude is not None and len(ok) > 1:
            ok = ok[ok != exclude]
        if len(ok) > 0:
            cid = int(ok[int(rng.integers(len(ok)))])
            a = int(p.clip_start[cid]) + int(rng.integers(0, int(p.clip_len[cid]) - n + 1))
            return np.asarray(p.audio[a:a + n], dtype=np.float32) / 32768.0, cid
            
        parts, got, first = [], 0, None
        while got < n:
            cid = int(cids[int(rng.integers(len(cids)))])
            first = cid if first is None else first
            a, L = int(p.clip_start[cid]), int(p.clip_len[cid])
            parts.append(np.asarray(p.audio[a:a + L], dtype=np.float32) / 32768.0)
            got += L
        return np.concatenate(parts)[:n], first

    def _rand_speed(self, rng):
        c = self.cfg
        if c.p_speed > 0 and rng.random() < c.p_speed:
            idx = int(rng.integers(len(c.speed_factors)))
            return float(c.speed_factors[idx])
        return 1.0

    def _segment(self, spk, dur_s, rng, speed=1.0, exclude=None):
        need = int(round(dur_s * SR))
        src = int(np.ceil(need * speed)) + 8
        for _ in range(6):
            x, cid = self._crop(spk, src, rng, exclude)
            if _active_ratio(x) >= self.cfg.min_active_ratio:
                break
        if speed != 1.0:
            x = _change_speed(x, speed)
        return _fit(x, need), cid

    def _level(self, x, rng):
        x = x * 10 ** ((-23.0 - _rms_db(x)) / 20)
        j = self.cfg.gain_jitter_db
        return x * 10 ** (rng.uniform(-j, j) / 20) if j > 0 else x

    def sample(self, rng):
        c, W = self.cfg, self.n_samples
        r = rng.random()
        kind = "change" if r < c.p_change else ("fake" if r < c.p_change + c.p_fake else "nochange")
        
        a = int(rng.integers(self.n_spk))
        b, tc, gap = a, None, 0.0

        # --- 1. Segment Generation & RIR Reverberation Strategy ---
        apply_rir_flag = (self.assets is not None) and (rng.random() < c.p_rir)
        rir_same = False

        if kind == "nochange":
            x, _ = self._segment(a, c.window_s, rng, self._rand_speed(rng))
            x = self._level(x, rng).astype(np.float32)
            
            if apply_rir_flag:
                rir_a = self.assets.sample_rir(rng, max_ms=c.rir_max_ms)
                wet_ratio = float(rng.uniform(*c.rir_wet_range))
                canvas = apply_reverb(x, rir_a, active_range=(0, len(x)), wet_ratio=wet_ratio)
            else:
                canvas = x

        else:
            tc = float(rng.uniform(c.min_segment_s, c.window_s - c.min_segment_s))
            gap_idx = int(rng.integers(len(c.gap_options)))
            gap = float(c.gap_options[gap_idx])
            
            if kind == "change":
                b = int(rng.integers(self.n_spk - 1))
                b += b >= a
                sa, sb = self._rand_speed(rng), self._rand_speed(rng)
            else:
                sa = sb = self._rand_speed(rng)

            xa, ca = self._segment(a, tc - gap, rng, sa)
            xb, _ = self._segment(b, c.window_s - tc, rng, sb, exclude=ca if kind == "fake" else None)
            
            fa, fb = (int(rng.uniform(*c.fade_ms) * SR / 1000) for _ in range(2))
            xa = _fade(self._level(xa, rng), fa, out=True)
            xb = _fade(self._level(xb, rng), fb, out=False)

            ca_arr = np.zeros(W, dtype=np.float32)
            cb_arr = np.zeros(W, dtype=np.float32)

            len_a = min(len(xa), W)
            ca_arr[:len_a] = xa[:len_a]
            
            s0 = int(round(tc * SR))
            len_b = min(len(xb), W - s0)
            if len_b > 0 and s0 < W:
                cb_arr[s0:s0 + len_b] = xb[:len_b]

            if apply_rir_flag:
                wet_ratio = float(rng.uniform(*c.rir_wet_range))
                rir_a = self.assets.sample_rir(rng, max_ms=c.rir_max_ms)
                
                if rng.random() < c.p_same_rir:
                    rir_b = rir_a
                    rir_same = True
                else:
                    rir_b = self.assets.sample_rir(rng, max_ms=c.rir_max_ms)

                ca_arr = apply_reverb(ca_arr, rir_a, active_range=(0, len_a), wet_ratio=wet_ratio)
                cb_arr = apply_reverb(cb_arr, rir_b, active_range=(s0, s0 + len_b), wet_ratio=wet_ratio)

            canvas = ca_arr + cb_arr

        # Background Noise / Music Injection
        snr_db = np.nan
        noise_cat = "none"
        if (self.assets is not None) and (rng.random() < c.p_musan):
            cat_idx = int(rng.integers(len(c.musan_types)))
            noise_cat = str(c.musan_types[cat_idx])
            snr_db = float(rng.uniform(*c.musan_snr_db_range))
            bg_noise = self.assets.sample_noise(noise_cat, W, rng)
            canvas = mix_background_noise(canvas, bg_noise, snr_db=snr_db)

        # Peak normalization guard
        m = float(np.abs(canvas).max())
        if m > 0.99:
            canvas = canvas / m * 0.99

        # Labels Generation
        labels = np.zeros(self.n_frames, np.float32)
        step = np.zeros(self.n_frames, np.float32)
        if kind == "change":
            tol = c.tol_ms / 1000.0
            d = np.abs(self.frame_t - tc)
            m_ = d <= tol
            labels[m_] = np.exp(-0.5 * (d[m_] / (tol * c.std_scale)) ** 2)
            step[self.frame_t >= tc] = 1.0

        return dict(
            wave=canvas.astype(np.float32),
            labels=labels,
            step=step,
            kind=kind,
            t_change=tc if kind == "change" else np.nan,
            spk_a=a,
            spk_b=b,
            gap=gap,
            rir=apply_rir_flag,
            rir_same=rir_same,
            snr_db=snr_db,
            noise_cat=noise_cat
        )


def make_fixed_set(sampler, n, seed, path):
    """Freeze a validation/test set with complete condition metadata."""
    rng = np.random.default_rng(seed)
    waves = np.zeros((n, sampler.n_samples), np.int16)
    labels = np.zeros((n, sampler.n_frames), np.float32)
    step = np.zeros_like(labels)
    
    kinds = []
    tch = np.zeros(n, np.float32)
    gaps = np.zeros(n, np.float32)
    spk_a = np.zeros(n, np.int32)
    spk_b = np.zeros(n, np.int32)
    rirs = np.zeros(n, np.bool_)
    snr_dbs = np.zeros(n, np.float32)

    for i in range(n):
        s = sampler.sample(rng)
        waves[i] = np.clip(s["wave"] * 32767.0, -32768, 32767).astype(np.int16)
        labels[i] = s["labels"]
        step[i] = s["step"]
        tch[i] = s["t_change"]
        kinds.append(s["kind"])
        gaps[i] = s["gap"]
        spk_a[i] = s["spk_a"]
        spk_b[i] = s["spk_b"]
        rirs[i] = s["rir"]
        snr_dbs[i] = s["snr_db"]

    np.savez_compressed(
        path,
        waves=waves,
        labels=labels,
        step=step,
        kind=np.array(kinds),
        t_change=tch,
        gap=gaps,
        spk_a=spk_a,
        spk_b=spk_b,
        rir=rirs,
        snr_db=snr_dbs
    )
    print(f"Saved {n} windows to {path}")