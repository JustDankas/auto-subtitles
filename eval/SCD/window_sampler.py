"""
window_sampler.py -- On-the-fly synthetic speaker-change windows (numpy/scipy only).

Replaces the fixed pre-generated window files. Every call to `sample(rng)` builds a
brand-new `window_s`-second clip from a memory-mapped speaker pool (see build_pools.py):

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
from scipy.signal import firwin, resample_poly

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

        # Load speaker gender metadata if present (defaults to unknown 'U')
        if "speaker_genders" in idx:
            self.speaker_genders = np.array([
                g.decode("utf-8") if isinstance(g, bytes) else str(g)
                for g in idx["speaker_genders"]
            ])
        else:
            self.speaker_genders = np.array(["U"] * len(self.speakers))

        # Pre-group indices by gender for fast same-gender lookup ('M', 'F')
        self.gender_to_spk_indices = {
            "M": np.flatnonzero(self.speaker_genders == "M"),
            "F": np.flatnonzero(self.speaker_genders == "F"),
        }

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
    p_same_gender: float = 0.5               # Chance to draw same-gender speaker B for change windows
    
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


def _active_ratio(x, frame=400):
    n = len(x) // frame
    if n == 0:
        return 0.0
    # Compute mean square power per frame
    m = np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)
    k = min(n - 1, int(0.95 * n))
    p95_sq = np.partition(m, k)[k]
    thresh = max(0.01 * p95_sq, 1e-10)
    return float(np.mean(m > thresh))


def _fade(x, n, out):
    n = min(int(n), len(x))
    if n <= 0:
        return x
    # Fade in-place on the allocated x array
    ramp = np.linspace(1.0 if out else 0.0, 0.0 if out else 1.0, n, dtype=np.float32)
    if out:
        x[-n:] *= ramp
    else:
        x[:n] *= ramp
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

        # Precompute label parameters
        self.tol = c.tol_ms / 1000.0
        self.inv_denom_sq = -0.5 / ((self.tol * c.std_scale) ** 2)

        # Precompute resample FIR filters for speed factors
        self.speed_resamplers = {}
        for s in set(c.speed_factors):
            f = Fraction(1.0 / s).limit_denominator(20)
            up, down = f.numerator, f.denominator
            max_rate = max(up, down)
            f_c = 1.0 / max_rate
            half_len = 10 * max_rate
            filt = firwin(2 * half_len + 1, f_c, window=("kaiser", 5.0))
            self.speed_resamplers[s] = (up, down, filt)

    def _crop(self, spk, n, rng, exclude=None):
        p = self.pool
        cids = p.spk_clips[spk]
        ok = cids[p.clip_len[cids] >= n]
        if exclude is not None and len(ok) > 1:
            ok = ok[ok != exclude]
        inv_scale = np.float32(1.0 / 32768.0)
        if len(ok) > 0:
            cid = int(ok[int(rng.integers(len(ok)))])
            a = int(p.clip_start[cid]) + int(rng.integers(0, int(p.clip_len[cid]) - n + 1))
            return p.audio[a:a + n].astype(np.float32) * inv_scale, cid
            
        parts, got, first = [], 0, None
        while got < n:
            cid = int(cids[int(rng.integers(len(cids)))])
            first = cid if first is None else first
            a, L = int(p.clip_start[cid]), int(p.clip_len[cid])
            parts.append(p.audio[a:a + L].astype(np.float32) * inv_scale)
            got += L
        return np.concatenate(parts)[:n], first

    def _rand_speed(self, rng):
        c = self.cfg
        if c.p_speed > 0 and rng.random() < c.p_speed:
            idx = int(rng.integers(len(c.speed_factors)))
            return float(c.speed_factors[idx])
        return 1.0

    def _change_speed(self, x, s):
        up, down, filt = self.speed_resamplers[s]
        return resample_poly(x, up, down, window=filt).astype(np.float32)

    def _segment(self, spk, dur_s, rng, speed=1.0, exclude=None):
        need = int(round(dur_s * SR))
        src = int(np.ceil(need * speed)) + 8
        for _ in range(6):
            x, cid = self._crop(spk, src, rng, exclude)
            if _active_ratio(x) >= self.cfg.min_active_ratio:
                break
        if speed != 1.0:
            x = self._change_speed(x, speed)
        return _fit(x, need), cid

    def _level(self, x, rng):
        rms = np.sqrt(np.mean(x ** 2)) + 1e-9
        # 10 ** (-23 / 20) == 0.07079457843841379
        base_scale = 0.07079457843841379 / rms
        j = self.cfg.gain_jitter_db
        if j > 0:
            scale = base_scale * (10.0 ** (rng.uniform(-j, j) * 0.05))
        else:
            scale = base_scale
        return x * scale

    def sample(self, rng, raw=False):
        c, W = self.cfg, self.n_samples
        r = rng.random()
        kind = "change" if r < c.p_change else ("fake" if r < c.p_change + c.p_fake else "nochange")
        
        a = int(rng.integers(self.n_spk))
        b, tc, gap = a, None, 0.0

        apply_rir_flag = (self.assets is not None) and (rng.random() < c.p_rir)
        rir_same = False

        # Prepare arrays for channel A and channel B
        seg_a = np.zeros(W, dtype=np.float32)
        seg_b = np.zeros(W, dtype=np.float32)
        range_a = [0, W]
        range_b = [0, W]

        rir_a_id = -1
        rir_b_id = -1
        wet_a = 0.0
        wet_b = 0.0

        if kind == "nochange":
            x, _ = self._segment(a, c.window_s, rng, self._rand_speed(rng))
            x = self._level(x, rng)
            seg_a[:len(x)] = x[:W]
            range_a = [0, len(x)]

            if apply_rir_flag:
                wet_a = float(rng.uniform(*c.rir_wet_range))
                if raw:
                    rir_a_id = self.assets.sample_rir_id(rng, max_ms=c.rir_max_ms)
                else:
                    rir_a = self.assets.sample_rir(rng, max_ms=c.rir_max_ms)
                    seg_a = apply_reverb(seg_a, rir_a, active_range=(0, len(x)), wet_ratio=wet_a)

        else:
            tc = float(rng.uniform(c.min_segment_s, c.window_s - c.min_segment_s))
            gap_idx = int(rng.integers(len(c.gap_options)))
            gap = float(c.gap_options[gap_idx])
            
            # Global speed factor per window across both sides to prevent label leaks
            sa = sb = self._rand_speed(rng)

            if kind == "change":
                spk_a_gender = self.pool.speaker_genders[a]
                same_gender_candidates = self.pool.gender_to_spk_indices.get(spk_a_gender, np.array([], dtype=int))
                same_gender_candidates = same_gender_candidates[same_gender_candidates != a]

                # Sample same-gender speaker with probability p_same_gender if candidates exist
                if (
                    c.p_same_gender > 0
                    and len(same_gender_candidates) > 0
                    and rng.random() < c.p_same_gender
                ):
                    b = int(same_gender_candidates[int(rng.integers(len(same_gender_candidates)))])
                else:
                    b = int(rng.integers(self.n_spk - 1))
                    b += b >= a

            xa, ca = self._segment(a, tc - gap, rng, sa)
            xb, _ = self._segment(b, c.window_s - tc, rng, sb, exclude=ca if kind == "fake" else None)
            
            fa, fb = (int(rng.uniform(*c.fade_ms) * SR / 1000) for _ in range(2))
            xa = _fade(self._level(xa, rng), fa, out=True)
            xb = _fade(self._level(xb, rng), fb, out=False)

            len_a = min(len(xa), W)
            seg_a[:len_a] = xa[:len_a]
            range_a = [0, len_a]
            
            s0 = int(round(tc * SR))
            len_b = min(len(xb), W - s0)
            if len_b > 0 and s0 < W:
                seg_b[s0:s0 + len_b] = xb[:len_b]
                range_b = [s0, s0 + len_b]

            if apply_rir_flag:
                wet_a = wet_b = float(rng.uniform(*c.rir_wet_range))
                if raw:
                    rir_a_id = self.assets.sample_rir_id(rng, max_ms=c.rir_max_ms)
                    if rng.random() < c.p_same_rir:
                        rir_b_id = rir_a_id
                        rir_same = True
                    else:
                        rir_b_id = self.assets.sample_rir_id(rng, max_ms=c.rir_max_ms)
                else:
                    rir_a = self.assets.sample_rir(rng, max_ms=c.rir_max_ms)
                    if rng.random() < c.p_same_rir:
                        rir_b = rir_a
                        rir_same = True
                    else:
                        rir_b = self.assets.sample_rir(rng, max_ms=c.rir_max_ms)

                    seg_a = apply_reverb(seg_a, rir_a, active_range=(0, len_a), wet_ratio=wet_a)
                    seg_b = apply_reverb(seg_b, rir_b, active_range=(s0, s0 + len_b), wet_ratio=wet_b)

        # Background Noise / Music Selection
        snr_db = np.nan
        noise_cat = "none"
        noise = np.zeros(W, dtype=np.float32)

        if (self.assets is not None) and (rng.random() < c.p_musan):
            cat_idx = int(rng.integers(len(c.musan_types)))
            noise_cat = str(c.musan_types[cat_idx])
            snr_db = float(rng.uniform(*c.musan_snr_db_range))
            noise = self.assets.sample_noise(noise_cat, W, rng)

        # Labels Generation
        labels = np.zeros(self.n_frames, np.float32)
        step = np.zeros(self.n_frames, np.float32)
        if kind == "change":
            d = np.abs(self.frame_t - tc)
            m_ = d <= self.tol
            labels[m_] = np.exp((d[m_] ** 2) * self.inv_denom_sq)
            step[self.frame_t >= tc] = 1.0

        if raw:
            return dict(
                seg_a=seg_a,
                seg_b=seg_b,
                range_a=np.array(range_a, dtype=np.int64),
                range_b=np.array(range_b, dtype=np.int64),
                rir_a_id=rir_a_id,
                rir_b_id=rir_b_id,
                wet_a=wet_a,
                wet_b=wet_b,
                noise=noise,
                snr_db=snr_db,
                labels=labels,
                step=step,
                kind=kind,
                t_change=tc if kind == "change" else np.nan,
                spk_a=a,
                spk_b=b,
                gap=gap,
                rir=apply_rir_flag,
                rir_same=rir_same,
                noise_cat=noise_cat,
            )

        # CPU Path Output (Fully blended waveform)
        canvas = seg_a + seg_b
        if not np.isnan(snr_db):
            canvas = mix_background_noise(canvas, noise, snr_db=snr_db)

        # Peak normalization guard
        m = float(np.abs(canvas).max())
        if m > 0.99:
            canvas = canvas / m * 0.99

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
        s = sampler.sample(rng, raw=False)
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