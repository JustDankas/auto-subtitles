"""
window_sampler.py -- on-the-fly synthetic speaker-change windows (numpy/scipy only).

Replaces the fixed pre-generated window files.  Every call to `sample(rng)` builds a
brand-new `window_s`-second clip from a memory-mapped speaker pool (see build_pool.py):

  kind "change"   : speaker A, then a DIFFERENT speaker B          -> Gaussian bump at t_change
  kind "fake"     : the SAME speaker on both sides (different utterances), with the same
                    cut / gap / overlap / fade statistics as a real change -> label all 0.
                    These are the hard negatives: they stop the net from learning
                    "splice artifact / pause  ==  change" and force it to use voice identity.
  kind "nochange" : one speaker for the whole window                -> label all 0

What is new compared with the pre-generated windows:
  * change position is continuous-uniform in [min_segment_s, window_s - min_segment_s]
  * segments are RANDOM CROPS from anywhere in an utterance (the old generator only ever
    used the first 1-2.5 s of a clip); crops that are mostly silence are re-drawn
  * speed perturbation (x0.9 / x1.1) turns each real speaker into up to 3 "pseudo-speakers"
  * per-segment gain jitter and random fade lengths, applied identically to real and fake
    changes so level jumps / click-free splices cannot leak the label
  * infinite pairs: nothing is ever reused, so nothing can be memorised

Also returns `step` (0 before t_change, 1 after) -- an optional dense auxiliary target.
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

SR = 16000
NOISE_DB = {"zero": None, "mild": -60.0, "moderate": -45.0}


class Pool:
    def __init__(self, pool_path, index_path):
        self.audio = np.load(pool_path, mmap_mode="r")          # int16, flat
        idx = np.load(index_path)
        self.speakers = [str(s) for s in idx["speakers"]]
        self.clip_spk, self.clip_start, self.clip_len = idx["clip_spk"], idx["clip_start"], idx["clip_len"]
        self.spk_clips = [np.flatnonzero(self.clip_spk == i) for i in range(len(self.speakers))]


@dataclass
class SamplerConfig:
    window_s: float = 3.0
    min_segment_s: float = 0.5
    gap_options: tuple = (-0.2, -0.1, 0.0, 0.0, 0.0, 0.1, 0.2)   # <0 overlap, 0 hard cut, >0 silence
    noise_options: tuple = ("zero", "mild", "moderate")
    noise_probs: tuple = (0.5, 0.25, 0.25)
    p_change: float = 0.5
    p_fake: float = 0.2            # same-speaker splice/pause windows (label 0); rest = nochange
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
    def __init__(self, pool, cfg=None):
        self.pool, self.cfg = pool, cfg or SamplerConfig()
        c = self.cfg
        self.n_spk = len(pool.spk_clips)
        if self.n_spk < 2:
            raise ValueError("need at least 2 speakers in the pool")
        if max(abs(g) for g in c.gap_options) > 0.4 * c.min_segment_s:
            raise ValueError("gap_options must stay within 0.4 * min_segment_s so a gap/overlap "
                             "can never eat the guaranteed clean audio on either side")
        self.n_samples = int(round(c.window_s * SR))
        self.n_frames = int(np.ceil(c.window_s * 1000 / c.frame_hop_ms))
        self.frame_t = np.arange(self.n_frames) * (c.frame_hop_ms / 1000.0)

    # ---- audio access -------------------------------------------------------------------
    def _crop(self, spk, n, rng, exclude=None):
        p = self.pool
        cids = p.spk_clips[spk]
        ok = cids[p.clip_len[cids] >= n]
        if exclude is not None and len(ok) > 1:
            ok = ok[ok != exclude]
        if len(ok):
            cid = int(rng.choice(ok))
            a = int(p.clip_start[cid]) + int(rng.integers(0, int(p.clip_len[cid]) - n + 1))
            return np.asarray(p.audio[a:a + n], dtype=np.float32) / 32768.0, cid
        parts, got, first = [], 0, None                        # no single clip long enough: stitch
        while got < n:
            cid = int(rng.choice(cids))
            first = cid if first is None else first
            a, L = int(p.clip_start[cid]), int(p.clip_len[cid])
            parts.append(np.asarray(p.audio[a:a + L], dtype=np.float32) / 32768.0)
            got += L
        return np.concatenate(parts)[:n], first

    def _rand_speed(self, rng):
        c = self.cfg
        return float(rng.choice(c.speed_factors)) if (c.p_speed > 0 and rng.random() < c.p_speed) else 1.0

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

    # ---- one window ---------------------------------------------------------------------
    def sample(self, rng):
        c, W = self.cfg, self.n_samples
        r = rng.random()
        kind = "change" if r < c.p_change else ("fake" if r < c.p_change + c.p_fake else "nochange")
        noise = str(rng.choice(c.noise_options, p=c.noise_probs))
        a = int(rng.integers(self.n_spk))
        b, tc, gap = a, None, 0.0

        if kind == "nochange":
            x, _ = self._segment(a, c.window_s, rng, self._rand_speed(rng))
            canvas = self._level(x, rng).astype(np.float32)
        else:
            tc = float(rng.uniform(c.min_segment_s, c.window_s - c.min_segment_s))
            gap = float(rng.choice(c.gap_options))
            if kind == "change":
                b = int(rng.integers(self.n_spk - 1))
                b += b >= a                                     # any speaker except A
                sa, sb = self._rand_speed(rng), self._rand_speed(rng)
            else:
                sa = sb = self._rand_speed(rng)                 # same pseudo-speaker on both sides
            xa, ca = self._segment(a, tc - gap, rng, sa)
            xb, _ = self._segment(b, c.window_s - tc, rng, sb, exclude=ca if kind == "fake" else None)
            fa, fb = (int(rng.uniform(*c.fade_ms) * SR / 1000) for _ in range(2))
            xa = _fade(self._level(xa, rng), fa, out=True)
            xb = _fade(self._level(xb, rng), fb, out=False)
            canvas = np.zeros(W, np.float32)
            canvas[: len(xa)] += xa[:W]
            s0 = int(round(tc * SR))
            canvas[s0:s0 + len(xb)] += xb[: W - s0]

        db = NOISE_DB[noise]
        if db is not None:
            n = rng.normal(0.0, 1.0, W)
            canvas = canvas + (n * (10 ** (db / 20) / (np.sqrt(np.mean(n ** 2)) + 1e-9))).astype(np.float32)
        m = float(np.abs(canvas).max())
        if m > 0.99:
            canvas = canvas / m * 0.99

        labels = np.zeros(self.n_frames, np.float32)
        step = np.zeros(self.n_frames, np.float32)
        if kind == "change":
            tol = c.tol_ms / 1000.0
            d = np.abs(self.frame_t - tc)
            m_ = d <= tol
            labels[m_] = np.exp(-0.5 * (d[m_] / (tol * c.std_scale)) ** 2)
            step[self.frame_t >= tc] = 1.0
        return dict(wave=canvas.astype(np.float32), labels=labels, step=step, kind=kind,
                    t_change=tc if kind == "change" else np.nan,
                    t_splice=tc if tc is not None else np.nan,
                    spk_a=a, spk_b=b, gap=gap, noise=noise)


def make_fixed_set(sampler, n, seed, path):
    """Freeze a validation/test set once so numbers are comparable across runs and epochs."""
    rng = np.random.default_rng(seed)
    waves = np.zeros((n, sampler.n_samples), np.int16)
    labels = np.zeros((n, sampler.n_frames), np.float32)
    step = np.zeros_like(labels)
    kinds, tch = [], np.zeros(n, np.float32)
    for i in range(n):
        s = sampler.sample(rng)
        waves[i] = np.clip(s["wave"] * 32767.0, -32768, 32767).astype(np.int16)
        labels[i], step[i], tch[i] = s["labels"], s["step"], s["t_change"]
        kinds.append(s["kind"])
    np.savez_compressed(path, waves=waves, labels=labels, step=step, kind=np.array(kinds), t_change=tch)
    print(f"saved {n} windows -> {path}")