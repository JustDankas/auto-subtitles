"""
build_pools.py -- flatten <root>/<speaker>/**/*.{flac,wav} into ONE int16 array + an index.

Why: the on-the-fly sampler needs cheap random access to many speakers.  Reading
thousands of small FLACs per epoch from Google Drive is slow; one memory-mapped
int16 file is not.  Build it once per split (train / val / test), copy the two
output files to Drive, and delete the raw corpus.

Usage (LibriSpeech layout: <subset>/<speaker>/<chapter>/<utt>.flac):
  python build_pools.py --roots LibriSpeech/train-clean-100 LibriSpeech/train-clean-360 \
      --out-dir pools/train --max-clips-per-speaker 24 --max-clip-s 8
  python build_pools.py --roots LibriSpeech/dev-clean  --out-dir pools/val
  python build_pools.py --roots LibriSpeech/test-clean --out-dir pools/test

Size estimate = speakers x clips x seconds x 32 kB/s  (int16 @ 16 kHz), printed at the end.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SR = 16000


def build(roots, out_dir, max_clips=24, min_clip_s=2.0, max_clip_s=8.0, seed=0,
          exts=(".flac", ".wav")):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speaker_dirs = sorted({d for r in roots for d in Path(r).iterdir() if d.is_dir()})
    chunks, names, clip_spk, clip_start, clip_len = [], [], [], [], []
    cursor = 0
    max_len = int(max_clip_s * SR)

    for spk_dir in speaker_dirs:
        files = sorted(p for p in spk_dir.rglob("*") if p.suffix.lower() in exts)
        rng.shuffle(files)                         # spread picks across chapters/sessions
        kept = []
        print("Processing files", len(files))
        for f in tqdm(files):
            if len(kept) >= max_clips:
                break
            info = sf.info(str(f))
            if info.samplerate != SR or info.duration < min_clip_s:
                continue
            x, _ = sf.read(str(f), dtype="int16", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1).astype(np.int16)
            if len(x) > max_len:                   # keep a random max_clip_s slice of long utterances
                s = int(rng.integers(0, len(x) - max_len + 1))
                x = x[s:s + max_len]
            kept.append(x)
        if not kept:
            continue
        spk_idx = len(names)
        names.append(spk_dir.name)
        for x in kept:
            chunks.append(x)
            clip_spk.append(spk_idx)
            clip_start.append(cursor)
            clip_len.append(len(x))
            cursor += len(x)

    audio = np.concatenate(chunks)
    np.save(out_dir / "pool.npy", audio)
    np.savez(out_dir / "index.npz", speakers=np.array(names), clip_spk=np.array(clip_spk, dtype=np.int64),
             clip_start=np.array(clip_start, dtype=np.int64), clip_len=np.array(clip_len, dtype=np.int64))
    print(f"speakers={len(names)}  clips={len(clip_spk)}  hours={len(audio) / SR / 3600:.2f}  "
          f"size={audio.nbytes / 1e9:.2f} GB  -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-clips-per-speaker", type=int, default=24)
    ap.add_argument("--min-clip-s", type=float, default=2.0)
    ap.add_argument("--max-clip-s", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.roots, a.out_dir, a.max_clips_per_speaker, a.min_clip_s, a.max_clip_s, a.seed)