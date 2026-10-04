"""
build_pools.py -- Builds memory-mapped speaker audio pools from VoxCeleb / LibriSpeech.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SR = 16000
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def get_file_hash(filepath):
    """Computes SHA-256 hash of a file for provenance tracking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_speaker_files(speaker_entry):
    """
    Given a speaker entry (directory path or file list), returns (speaker_id, file_paths).
    """
    path_obj = Path(speaker_entry)
    
    if path_obj.is_dir():
        spk_id = path_obj.name
        files = [
            p for p in path_obj.rglob("*")
            if p.suffix.lower() in AUDIO_EXTENSIONS
        ]
        return spk_id, files
    elif path_obj.is_file():
        spk_id = path_obj.parent.name
        return spk_id, [path_obj]
    else:
        # Fallback if path doesn't exist on disk yet
        spk_id = path_obj.name
        return spk_id, []


def process_split(
    split_name,
    speakers_list,
    out_dir,
    clips_per_spk=24,
    min_clip_s=3.5,
    max_clip_s=15.0,
    seed=42,
):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir) / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    min_len = int(round(min_clip_s * SR))
    max_len = int(round(max_clip_s * SR))

    # Parse speaker directories
    speakers_dict = {}
    for entry in speakers_list:
        spk_id, files = _get_speaker_files(entry)
        if files:
            speakers_dict[spk_id] = files

    spk_list = sorted(list(speakers_dict.keys()))
    print(
        f"[{split_name.upper()}] Processing {len(spk_list)} speakers "
        f"({clips_per_spk} clips/spk, min_clip_s={min_clip_s})..."
    )

    # --- PASS 1: Index files and determine sample ranges without reading audio ---
    clip_spk = []
    clip_start = []
    clip_len = []
    file_records = []  # List of tuples: (path, crop_start, crop_len)

    total_samples = 0

    for spk_idx, spk in enumerate(tqdm(spk_list, desc=f"Pass 1: Metadata ({split_name})")):
        files = list(speakers_dict[spk])
        rng.shuffle(files)

        selected_for_spk = 0
        for fpath in files:
            if selected_for_spk >= clips_per_spk:
                break
            try:
                info = sf.info(str(fpath))
                if info.samplerate != SR:
                    continue

                num_frames = info.frames
                if num_frames < min_len:
                    continue

                # Crop if longer than max_len
                if num_frames > max_len:
                    crop_len = max_len
                    max_start = num_frames - max_len
                    crop_start = int(rng.integers(0, max_start + 1))
                else:
                    crop_len = num_frames
                    crop_start = 0

                clip_spk.append(spk_idx)
                clip_start.append(total_samples)
                clip_len.append(crop_len)

                file_records.append((str(fpath), crop_start, crop_len))
                total_samples += crop_len
                selected_for_spk += 1

            except Exception:
                continue

    clip_spk = np.array(clip_spk, dtype=np.int32)
    clip_start = np.array(clip_start, dtype=np.int64)
    clip_len = np.array(clip_len, dtype=np.int32)

    # --- PASS 2: Memory-mapped writing ---
    pool_path = out_dir / "pool.npy"
    print(f"Pass 2: Writing {total_samples / SR / 3600:.2f} hours to memmap ({pool_path.name})...")
    mmap_arr = np.lib.format.open_memmap(
        pool_path, mode="w+", dtype="int16", shape=(total_samples,)
    )

    cursor = 0
    for fpath, c_start, c_len in tqdm(file_records, desc=f"Pass 2: Writing ({split_name})"):
        audio, _ = sf.read(
            fpath, start=c_start, frames=c_len, dtype="int16", always_2d=False
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.int16)

        mmap_arr[cursor : cursor + c_len] = audio
        cursor += c_len

    mmap_arr.flush()
    del mmap_arr

    # Save index
    index_path = out_dir / "index.npz"
    np.savez_compressed(
        index_path,
        speakers=np.array(spk_list),
        clip_spk=clip_spk,
        clip_start=clip_start,
        clip_len=clip_len,
    )

    # Save metadata for provenance tracking
    total_hours = float(total_samples / (SR * 3600.0))
    meta = {
        "split": split_name,
        "seed": seed,
        "speakers_count": len(spk_list),
        "clips_count": len(clip_spk),
        "total_samples": int(total_samples),
        "total_hours": round(total_hours, 3),
        "clips_per_spk": clips_per_spk,
        "min_clip_s": min_clip_s,
        "max_clip_s": max_clip_s,
        "sys_argv": sys.argv,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[{split_name.upper()}] Done. {len(spk_list)} speakers, "
        f"{len(clip_spk)} clips, {total_hours:.2f} hours."
    )


def build_pools(
    splits_json,
    out_dir,
    clips_per_spk=24,
    min_clip_s=3.5,
    max_clip_s=15.0,
    seed=42,
):
    splits_path = Path(splits_json)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    splits_hash = get_file_hash(splits_path)

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    global_meta = {
        "splits_json": str(splits_path.resolve()),
        "splits_json_hash": splits_hash,
        "seed": seed,
        "clips_per_spk": clips_per_spk,
        "min_clip_s": min_clip_s,
        "max_clip_s": max_clip_s,
    }
    with open(out_base / "meta.json", "w", encoding="utf-8") as f:
        json.dump(global_meta, f, indent=2)

    for split_name in ["train", "val", "test"]:
        if split_name in splits:
            process_split(
                split_name=split_name,
                speakers_list=splits[split_name],
                out_dir=out_base,
                clips_per_spk=clips_per_spk,
                min_clip_s=min_clip_s,
                max_clip_s=max_clip_s,
                seed=seed,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build audio pools from speaker splits.")
    parser.add_argument("--splits_json", default="./data/splits/splits.json")
    parser.add_argument("--out_dir", default="./data/pools")
    parser.add_argument("--clips_per_spk", type=int, default=24)
    parser.add_argument("--min_clip_s", type=float, default=3.5)
    parser.add_argument("--max_clip_s", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_pools(
        splits_json=args.splits_json,
        out_dir=args.out_dir,
        clips_per_spk=args.clips_per_spk,
        min_clip_s=args.min_clip_s,
        max_clip_s=args.max_clip_s,
        seed=args.seed,
    )