"""
build_aug_banks.py -- One-time conversion of MUSAN + RIRs into mmap-able banks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SR = 16000

# -----------------------------------------------------------------------------
# 1. RIR Selection & Parsing
# -----------------------------------------------------------------------------
from pathlib import Path


def parse_rir_list(rir_root):
    rir_root = Path(rir_root)
    rir_paths = []
    
    # Check for Kaldi rir_list text files first
    list_files = sorted(list(rir_root.rglob("rir_list")))
    
    if list_files:
        print(f"Found {len(list_files)} Kaldi rir_list file(s).")
        for lf in list_files:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    if tokens:
                        # Extract wav path string and convert to Path object
                        rel_path = Path(tokens[-1])
                        
                        # Strip leading folder if it matches rir_root's name (e.g. 'RIRS_NOISES')
                        if rel_path.parts and rel_path.parts[0] == rir_root.name:
                            rel_path = Path(*rel_path.parts[1:])
                            
                        wav_path = rir_root / rel_path
                        
                        if wav_path.exists():
                            rir_paths.append(wav_path)
        rule_used = "Kaldi rir_list files"
    else:
        # Fallback: strict filtering rule
        print("No rir_list files found. Using fallback filtering rule...")
        for p in sorted(list(rir_root.rglob("*.wav"))):
            rel_parts = p.relative_to(rir_root).parts
            if "pointsource_noises" in rel_parts:
                continue
            if "noise" in p.name.lower():
                continue
            if "simulated_rirs" in rel_parts or "real_rirs_isotropic_noises" in rel_parts:
                rir_paths.append(p)
        rule_used = "Fallback filter rules"

    rir_paths = sorted(list(set(rir_paths)))
    print(f"Rule [{rule_used}] kept {len(rir_paths)} files.")
    return rir_paths

# -----------------------------------------------------------------------------
# 2. Stratification, Hold-out, and Subsampling
# -----------------------------------------------------------------------------
def get_rir_category(p, rir_root):
    rel_parts = p.relative_to(rir_root).parts
    if "simulated_rirs" in rel_parts:
        if "smallroom" in rel_parts:
            return "smallroom"
        elif "mediumroom" in rel_parts:
            return "mediumroom"
        elif "largeroom" in rel_parts:
            return "largeroom"
    elif "real_rirs_isotropic_noises" in rel_parts:
        return "real"
    return "other"

def stratify_and_subsample_rirs(rir_paths, rir_root, seed=42, target_train=4000, target_val=500):
    rng = np.random.default_rng(seed)
    
    # Categorize RIRs
    cats = {"smallroom": [], "mediumroom": [], "largeroom": [], "real": [], "other": []}
    for p in rir_paths:
        c = get_rir_category(p, rir_root)
        cats[c].append(p)

    train_groups, val_groups = [], []
    
    # Process each category to hold out ~10% by room/group
    for cat_name, paths in cats.items():
        if not paths:
            continue
            
        groups = {}
        for p in paths:
            # Simulated grouped by parent directory (room); real grouped by unique file path
            group_key = p.parent.name if cat_name != "real" else str(p)
            groups.setdefault(group_key, []).append(p)
            
        group_keys = sorted(list(groups.keys()))
        rng.shuffle(group_keys)
        
        val_cutoff = max(1, int(0.1 * len(group_keys)))
        val_k = group_keys[:val_cutoff]
        train_k = group_keys[val_cutoff:]
        
        cat_train = [p for k in train_k for p in groups[k]]
        cat_val = [p for k in val_k for p in groups[k]]
        
        # Subsample proportionally across categories
        per_cat_train = target_train // 4
        per_cat_val = target_val // 4
        
        if len(cat_train) > per_cat_train:
            rng.shuffle(cat_train)
            cat_train = cat_train[:per_cat_train]
            
        if len(cat_val) > per_cat_val:
            rng.shuffle(cat_val)
            cat_val = cat_val[:per_cat_val]
            
        train_groups.extend(cat_train)
        val_groups.extend(cat_val)
        
    return sorted(train_groups), sorted(val_groups)

# -----------------------------------------------------------------------------
# 3. Per-RIR Processing
# -----------------------------------------------------------------------------
def process_rir(path, max_samples=6400):
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None

    # Assert sampling rate
    if sr != SR:
        return None

    rir = data[:, 0]  # Mono channel 0
    
    # Peak-align so speech is not delayed by direct-path onset
    peak_idx = np.argmax(np.abs(rir))
    rir = rir[peak_idx:]
    
    if len(rir) > max_samples:
        rir = rir[:max_samples]
    
    # 10 ms cosine fade on tail (160 samples @ 16 kHz)
    fade_len = min(160, len(rir))
    if fade_len > 0:
        fade = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        rir[-fade_len:] *= fade
        
    # Energy normalization
    energy = np.sqrt(np.sum(rir ** 2))
    if energy < 1e-6:
        return None
    rir = rir / energy
    
    # Zero-pad to exact max_samples
    padded = np.zeros(max_samples, dtype=np.float32)
    padded[:len(rir)] = rir
    return padded

# -----------------------------------------------------------------------------
# 4 & 5. Build Augmentation Banks (Deterministic Two-Pass Writing)
# -----------------------------------------------------------------------------
def build_banks(musan_dir, rir_dir, out_dir, seed=42, music_sec_per_file=60):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rir_root = Path(rir_dir)

    # --- 1. RIR Processing ---
    all_rirs = parse_rir_list(rir_root)
    train_rir_paths, val_rir_paths = stratify_and_subsample_rirs(all_rirs, rir_root, seed=seed)
    
    # Verify disjoint sets
    assert set(train_rir_paths).isdisjoint(set(val_rir_paths)), "Train and Val RIR sets must be disjoint!"

    def compile_and_write_rirs(path_list, out_path):
        processed = []
        for p in tqdm(path_list, desc=f"RIR Processing ({out_path.name})"):
            res = process_rir(p)
            if res is not None:
                processed.append(res)
        arr = np.array(processed, dtype=np.float32) if processed else np.zeros((0, 6400), dtype=np.float32)
        np.save(out_path, arr)
        return arr

    rirs_train = compile_and_write_rirs(train_rir_paths, out_dir / "rirs_train.npy")
    rirs_val = compile_and_write_rirs(val_rir_paths, out_dir / "rirs_val.npy")
    print(f"RIR Banks Saved: Train={rirs_train.shape}, Val={rirs_val.shape}")

    # Acceptance Verification on RIRs
    for name, arr in [("Train", rirs_train), ("Val", rirs_val)]:
        if len(arr) > 0:
            assert np.all(np.isfinite(arr)), f"{name} RIR array contains non-finite values!"
            # Peak index of aligned RIRs must be 0 (< 50)
            peak_indices = np.argmax(np.abs(arr), axis=1)
            assert np.all(peak_indices < 50), f"{name} RIR peak indices exceed threshold 50!"

    # --- 2. MUSAN Processing (Music & Noise only; Low-RAM 2-Pass Write) ---
    musan_root = Path(musan_dir)
    for cat in ["noise", "music"]:
        print(f"Processing MUSAN {cat}...")
        files = sorted(list((musan_root / cat).rglob("*.wav")))
        rng.shuffle(files)
        
        val_count = max(1, int(0.1 * len(files)))
        val_files = sorted(files[:val_count])
        train_files = sorted(files[val_count:])
        
        assert set(train_files).isdisjoint(set(val_files)), f"Train and Val {cat} sets must be disjoint!"
        
        for split, split_files in [("train", train_files), ("val", val_files)]:
            # PASS 1: Collect metadata and compute total length
            starts, lens = [], []
            valid_files = []
            total_samples = 0
            
            for f in tqdm(split_files, desc=f"Pass 1: Metadata [{cat}_{split}]"):
                try:
                    info = sf.info(str(f))
                    if info.samplerate != SR:
                        continue
                    audio, _ = sf.read(str(f), dtype="int16", always_2d=False)
                except Exception:
                    continue

                if audio.ndim > 1:
                    audio = audio.mean(axis=1).astype(np.int16)
                    
                # Drop files with RMS below -70 dBFS (~10 counts in int16)
                rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
                if rms < 10:
                    continue
                    
                target_len = len(audio)
                if cat == "music" and target_len > music_sec_per_file * SR:
                    target_len = music_sec_per_file * SR

                starts.append(total_samples)
                lens.append(target_len)
                valid_files.append((f, target_len))
                total_samples += target_len

            # PASS 2: Memory-mapped writing to keep RAM usage low
            mmap_path = out_dir / f"{cat}_{split}.npy"
            mmap_arr = np.lib.format.open_memmap(mmap_path, mode="w+", dtype="int16", shape=(total_samples,))
            
            cursor = 0
            for f, t_len in tqdm(valid_files, desc=f"Pass 2: Writing [{cat}_{split}]"):
                audio, _ = sf.read(str(f), dtype="int16", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1).astype(np.int16)
                
                if cat == "music" and len(audio) > t_len:
                    max_start = len(audio) - t_len
                    s_idx = rng.integers(0, max_start)
                    audio = audio[s_idx : s_idx + t_len]
                    
                mmap_arr[cursor : cursor + t_len] = audio
                cursor += t_len
                
            mmap_arr.flush()
            del mmap_arr
            
            np.savez(out_dir / f"{cat}_{split}_idx.npz", start=np.array(starts, dtype=np.int64), len=np.array(lens, dtype=np.int64))

    meta = {
        "seed": seed,
        "musan_dir": str(musan_dir),
        "rir_dir": str(rir_dir),
        "rirs_train_count": len(rirs_train),
        "rirs_val_count": len(rirs_val)
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Augmentation Bank Creation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musan_dir", default="./data/aug/musan")
    parser.add_argument("--rir_dir", default="./data/aug/RIRS_NOISES")
    parser.add_argument("--out_dir", default="./data/aug_banks")
    args = parser.parse_args()
    build_banks(args.musan_dir, args.rir_dir, args.out_dir)