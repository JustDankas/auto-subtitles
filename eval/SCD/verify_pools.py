"""
verify_pools.py -- Verifies integrity, disjointness, and quality of created audio pools.
"""
import argparse
from pathlib import Path

import numpy as np

SR = 16000


def verify_pools(pools_dir="./data/pools", seed=42):
    pools_path = Path(pools_dir)
    splits = ["train", "val", "test"]
    
    speakers_by_split = {}
    rng = np.random.default_rng(seed)

    print("=" * 80)
    print("POOL VERIFICATION & DIAGNOSTICS")
    print("=" * 80)

    # 1. Load indices and verify speaker disjointness
    for split in splits:
        split_dir = pools_path / split
        index_file = split_dir / "index.npz"
        
        if not index_file.exists():
            print(f"[-] Split '{split}' index not found at {index_file}. Skipping...")
            continue

        idx = np.load(index_file)
        spks = set(str(s) for s in idx["speakers"])
        speakers_by_split[split] = spks

    loaded_splits = list(speakers_by_split.keys())
    for i in range(len(loaded_splits)):
        for j in range(i + 1, len(loaded_splits)):
            s1, s2 = loaded_splits[i], loaded_splits[j]
            overlap = speakers_by_split[s1].intersection(speakers_by_split[s2])
            assert len(overlap) == 0, f"CRITICAL: Speaker overlap detected between '{s1}' and '{s2}': {overlap}"

    print("[+] Speaker Disjointness Assertion Passed: No overlapping speakers across splits.\n")

    # 2. Detailed Split Inspection
    for split in splits:
        split_dir = pools_path / split
        index_file = split_dir / "index.npz"
        pool_file = split_dir / "pool.npy"

        if not index_file.exists() or not pool_file.exists():
            continue

        print(f"--- Split: {split.upper()} ---")
        idx = np.load(index_file)
        spks = idx["speakers"]
        clip_spk = idx["clip_spk"]
        clip_start = idx["clip_start"]
        clip_len = idx["clip_len"]

        # Memmap pool array
        pool_arr = np.load(pool_file, mmap_mode="r")
        total_samples = len(pool_arr)
        total_hours = total_samples / (SR * 3600.0)

        clip_durations_s = clip_len / float(SR)
        short_clips_count = np.sum(clip_durations_s < 3.0)

        print(f"Speakers Count   : {len(spks)}")
        print(f"Total Clips      : {len(clip_spk)}")
        print(f"Total Duration   : {total_hours:.2f} hours ({total_samples} samples)")
        print(f"Clips < 3.0s     : {short_clips_count} ({short_clips_count / len(clip_spk) * 100:.1f}%)")

        # Clip-length Histogram
        counts, bin_edges = np.histogram(clip_durations_s, bins=[0.0, 2.0, 3.0, 3.5, 5.0, 10.0, 15.0, 30.0])
        print("Clip Length Histogram:")
        for k in range(len(counts)):
            print(f"  [{bin_edges[k]:4.1f}s - {bin_edges[k+1]:4.1f}s): {counts[k]} clips")

        # Bounds Assertion
        max_bounds = clip_start + clip_len
        assert np.all(max_bounds <= total_samples), f"CRITICAL: Array bounds exceeded in split '{split}'!"
        print("[+] Array Bounds Check Passed: All (clip_start + clip_len) <= len(pool).")

        # Spot-check 20 random clips
        num_checks = min(20, len(clip_spk))
        sample_indices = rng.choice(len(clip_spk), size=num_checks, replace=False)
        
        valid_spot_checks = 0
        for s_idx in sample_indices:
            c_st = clip_start[s_idx]
            c_ln = clip_len[s_idx]
            clip_data = pool_arr[c_st : c_st + c_ln]

            # Assert non-empty and non-zero
            assert len(clip_data) == c_ln, "Clip length mismatch on disk!"
            assert np.abs(clip_data).max() > 0, "Encountered all-zero audio clip!"
            assert clip_data.dtype == np.int16, "Audio dtype is not int16!"
            valid_spot_checks += 1

        print(f"[+] Spot-check Passed: {valid_spot_checks} random clips verified (int16 valid, non-zero).\n")

    print("=" * 80)
    print("VERIFICATION COMPLETE: ALL POOLS VALIDATED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify audio pool integrity and disjointness.")
    parser.add_argument("--pools_dir", default="C:/src/data/pools")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    verify_pools(pools_dir=args.pools_dir, seed=args.seed)