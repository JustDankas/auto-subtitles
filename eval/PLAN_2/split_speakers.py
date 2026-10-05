import argparse
import json
import random
from pathlib import Path


def split_speakers(roots, out_json="splits.json", ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Finds all unique speaker directories across multiple root folders,
    splits them deterministically, and saves a JSON file mapping splits 
    to absolute speaker directory paths.
    """
    # Map speaker_id -> list of absolute paths (handles same speaker across roots)
    speaker_map = {}
    for r in roots:
        root_path = Path(r).resolve()
        if not root_path.exists():
            continue
        for spk_dir in root_path.iterdir():
            if spk_dir.is_dir():
                speaker_map.setdefault(spk_dir.name, []).append(str(spk_dir))

    speaker_ids = sorted(speaker_map.keys())
    rng = random.Random(seed)
    rng.shuffle(speaker_ids)

    n = len(speaker_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    split_ids = {
        "train": speaker_ids[:n_train],
        "val": speaker_ids[n_train : n_train + n_val],
        "test": speaker_ids[n_train + n_val :],
    }

    # Build the final output JSON mapping split -> list of absolute speaker directory paths
    splits_json_data = {}
    for split, spks in split_ids.items():
        paths = []
        for spk in spks:
            paths.extend(speaker_map[spk])
        splits_json_data[split] = paths
        print(f"{split}: {len(spks)} unique speakers ({len(paths)} speaker directories)")

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(splits_json_data, f, indent=2)

    print(f"Saved splits to {out_json.resolve()}")
    return splits_json_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split speakers across multiple dataset roots into JSON.")
    parser.add_argument("--roots", nargs="+", required=True, help="Input directory paths (e.g. train-clean-100 train-clean-360)")
    parser.add_argument("--out-json", default="splits.json", help="Path to save the generated JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    args = parser.parse_args()

    split_speakers(roots=args.roots, out_json=args.out_json, seed=args.seed)