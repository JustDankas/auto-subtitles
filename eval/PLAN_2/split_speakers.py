import random, shutil
from pathlib import Path

def split_speakers(speakers_dir, out_root, ratios=(0.7, 0.15, 0.15), seed=42):
    speakers = sorted(p.name for p in Path(speakers_dir).iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n = len(speakers)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    splits = {
        "train": speakers[:n_train],
        "val": speakers[n_train:n_train + n_val],
        "test": speakers[n_train + n_val:],
    }
    for split, spk_list in splits.items():
        split_dir = Path(out_root) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for spk in spk_list:
            shutil.copytree(Path(speakers_dir) / spk, split_dir / spk, dirs_exist_ok=True)
        print(f"{split}: {len(spk_list)} speakers")
    return splits
