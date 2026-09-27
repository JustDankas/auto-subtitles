from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000

class SCDDataset(Dataset):
    def __init__(self, session_dir, n_mels=64):
        self.audio_dir = Path(session_dir) / "audio"
        self.label_dir = Path(session_dir) / "labels"
        self.session_ids = sorted(p.stem for p in self.audio_dir.glob("*.wav"))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=n_mels,
        )

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        sid = self.session_ids[idx]
        wav, sr = torchaudio.load(str(self.audio_dir / f"{sid}.wav"))
        assert sr == SAMPLE_RATE
        feats = self.mel(wav).squeeze(0)                     # (n_mels, T)
        feats = torch.log(feats + 1e-6).transpose(0, 1)       # (T, n_mels)
        ## Fix 1 - normalize feats
        # Standardize features (mean=0, std=1 across time)
        mean = feats.mean(dim=0, keepdim=True)
        std = feats.std(dim=0, keepdim=True) + 1e-5
        feats = (feats - mean) / std
        ## Fix 1
        labels = np.load(self.label_dir / f"{sid}.npy")
        labels = torch.from_numpy(labels).float()
        T = min(feats.shape[0], labels.shape[0])              # guard off-by-one rounding
        return feats[:T], labels[:T]


def collate_fn(batch):
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats])
    T_max = lengths.max().item()
    n_mels = feats[0].shape[1]
    feat_batch = torch.zeros(len(feats), T_max, n_mels)
    label_batch = torch.zeros(len(feats), T_max)
    mask = torch.zeros(len(feats), T_max, dtype=torch.bool)
    for i, (f, l) in enumerate(zip(feats, labels)):
        T = f.shape[0]
        feat_batch[i, :T] = f
        label_batch[i, :T] = l
        mask[i, :T] = True
    return feat_batch, label_batch, mask, lengths
