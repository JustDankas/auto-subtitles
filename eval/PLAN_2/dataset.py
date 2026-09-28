import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000

def spec_augment(
    feats: torch.Tensor, 
    time_mask_frames: int = 10,     # Reduced from 30 -> 10 frames (~100ms max)
    freq_mask_bins: int = 8,        # Mask up to 8 mel bins
    n_time_masks: int = 1,          # Reduced from 2 -> 1 mask
    n_freq_masks: int = 1,          # Reduced from 2 -> 1 mask
    p: float = 0.5                  # Only apply to 50% of training samples
) -> torch.Tensor:
    if random.random() > p:
        return feats

    feats = feats.clone()
    T, F = feats.shape
    
    # 1. Frequency Masking
    for _ in range(n_freq_masks):
        f = random.randint(0, max(F - freq_mask_bins, 0))
        feats[:, f:f + freq_mask_bins] = 0.0

    # 2. Gentle Time Masking
    for _ in range(n_time_masks):
        t = random.randint(0, max(T - time_mask_frames, 0))
        feats[t:t + time_mask_frames, :] = 0.0

    return feats



class SCDDataset(Dataset):
    """
    Every file under `session_dir` is now already a fixed-duration window
    (see synthetic.py's `generate_window_dataset`), each containing at most
    one speaker change -- so unlike the old session-based dataset, there is
    no cropping/jittering to do here. Train and val/test use this exact same
    loading path, which is the point: the model now sees the same
    distribution (one bounded window, fresh state) at train and eval time,
    instead of training on short crops and evaluating on full multi-minute
    sessions.
    """

    def __init__(self, session_dir, n_mels: int = 64, augment: bool = False):
        self.audio_dir = Path(session_dir) / "audio"
        self.label_dir = Path(session_dir) / "labels"
        self.session_ids = sorted(p.stem for p in self.audio_dir.glob("*.wav"))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.augment = augment
        

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        sid = self.session_ids[idx]
        wav, sr = torchaudio.load(str(self.audio_dir / f"{sid}.wav"))
        assert sr == SAMPLE_RATE
        feats = self.mel(wav).squeeze(0)                     # (n_mels, T)
        feats = torch.log(feats + 1e-6).transpose(0, 1)       # (T, n_mels)

        # 1. Apply SpecAugment BEFORE normalization on raw log-mel features
        if self.augment:
            feats = spec_augment(feats, time_mask_frames=10, p=0.35)

        # 2. Normalize AFTER SpecAugment so std/mean calculations stay valid
        mean = feats.mean(dim=0, keepdim=True)
        std = feats.std(dim=0, keepdim=True) + 1e-5
        feats = (feats - mean) / std

        labels = np.load(self.label_dir / f"{sid}.npy")
        labels = torch.from_numpy(labels).float()

        # All windows are nominally the same length, but guard against a
        # +-1 frame rounding mismatch between the mel framing and the label
        # generator's np.ceil(duration_s * 1000 / frame_hop_ms).
        T = min(feats.shape[0], labels.shape[0])
        return feats[:T], labels[:T]


def collate_fn(batch):
    """
    Windows should already be equal-length, so this is mostly a no-op safety
    net for the occasional off-by-one frame -- kept so a stray mismatched
    file doesn't crash a whole batch.
    """
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
