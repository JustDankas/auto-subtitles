from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000


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

    def __init__(self, session_dir, n_mels: int = 64):
        self.audio_dir = Path(session_dir) / "audio"
        self.label_dir = Path(session_dir) / "labels"
        self.session_ids = sorted(p.stem for p in self.audio_dir.glob("*.wav"))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=n_mels
        )

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        sid = self.session_ids[idx]
        wav, sr = torchaudio.load(str(self.audio_dir / f"{sid}.wav"))
        assert sr == SAMPLE_RATE
        feats = self.mel(wav).squeeze(0)                     # (n_mels, T)
        feats = torch.log(feats + 1e-6).transpose(0, 1)       # (T, n_mels)

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
