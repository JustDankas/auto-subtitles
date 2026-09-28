import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000


class SCDDataset(Dataset):
    def __init__(self, session_dir):
        self.audio_dir = Path(session_dir) / "audio"
        self.label_dir = Path(session_dir) / "labels"
        self.session_ids = sorted(p.stem for p in self.audio_dir.glob("*.wav"))

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        sid = self.session_ids[idx]
        wav, sr = torchaudio.load(str(self.audio_dir / f"{sid}.wav"))
        assert sr == SAMPLE_RATE

        labels = np.load(self.label_dir / f"{sid}.npy")
        labels = torch.from_numpy(labels).float()

        return wav.squeeze(0), labels


def collate_fn(batch):
    wavs, labels = zip(*batch)
    
    # Pad raw audio waveforms if needed
    wav_lengths = torch.tensor([w.shape[0] for w in wavs])
    max_wav_len = wav_lengths.max().item()
    
    label_lengths = torch.tensor([l.shape[0] for l in labels])
    max_label_len = label_lengths.max().item()

    wav_batch = torch.zeros(len(wavs), max_wav_len)
    label_batch = torch.zeros(len(labels), max_label_len)
    mask = torch.zeros(len(labels), max_label_len, dtype=torch.bool)

    for i, (w, l) in enumerate(zip(wavs, labels)):
        wav_batch[i, :w.shape[0]] = w
        label_batch[i, :l.shape[0]] = l
        mask[i, :l.shape[0]] = True

    return wav_batch, label_batch, mask, label_lengths