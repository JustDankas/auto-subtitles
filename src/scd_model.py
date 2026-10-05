"""
model.py -- non-causal, residual, downsampled TCN for fixed-window speaker-change detection.

Output has 2 channels: channel 0 is the change-bump target (what you train/
threshold on), channel 1 is an auxiliary "which side of the change am I on"
step target (0 before t_change, 1 after) that supervises every frame instead
of only the ~17 near the boundary -- a data-efficiency lever, not a change to
what you evaluate.

Note on causality (causal=True):
Setting causal=True applies left-only padding in the stem/blocks and nearest-neighbor
upsampling. However, because AudioFeatureExtractorGPU normalizes mean and std across
the entire time dimension of the clip, information still leaks across the full window.
True streaming causality would require streaming or causal running normalization.
"""
import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

SAMPLE_RATE = 16000


class GPUSpecAugment(nn.Module):
    """
    Two fixes vs. the previous version:
      1. `iid_masks=True` -- otherwise torchaudio applies ONE mask to the whole
         batch, not one per example.
      2. Applied AFTER instance normalization (in AudioFeatureExtractorGPU),
         not on raw log-mels -- masking to 0 is only a neutral "erase this"
         value once 0 is the post-normalization mean.
    """

    def __init__(self, time_mask_param: int = 10, freq_mask_param: int = 8, n_masks: int = 2):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)
        self.n_masks = n_masks

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, n_mels, T), already normalized
        if self.training:
            for _ in range(self.n_masks):
                feats = self.time_mask(self.freq_mask(feats))
        return feats


class AudioFeatureExtractorGPU(nn.Module):
    """wav (B, num_samples) -> normalized log-mel (B, n_mels, T), augmented on GPU."""

    def __init__(
        self,
        n_mels: int = 64,
        augment: bool = False,
        time_mask_param: int = 10,
        freq_mask_param: int = 8,
        n_masks: int = 2,
    ):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.augment = (
            GPUSpecAugment(
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
                n_masks=n_masks,
            )
            if augment
            else None
        )

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        # Protect STFT / mel transforms against mixed-precision precision loss or CUDA kernel issues
        with torch.autocast("cuda", enabled=False):
            wavs = wavs.float()
            mel = self.mel_transform(wavs)               # (B, n_mels, T)
            feats = torch.log(mel + 1e-6)

            mean = feats.mean(dim=2, keepdim=True)        # normalize across time, per mel bin
            std = feats.std(dim=2, keepdim=True) + 1e-5
            feats = (feats - mean) / std

            if self.augment is not None:
                feats = self.augment(feats)               # masking after normalization

        return feats                                   # (B, n_mels, T)


class ResTCNBlock(nn.Module):
    """Residual dilated conv block using Depthwise Separable or Dense Convolutions.
    causal=False -> symmetric padding, so every frame sees context on BOTH
    sides within the window (see module docstring).
    
    Structure: Post-activation GELU(x + BN2(Conv2(y))) where y is GELU(BN1(Conv1(x))).
    """

    def __init__(
        self,
        ch: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.15,
        causal: bool = False,
        dense: bool = False,
    ):
        super().__init__()
        total = (kernel_size - 1) * dilation
        self.pad = (total, 0) if causal else (total // 2, total - total // 2)
        self.dense = dense

        if dense:
            # Standard Dense Convolution (often faster on tensor cores & provides higher model capacity)
            self.conv1 = nn.Conv1d(ch, ch, kernel_size, dilation=dilation, bias=False)
            self.bn1 = nn.BatchNorm1d(ch)
            self.conv2 = nn.Conv1d(ch, ch, 1)
            self.bn2 = nn.BatchNorm1d(ch)
        else:
            # Depthwise Separable Convolution
            self.conv1_dw = nn.Conv1d(ch, ch, kernel_size, dilation=dilation, groups=ch, bias=False)
            self.conv1_pw = nn.Conv1d(ch, ch, 1)
            self.bn1 = nn.BatchNorm1d(ch)
            self.conv2 = nn.Conv1d(ch, ch, 1)
            self.bn2 = nn.BatchNorm1d(ch)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        padded = F.pad(x, self.pad)
        if self.dense:
            y = self.drop(F.gelu(self.bn1(self.conv1(padded))))
        else:
            dw_out = self.conv1_dw(padded)
            y = self.drop(F.gelu(self.bn1(self.conv1_pw(dw_out))))
        return F.gelu(x + self.bn2(self.conv2(y)))


def describe_receptive_field(kernel_size: int, dilations, stem_kernel_strides=((5, 2), (5, 1))) -> dict:
    """
    Receptive field of stem + ResTCNBlock stack, in original 10ms mel frames.
    Computed layer-by-layer (RF += (k-1)*dilation*cumulative_stride at each
    layer, matching the standard formula). Printed at model init so a change to
    kernel_size / dilations / stem config can't silently shrink context below what
    the window needs.
    """
    rf, jump = 1, 1
    for k, s in stem_kernel_strides:
        rf += (k - 1) * jump
        jump *= s
    for d in dilations:
        rf += (kernel_size - 1) * d * jump

    rf_seconds = rf * 0.01

    assert rf_seconds >= 1.0, (
        f"Receptive field ({rf_seconds:.2f}s / {rf} frames) is under 1.0s. "
        f"Insufficient context for change detection."
    )

    if rf_seconds < 2.0:
        warnings.warn(
            f"[SCDNet] Receptive field is {rf_seconds:.2f}s ({rf} frames). "
            f"Note: dilations=(1,2,4,8) gives ~1.33s RF. Consider dilations=(1,2,4,8,16) "
            f"for ~2.61s context across a 3.0s window."
        )

    return dict(downsample_factor=jump, rf_input_frames=rf, rf_input_seconds=rf_seconds)


class SCDNet(nn.Module):
    """
    wav (B, num_samples) -> (B, n_outputs, T)
      out[:, 0] = change-bump logits (sigmoid -> P(change), what you threshold/report)
      out[:, 1] = step logits        (sigmoid -> P(after change), auxiliary training target)

    For a 3.0s/10ms-hop window, mel framing gives T_in=301; the stride-2/stride-1 stem
    gives 301->151 downsampling with symmetric padding preserving length at each step;
    default kernel_size=5, dilations=(1,2,4,8,16) give a receptive field of 261 downsampled-domain
    frames = 2.61s in the original 10ms grid, two-sided -- comfortably covers a 3s window
    from the middle.
    """

    def __init__(
        self,
        n_mels: int = 64,
        ch: int = 64,
        kernel_size: int = 5,
        dilations=(1, 2, 4, 8, 16),
        dropout: float = 0.15,
        causal: bool = False,
        augment: bool = False,
        n_outputs: int = 2,
        time_mask_param: int = 10,
        freq_mask_param: int = 8,
        n_masks: int = 2,
        dense: bool = False,
    ):
        super().__init__()
        self.frontend = AudioFeatureExtractorGPU(
            n_mels=n_mels,
            augment=augment,
            time_mask_param=time_mask_param,
            freq_mask_param=freq_mask_param,
            n_masks=n_masks,
        )
        if causal:
            self.stem = nn.Sequential(
                nn.ZeroPad1d((4, 0)),
                nn.Conv1d(n_mels, ch, 5, stride=2, padding=0),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.ZeroPad1d((4, 0)),
                nn.Conv1d(ch, ch, 5, stride=1, padding=0),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(n_mels, ch, 5, stride=2, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
                nn.Conv1d(ch, ch, 5, stride=1, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            )
        self.blocks = nn.Sequential(
            *[ResTCNBlock(ch, kernel_size, d, dropout, causal, dense) for d in dilations]
        )
        self.head = nn.Conv1d(ch, n_outputs, 1)

        self.causal = causal
        self.num_blocks = len(dilations)
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.receptive_field = describe_receptive_field(
            kernel_size, dilations, stem_kernel_strides=((5, 2), (5, 1))
        )
        print(
            f"[SCDNet] {self.num_blocks} residual blocks, kernel_size={kernel_size}, "
            f"dense={dense}, causal={causal} -> receptive field {self.receptive_field['rf_input_seconds']:.2f}s "
            f"({self.receptive_field['rf_input_frames']} frames, {'one' if causal else 'two'}-sided)"
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for b in self.blocks:
            nn.init.zeros_(b.bn2.weight)  # Output scale projection initialized to zero

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.frontend(wav)                 # (B, n_mels, T_in)
        t_in = x.shape[-1]
        out = self.head(self.blocks(self.stem(x)))  # (B, n_outputs, T_in // 2)

        mode = "nearest" if self.causal else "linear"
        align_corners = None if self.causal else False
        return F.interpolate(out, size=t_in, mode=mode, align_corners=align_corners)


def add_scd_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Helper to register SCDNet parameters to an argparse parser."""
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--time_mask_param", type=int, default=10, help="Max time mask length in frames")
    parser.add_argument("--freq_mask_param", type=int, default=8, help="Max frequency mask length in mel bins")
    parser.add_argument("--n_masks", type=int, default=2, help="Number of masks per domain")
    parser.add_argument("--dense", action="store_true", help="Use standard 1D convs instead of depthwise separable convs")
    return parser