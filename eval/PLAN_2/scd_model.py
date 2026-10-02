"""
model.py -- non-causal, residual, downsampled TCN for fixed-window speaker-change detection.

Replaces the causal SCDModel. The window is handed to the model whole, once per
call, so "non-causal" here means "two-sided within the window" -- every frame
can use audio from both before and after it *inside* the 3 s clip. That is the
fix for the recall ceiling diagnosed in scd_plateau_plan.md: a strictly causal
net cannot fire on a centered label before it has heard the new speaker.

Output has 2 channels: channel 0 is the change-bump target (what you train/
threshold on), channel 1 is an auxiliary "which side of the change am I on"
step target (0 before t_change, 1 after) that supervises every frame instead
of only the ~17 near the boundary -- a data-efficiency lever, not a change to
what you evaluate.
"""
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

    def __init__(self, n_mels: int = 64, augment: bool = False):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.augment = GPUSpecAugment() if augment else None

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(wavs)               # (B, n_mels, T)
        feats = torch.log(mel + 1e-6)

        mean = feats.mean(dim=2, keepdim=True)        # normalize across time, per mel bin
        std = feats.std(dim=2, keepdim=True) + 1e-5
        feats = (feats - mean) / std

        if self.augment is not None:
            feats = self.augment(feats)               # masking after normalization, see above
        return feats                                   # (B, n_mels, T)


class ResTCNBlock(nn.Module):
    """Residual dilated conv block using Depthwise Separable Convolutions.
    causal=False -> symmetric padding, so every frame sees context on BOTH
    sides within the window (see module docstring).
    """

    def __init__(self, ch: int, kernel_size: int = 5, dilation: int = 1,
                 dropout: float = 0.15, causal: bool = False):
        super().__init__()
        total = (kernel_size - 1) * dilation
        self.pad = (total, 0) if causal else (total // 2, total - total // 2)
        
        # Depthwise Separable Convolution 1: Depthwise (K x 1) + Pointwise (1 x 1)
        self.conv1_dw = nn.Conv1d(ch, ch, kernel_size, dilation=dilation, groups=ch, bias=False)
        self.conv1_pw = nn.Conv1d(ch, ch, 1)
        self.bn1 = nn.BatchNorm1d(ch)
        
        # Depthwise Separable Convolution 2: Depthwise (1 x 1 with kernel 1 is trivial pointwise, kept 1x1 for residual projection)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        # Apply depthwise spatial filtering followed by pointwise channel mixing
        dw_out = self.conv1_dw(F.pad(x, self.pad))
        y = self.drop(F.gelu(self.bn1(self.conv1_pw(dw_out))))
        return F.gelu(x + self.bn2(self.conv2(y)))


def describe_receptive_field(kernel_size: int, dilations, stem_kernel_strides=((5, 2), (5, 1))) -> dict:
    """
    Receptive field of stem + ResTCNBlock stack, in original 10ms mel frames.
    Computed layer-by-layer (RF += (k-1)*dilation*cumulative_stride at each
    layer, matching the standard formula) -- NOT by treating the stem as pure
    downsampling and adding the blocks' receptive field on top of it. Printed at
    model init so a change to kernel_size / dilations / stem config can't
    silently shrink context below what the window needs -- the non-causal
    analogue of the old required_cnn_layers() check.
    """
    rf, jump = 1, 1
    for k, s in stem_kernel_strides:
        rf += (k - 1) * jump
        jump *= s
    for d in dilations:
        rf += (kernel_size - 1) * d * jump
    return dict(downsample_factor=jump, rf_input_frames=rf, rf_input_seconds=rf * 0.01)


class SCDNet(nn.Module):
    """
    wav (B, num_samples) -> (B, n_outputs, T)
      out[:, 0] = change-bump logits (sigmoid -> P(change), what you threshold/report)
      out[:, 1] = step logits        (sigmoid -> P(after change), auxiliary training target)

    For a 3.0s/10ms-hop window, mel framing gives
    T_in=301; the stride-2/stride-1 stem gives 301->151 with symmetric
    padding preserving length at each step; default kernel_size=5,
    dilations=(1,2,4,8,16) give a receptive field of 261 downsampled-domain
    frames = 2.61s in the original 10ms grid, two-sided -- comfortably covers
    a 3s window from the middle, less so right at the edges (expected: a
    change 50ms before the window's right edge has less post-change evidence
    than one in the middle, by construction, not a bug).
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
    ):
        super().__init__()
        self.frontend = AudioFeatureExtractorGPU(n_mels=n_mels, augment=augment)
        # FIX 1: Causal stem padding
        # Non-causal uses symmetric padding=2. Causal needs left-only padding of (4, 0).
        if causal:
            self.stem = nn.Sequential(
                nn.ZeroPad1d((4, 0)),
                nn.Conv1d(n_mels, ch, 5, stride=2, padding=0),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.ZeroPad1d((4, 0)),
                nn.Conv1d(ch, ch, 5, stride=1, padding=0), # Downsample by 2x not 4x
                nn.BatchNorm1d(ch),
                nn.GELU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(n_mels, ch, 5, stride=2, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
                nn.Conv1d(ch, ch, 5, stride=1, padding=2), nn.BatchNorm1d(ch), nn.GELU(), # Downsample by 2x not 4x
            )
        self.blocks = nn.Sequential(
            *[ResTCNBlock(ch, kernel_size, d, dropout, causal) for d in dilations]
        )
        self.head = nn.Conv1d(ch, n_outputs, 1)

        self.causal = causal
        self.num_blocks = len(dilations)
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.receptive_field = describe_receptive_field(kernel_size, dilations, stem_kernel_strides=((5, 2), (5, 1)))  # second stem conv is stride=1, not 2
        print(
            f"[SCDNet] {self.num_blocks} residual blocks, kernel_size={kernel_size}, "
            f"causal={causal} -> receptive field {self.receptive_field['rf_input_seconds']:.2f}s "
            f"({'one' if causal else 'two'}-sided)"
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for b in self.blocks:
            nn.init.zeros_(b.bn2.weight)  # each residual block starts as identity

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.frontend(wav)                 # (B, n_mels, T_in)
        t_in = x.shape[-1]
        out = self.head(self.blocks(self.stem(x)))  # (B, n_outputs, T_in // 2)

        # FIX 2: Causal upsampling
        # mode="nearest" repeats past values into future frame bins without looking ahead
        mode = "nearest" if self.causal else "linear"
        align_corners = None if self.causal else False
        return F.interpolate(out, size=t_in, mode=mode, align_corners=align_corners)