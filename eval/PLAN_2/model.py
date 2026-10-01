import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

SAMPLE_RATE = 16000
class GPUSpecAugment(nn.Module):
    """GPU-accelerated SpecAugment using torchaudio built-ins."""
    def __init__(self, time_mask_param=10, freq_mask_param=8, p=0.5):
        super().__init__()
        self.p = p
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats shape expected: (B, n_mels, T)
        if self.training and (torch.rand(1).item() < self.p):
            feats = self.freq_mask(feats)
            feats = self.time_mask(feats)
        return feats


class AudioFeatureExtractorGPU(nn.Module):
    """Extracts Mel-Spectrogram, applies SpecAugment, and normalizes directly on GPU."""
    def __init__(self, n_mels=64, augment=False):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.augment = GPUSpecAugment() if augment else None

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        # 1. Mel Spectrogram (B, n_mels, T)
        mel = self.mel_transform(wavs)
        log_mel = torch.log(mel + 1e-6)

        # 2. SpecAugment (on raw log-mels)
        if self.augment is not None:
            log_mel = self.augment(log_mel)

        # 3. Transpose to (B, T, n_mels) for model
        feats = log_mel.transpose(1, 2)

        # 4. Instance Normalization across time steps T
        mean = feats.mean(dim=1, keepdim=True)
        std = feats.std(dim=1, keepdim=True) + 1e-5
        return (feats - mean) / std

def cnn_receptive_field(num_layers: int, kernel_size: int, dilation_base: int = 2) -> int:
    """Receptive field, in frames, of a stack of `num_layers` causal conv
    blocks with kernel_size `kernel_size` and dilation doubling each layer
    (1, 2, 4, ..., dilation_base**(num_layers-1))."""
    dilation_sum = dilation_base**num_layers - 1  # 1+2+4+...+2^(n-1) = 2^n - 1
    return 1 + (kernel_size - 1) * dilation_sum


def required_cnn_layers(window_frames: int, kernel_size: int, dilation_base: int = 2) -> int:
    """Smallest number of dilated causal conv layers whose receptive field
    covers `window_frames`. This is what the earlier fixed 2-layer/kernel=5
    CNN was missing -- its receptive field was ~13 frames (130ms), nowhere
    near enough context to judge a speaker change against a multi-second
    window."""
    n = 1
    while cnn_receptive_field(n, kernel_size, dilation_base) < window_frames:
        n += 1
    return n

class CausalConv1d(nn.Module):
    """Conv1d with explicit causal (left) padding."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.stride = stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        # Pad only on the left (past) side
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class ResTCNBlockCausal(nn.Module):
    """Residual dilated causal block."""
    def __init__(self, ch, kernel_size=5, dilation=1, dropout=0.15):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size=kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=1)  # 1x1 conv is inherently causal
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(F.gelu(self.bn1(self.conv1(x))))
        return F.gelu(x + self.bn2(self.conv2(y)))


class CausalSCDNet(nn.Module):
    """Fully Causal, CPU-Optimized Speaker Change Detection Network."""
    def __init__(
        self,
        n_mels: int = 64,
        ch: int = 64,
        kernel_size: int = 5,
        dilations: tuple = (1, 2, 4, 8),
        dropout: float = 0.15,
        augment: bool = False,
        n_outputs: int = 2,
    ):
        super().__init__()
        self.frontend = AudioFeatureExtractorGPU(n_mels=n_mels, augment=augment)
        
        # Causal downsampling stem (10ms -> 40ms frame resolution)
        self.stem = nn.Sequential(
            CausalConv1d(n_mels, ch, kernel_size=5, stride=2),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            CausalConv1d(ch, ch, kernel_size=5, stride=2),
            nn.BatchNorm1d(ch),
            nn.GELU(),
        )
        
        # Causal Residual TCN backbone
        self.blocks = nn.Sequential(
            *[ResTCNBlockCausal(ch, kernel_size, d, dropout) for d in dilations]
        )
        self.head = nn.Conv1d(ch, n_outputs, 1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for b in self.blocks:
            nn.init.zeros_(b.bn2.weight)

    def forward(self, wav):
        x = self.frontend(wav).transpose(1, 2)  # (B, n_mels, T_in)
        T_in = x.shape[-1]
        
        out = self.head(self.blocks(self.stem(x)))  # Downsampled sequence (B, n_outputs, T_down)
        
        # Causal upsampling using nearest neighbor (repeats last valid state into future frame grid)
        return F.interpolate(out, size=T_in, mode="nearest")