import torch
import torch.nn as nn
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

class CausalConv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, dropout=0.15):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):  # x: (B, C, T)
        x = nn.functional.pad(x, (self.pad, 0))  # left-pad only -> causal
        return self.dropout(self.act(self.norm(self.conv(x))))


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


class SCDModel(nn.Module):
    """
    backbone: "cnn" | "gru" | "lstm"
    input: (B, T, input_dim) log-mel/MFCC frames
    output: (B, T) raw logits, one per input frame (apply sigmoid for P(change))

    window_duration_s / frame_hop_ms describe the fixed-length window this
    model is trained and run on (see synthetic.py's windows mode) -- for the
    "cnn" backbone they're used to auto-size the network's depth so its
    receptive field actually covers the whole window; for "gru"/"lstm" they're
    stored for reference only, since an RNN run once per window with a fresh
    hidden state already sees the entire window regardless of depth.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = None,
        backbone: str = "gru",
        window_duration_s: float = 3.0,
        frame_hop_ms: float = 10.0,
        kernel_size: int = 7,
        augment: bool = False
    ):
        super().__init__()
        self.feature_extractor = AudioFeatureExtractorGPU(n_mels=input_dim, augment=augment)
        self.backbone_type = backbone
        self.window_duration_s = window_duration_s
        self.frame_hop_ms = frame_hop_ms
        self.window_frames = int(round(window_duration_s * 1000.0 / frame_hop_ms))

        if backbone == "cnn":
            if num_layers is None:
                num_layers = required_cnn_layers(self.window_frames, kernel_size)
                print(
                    f"[SCDModel] auto-selected {num_layers} CNN layers (kernel_size="
                    f"{kernel_size}) to cover a {window_duration_s}s ({self.window_frames}"
                    f"-frame) window"
                )
            rf = cnn_receptive_field(num_layers, kernel_size)
            if rf < self.window_frames:
                print(
                    f"[SCDModel] WARNING: receptive field is {rf} frames but the window "
                    f"is {self.window_frames} frames -- this model cannot see the start "
                    "of the window when judging its end. Increase num_layers or "
                    "kernel_size, or leave num_layers unset to auto-size it."
                )

            layers, in_ch = [], input_dim
            for i in range(num_layers):
                layers.append(
                    CausalConv1dBlock(in_ch, hidden_dim, kernel_size=kernel_size, dilation=2**i)
                )
                in_ch = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, 1)
            self.num_layers = num_layers
            self.receptive_field_frames = rf

        elif backbone in ("gru", "lstm"):
            num_layers = num_layers or 2
            rnn_cls = nn.GRU if backbone == "gru" else nn.LSTM
            # unidirectional == causal; bidirectional would leak future context
            self.backbone = rnn_cls(
                input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False
            )
            self.head = nn.Linear(hidden_dim, 1)
            self.num_layers = num_layers
            self.receptive_field_frames = None  # unbounded within the fed window
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        # 1. Convert raw audio batch (B, num_samples) to Log-Mel on GPU
        x = self.feature_extractor(x)  # Yields (B, T, input_dim)

        # 2. Run Backbone
        if self.backbone_type == "cnn":
            x = self.backbone(x.transpose(1, 2)).transpose(1, 2)  # (B,T,hidden)
        else:
            x, _ = self.backbone(x)
        return self.head(x).squeeze(-1)  # (B, T)
