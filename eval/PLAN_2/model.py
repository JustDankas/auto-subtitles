import torch
import torch.nn as nn


class CausalConv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):  # x: (B, C, T)
        x = nn.functional.pad(x, (self.pad, 0))  # left-pad only -> causal
        return self.act(self.norm(self.conv(x)))


class SCDModel(nn.Module):
    """
    backbone: "cnn" | "gru" | "lstm"
    input: (B, T, input_dim) log-mel/MFCC frames
    output: (B, T) raw logits, one per input frame (apply sigmoid for P(change))
    """
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=2, backbone="gru"):
        super().__init__()
        self.backbone_type = backbone

        if backbone == "cnn":
            layers, in_ch = [], input_dim
            for i in range(num_layers):
                layers.append(CausalConv1dBlock(in_ch, hidden_dim, kernel_size=5, dilation=2 ** i))
                in_ch = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, 1)

        elif backbone in ("gru", "lstm"):
            rnn_cls = nn.GRU if backbone == "gru" else nn.LSTM
            # unidirectional == causal; bidirectional would leak future context
            self.backbone = rnn_cls(input_dim, hidden_dim, num_layers,
                                     batch_first=True, bidirectional=False)
            self.head = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        if self.backbone_type == "cnn":
            x = self.backbone(x.transpose(1, 2)).transpose(1, 2)  # (B,T,hidden)
        else:
            x, _ = self.backbone(x)
        return self.head(x).squeeze(-1)  # (B, T)
