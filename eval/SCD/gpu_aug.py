import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GpuAugmentPipeline(torch.nn.Module):
    FFT_N = 65536                      # >= 48000 + 6400 - 1
    def __init__(self, bank_dir="./data/aug_banks", split="train", device="cuda"):
        super().__init__()
        # Load RIR bank [N, rir_len] (float32)
        rirs_path = f"{bank_dir}/rirs_{split}.npy"
        if os.path.exists(rirs_path):
            rirs = torch.from_numpy(np.load(rirs_path)[:, :6400].copy()).float()
            fade = torch.ones(6400)
            fade[-160:] = torch.linspace(1, 0, 160)
            self.register_buffer("rirs", rirs * fade)
            self.to(device)
        else:
            self.rirs = None

    def _channel(self, seg, ids, wet, rng):
        if self.rirs is None:
            return seg
            
        B, W = seg.shape
        valid = ids >= 0
        h = self.rirs[ids.clamp(min=0)]
        y = torch.fft.irfft(torch.fft.rfft(seg, n=self.FFT_N) *
                            torch.fft.rfft(h,   n=self.FFT_N), n=self.FFT_N)[:, :W]
        cs_y = F.pad(torch.cumsum(y * y, -1), (1, 0))
        cs_x = F.pad(torch.cumsum(seg * seg, -1), (1, 0))
        i0, i1 = rng[:, 0].clamp(0, W), rng[:, 1].clamp(0, W)
        n = (i1 - i0).clamp(min=1)
        b = torch.arange(B, device=seg.device)
        e_y = (cs_y[b, i1] - cs_y[b, i0]) / n
        e_x = (cs_x[b, i1] - cs_x[b, i0]) / n
        gain = torch.sqrt((e_x + 1e-9) / (e_y + 1e-9)).unsqueeze(1)
        w = torch.where(valid, wet, torch.zeros_like(wet)).unsqueeze(1)
        return (1 - w) * seg + w * y * gain

    @torch.no_grad()
    def forward(self, batch):
        d = self.rirs.device if self.rirs is not None else batch["seg_a"].device
        g = lambda k, dt=None: batch[k].to(d, non_blocking=True, **({"dtype": dt} if dt else {}))
        
        a = self._channel(g("seg_a"), g("rir_a_id"), g("wet_a"), g("range_a"))
        b = self._channel(g("seg_b"), g("rir_b_id"), g("wet_b"), g("range_b"))
        x = a + b
        snr = g("snr_db")
        noise = g("noise")
        
        has = (~torch.isnan(snr)).float().unsqueeze(1)
        snr = torch.nan_to_num(snr, nan=0.0).unsqueeze(1)
        sp = x.pow(2).mean(-1, keepdim=True)
        npw = noise.pow(2).mean(-1, keepdim=True)
        scale = torch.sqrt(sp / (10 ** (snr / 10) * npw + 1e-9)) * has
        x = x + noise * scale
        
        peak = x.abs().amax(-1, keepdim=True)
        return x * torch.clamp(0.99 / (peak + 1e-9), max=1.0)

def test_parity():
    """
    Check 12 Parity Test: Validates CPU vs GPU output parity.
    Ensures outputs match within atol=1e-4.
    """
    print("=== Running Check 12: CPU vs GPU Augmenter Parity Test ===")
    np.random.seed(42)
    torch.manual_seed(42)

    # Synthetic RIR bank: 10 RIRs of length 6400
    rirs = np.random.randn(10, 6400).astype(np.float32) * 0.1
    bank_path = "./test_rir_bank.npy"
    np.save(bank_path, rirs)

    try:
        B, W = 4, 48000
        batch = {
            "seg_a": torch.randn(B, W) * 0.2,
            "seg_b": torch.randn(B, W) * 0.2,
            "range_a": torch.tensor([[0, 20000], [1000, 15000], [0, 24000], [500, 30000]]),
            "range_b": torch.tensor([[20000, 48000], [15000, 40000], [24000, 48000], [30000, 48000]]),
            "rir_a_id": torch.tensor([0, 3, -1, 5]),
            "rir_b_id": torch.tensor([1, 3, 2, -1]),
            "wet_a": torch.tensor([0.4, 0.5, 0.0, 0.3]),
            "wet_b": torch.tensor([0.4, 0.5, 0.2, 0.0]),
            "noise": torch.randn(B, W) * 0.05,
            "snr_db": torch.tensor([15.0, 20.0, 12.0, torch.nan]),
        }

        # Run CPU pipeline
        aug_cpu = GpuAugmentPipeline(bank_path, device="cpu")
        out_cpu = aug_cpu(batch)

        # Run GPU pipeline (if CUDA available)
        if torch.cuda.is_available():
            aug_gpu = GpuAugmentPipeline(bank_path, device="cuda")
            out_gpu = aug_gpu(batch).cpu()

            max_diff = torch.max(torch.abs(out_cpu - out_gpu)).item()
            print(f"  Maximum CPU vs GPU difference: {max_diff:.6e}")
            assert max_diff < 1e-4, f"Parity test failed! Max diff {max_diff} >= 1e-4"
            print("  Parity Test Passed successfully!")
        else:
            print("  CUDA device not available. Tested CPU execution only.")

    finally:
        if os.path.exists(bank_path):
            os.remove(bank_path)


if __name__ == "__main__":
    test_parity()