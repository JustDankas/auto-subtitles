import numpy as np
import pandas as pd
import torch
from dataset import SCDDataset


def frames_to_events(probs, frame_hop_ms=10.0, threshold=0.5, nms_window_ms=300):
    positive = probs > threshold
    events = []
    i = 0
    while i < len(positive):
        if positive[i]:
            j = i
            while j < len(positive) and positive[j]:
                j += 1
            peak = i + int(np.argmax(probs[i:j]))
            events.append(peak * frame_hop_ms / 1000.0)
            i = j
        else:
            i += 1
    # merge events closer than nms_window_ms
    merged = []
    for t in events:
        if merged and (t - merged[-1]) * 1000 < nms_window_ms:
            continue
        merged.append(t)
    return merged


def boundary_f1(pred_times, true_times, collar_s=0.5):
    matched_true, matched_pred = set(), set()
    for pi, pt in enumerate(pred_times):
        for ti, tt in enumerate(true_times):
            if ti in matched_true:
                continue
            if abs(pt - tt) <= collar_s:
                matched_true.add(ti)
                matched_pred.add(pi)
                break
    tp = len(matched_pred)
    precision = tp / max(len(pred_times), 1)
    recall = tp / max(len(true_times), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


# --- run over the held-out test split ---
def evaluate(work_dir: str, model, device="cuda", checkpoint_path=None):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    manifest = pd.read_csv(f"{work_dir}/data/synth_test/annotations/manifest.csv")
    test_ds = SCDDataset(f"{work_dir}/data/synth_test")

    for collar in (0.2, 0.5):
        precisions, recalls, f1s = [], [], []
        for idx, sid in enumerate(test_ds.session_ids):
            feats, _ = test_ds[idx]
            with torch.no_grad():
                probs = torch.sigmoid(model(feats.unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
            pred_times = frames_to_events(probs)
            g = manifest[manifest.file_id == sid].sort_values("start_s").reset_index(drop=True)
            true_times = [g.start_s[i] for i in range(1, len(g)) if g.speaker_id[i] != g.speaker_id[i - 1]]
            p, r, f1 = boundary_f1(pred_times, true_times, collar_s=collar)
            precisions.append(p); recalls.append(r); f1s.append(f1)
        print(f"collar={collar}s: P={np.mean(precisions):.3f} R={np.mean(recalls):.3f} F1={np.mean(f1s):.3f}")
