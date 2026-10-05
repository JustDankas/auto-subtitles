"""eval_utils.py -- metrics that measure what the product needs, for fixed-window SCD.

Important distinction on metrics:
  - AP / AUROC are *detection-only* window-level metrics. High window-level scores
    can occur even if the model's peak prediction within the window is mislocated.
  - Event-level metrics (Precision, Recall, F1) measure *localization* accuracy
    within a temporal collar (e.g., 200 ms).
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import rankdata

try:
    from sklearn.metrics import average_precision_score as _sklearn_ap
except ImportError:
    _sklearn_ap = None

HOP_S = 0.01


def smooth(probs: np.ndarray, k: int = 5) -> np.ndarray:
    """1D uniform smoothing filter matching scipy.ndimage.uniform_filter1d(mode='nearest')."""
    if k <= 1:
        return probs.copy() if isinstance(probs, np.ndarray) else np.array(probs)
    return uniform_filter1d(probs.astype(np.float32), size=k, axis=-1, mode="nearest")


def true_change_time(labels: np.ndarray, hop_s: float = HOP_S, thr: float = 0.5):
    """(has_change[N] bool, t_change[N] seconds, nan where no change)."""
    has = labels.max(axis=1) > thr
    t = np.where(has, labels.argmax(axis=1) * hop_s, np.nan)
    return has, t


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware Average Precision."""
    y = y.astype(bool)
    if not y.any():
        return float("nan")
    if _sklearn_ap is not None:
        return float(_sklearn_ap(y, scores))

    # Fallback tie-aware AP implementation
    sort_idx = np.argsort(-scores)
    scores_sorted = scores[sort_idx]
    y_sorted = y[sort_idx]

    distinct_value_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    precision = tps / (tps + fps)
    precision = np.r_[1.0, precision]
    recall = tps / tps[-1]
    recall = np.r_[0.0, recall]

    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware ROC AUC score via rankdata."""
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(scores)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def event_outcomes(probs, has, t_true, thr, collar_s, smooth_k=1, hop_s=HOP_S):
    p = smooth(probs, smooth_k)
    t_pred = p.argmax(axis=1) * hop_s
    fired = p.max(axis=1) >= thr
    hit = fired & has & (np.abs(t_pred - np.nan_to_num(t_true)) <= collar_s)
    tp = int(hit.sum())
    fp = int((fired & ~hit).sum())   # false alarm, or right window but wrong place
    fn = int((has & ~hit).sum())     # missed, or fired in the wrong place
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                tp=tp, fp=fp, fn=fn, thr=float(thr))


def sweep_threshold(probs, has, t_true, collar_s, grid=None, smooth_k=1, hop_s=HOP_S):
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    p = smooth(probs, smooth_k)
    return max((event_outcomes(p, has, t_true, t, collar_s, smooth_k=1, hop_s=hop_s) for t in grid),
               key=lambda d: d["f1"])


def select_threshold(probs, labels, collar_s=0.2, grid=None, smooth_k=1, hop_s=HOP_S) -> float:
    """Select optimal threshold on validation set based on event F1 score."""
    has, t = true_change_time(labels, hop_s)
    best_event = sweep_threshold(probs, has, t, collar_s, grid=grid, smooth_k=smooth_k, hop_s=hop_s)
    return float(best_event["thr"])


def evaluate_at(probs, labels, thr: float, collar_s=0.2, smooth_k=1, hop_s=HOP_S) -> dict:
    """Evaluate performance using a fixed, pre-selected threshold (e.g., on test set)."""
    has, t = true_change_time(labels, hop_s)
    p = smooth(probs, smooth_k)
    return event_outcomes(p, has, t, thr, collar_s, smooth_k=1, hop_s=hop_s)


def lag_report(probs, has, t_true, smooth_k=1, hop_s=HOP_S):
    """Peak time minus true change time (ms) over windows that contain a change."""
    if not has.any():
        return {}
    p = smooth(probs, smooth_k)
    lag = (p[has].argmax(axis=1) * hop_s - t_true[has]) * 1000.0
    return dict(median_ms=float(np.median(lag)), p10_ms=float(np.percentile(lag, 10)),
                p90_ms=float(np.percentile(lag, 90)),
                within_100ms=float(np.mean(np.abs(lag) <= 100)),
                within_200ms=float(np.mean(np.abs(lag) <= 200)))


def recall_by_evidence(probs, has, t_true, thr, collar_s, window_s=3.0,
                       edges=(0.0, 0.75, 1.25, 1.75, 2.25, 3.0), smooth_k=1, hop_s=HOP_S):
    """Recall bucketed by seconds of audio AFTER the change (window_s - t_change)."""
    p = smooth(probs, smooth_k)
    t_pred = p.argmax(axis=1) * hop_s
    hit = (p.max(axis=1) >= thr) & has & (np.abs(t_pred - np.nan_to_num(t_true)) <= collar_s)
    evidence = window_s - np.nan_to_num(t_true)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = has & (evidence > lo) & (evidence <= hi)
        out[f"{lo:.2f}-{hi:.2f}s"] = (int(m.sum()), float(hit[m].mean()) if m.any() else float("nan"))
    return out


def false_alarm_by_kind(probs, kinds: np.ndarray, thr: float, smooth_k=1) -> dict:
    """
    Computes fire rates across window types ('nochange', 'fake', etc.).
    If fire rate on 'fake' >> 'nochange', the model relies on splice/channel artifacts.
    """
    p = smooth(probs, smooth_k)
    scores = p.max(axis=1)
    fired = scores >= thr
    kinds = np.asarray(kinds)
    
    unique_kinds = np.unique(kinds)
    rates = {}
    for k in unique_kinds:
        m = kinds == k
        rates[str(k)] = float(fired[m].mean()) if m.any() else float("nan")
    return rates


def false_alarms_per_minute(fp_rate_nochange: float, window_s: float = 3.0) -> float:
    """
    Calculates false alarms per minute on no-change audio.
    
    Note: Valid for non-overlapping windows. Continuous streaming deployment
    with overlapping windows will scale this rate proportionally to step size.
    """
    return float(fp_rate_nochange * (60.0 / window_s))


def slice_report(probs, labels, mask: np.ndarray, collars_s=(0.1, 0.2, 0.5), smooth_k=1, window_s=3.0, hop_s=HOP_S) -> dict:
    """Run full_report over a specific subset defined by boolean mask (e.g. SNR bin, RIR)."""
    mask = np.asarray(mask, dtype=bool)
    p_sub = probs[mask]
    l_sub = labels[mask]
    if len(p_sub) == 0:
        return {}
    return full_report(p_sub, l_sub, collars_s=collars_s, smooth_k=smooth_k, window_s=window_s, hop_s=hop_s)


def full_report(probs, labels, collars_s=(0.1, 0.2, 0.5), smooth_k=5, window_s=3.0, hop_s=HOP_S):
    # Pre-smooth probabilities once for efficiency
    p = smooth(probs, smooth_k)
    has, t = true_change_time(labels, hop_s)
    score = p.max(axis=1)

    rep = dict(n=len(probs), frac_change=float(has.mean()),
               AP=average_precision(score, has), AUROC=auroc(score, has))
    for c in collars_s:
        rep[f"event@{int(round(c * 1000))}ms"] = sweep_threshold(p, has, t, c, smooth_k=1, hop_s=hop_s)
    rep["lag"] = lag_report(p, has, t, smooth_k=1, hop_s=hop_s)
    best = rep["event@200ms"] if "event@200ms" in rep else next(v for k, v in rep.items() if k.startswith("event@"))
    rep["recall_by_evidence@200ms"] = recall_by_evidence(
        p, has, t, best["thr"], 0.2, window_s, smooth_k=1, hop_s=hop_s)
    return rep


def print_report(rep):
    print(f"windows={rep['n']}  frac_with_change={rep['frac_change']:.2f}  "
          f"AP={rep['AP']:.3f}  AUROC={rep['AUROC']:.3f}")
    for k, v in rep.items():
        if k.startswith("event@"):
            print(f"  {k:12s} F1={v['f1']:.3f}  P={v['precision']:.3f}  R={v['recall']:.3f}  (thr={v['thr']:.2f})")
    if rep.get("lag"):
        l = rep["lag"]
        print(f"  lag: median={l['median_ms']:+.0f} ms  p10={l['p10_ms']:+.0f}  p90={l['p90_ms']:+.0f}  "
              f"within100={l['within_100ms']:.2f}  within200={l['within_200ms']:.2f}")
    if rep.get("recall_by_evidence@200ms"):
        print("  recall@200ms by seconds of post-change audio:")
        for k, (n, r) in rep["recall_by_evidence@200ms"].items():
            print(f"    {k:11s} n={n:5d}  recall={r:.2f}")