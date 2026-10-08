"""eval_utils.py -- metrics that measure what the product needs, for fixed-window SCD.

Changes vs. the previous version
  * ROI (region of interest): the real app only trusts detections whose peak lies in
    [0.5 s, 2.5 s] of the 3 s window. Every metric here (AP/AUROC score, event
    outcomes, false alarms by kind, lag, recall-by-evidence) now looks ONLY at that
    region. Probabilities are smoothed over the FULL window first (so edge context is
    respected) and then everything outside the ROI is zeroed. Pass roi_s=None to get the
    old whole-window behaviour (regression-tested to be bit-identical).
  * false_alarms_per_minute() divides by the ROI span (2 s of "decided" audio per
    window), not by the 3 s window.
  * full_report(..., kinds=...) also returns per-kind false-alarm rates / FA-per-minute
    at the tuned threshold and the mean ROI peak score per kind.
  * threshold_for_fa_budget(): pick the lowest threshold that meets a false-alarm
    budget on no-change audio (what the captioning UI actually feels).

Important distinction on metrics:
  - AP / AUROC are *detection-only* window-level metrics (does this window contain a
    change in the ROI?). They do not check where the peak is.
  - Event-level metrics (Precision, Recall, F1) measure *localization* accuracy within
    a temporal collar (e.g., 200 ms).
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import rankdata

try:
    from sklearn.metrics import average_precision_score as _sklearn_ap
except ImportError:
    _sklearn_ap = None

HOP_S = 0.01
DEFAULT_ROI_S = (0.5, 2.5)   # span of the 3 s window that the real app acts on


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def smooth(probs: np.ndarray, k: int = 5) -> np.ndarray:
    """1D uniform smoothing filter matching scipy.ndimage.uniform_filter1d(mode='nearest')."""
    if k <= 1:
        return probs.copy() if isinstance(probs, np.ndarray) else np.array(probs)
    return uniform_filter1d(probs.astype(np.float32), size=k, axis=-1, mode="nearest")


def roi_bounds(n_frames: int, roi_s=DEFAULT_ROI_S, hop_s: float = HOP_S):
    """Frame slice [lo, hi) covering roi_s (inclusive of the end frame)."""
    if roi_s is None:
        return 0, n_frames
    lo = max(int(round(roi_s[0] / hop_s)), 0)
    hi = min(int(round(roi_s[1] / hop_s)) + 1, n_frames)
    return lo, hi


def prepare(probs, smooth_k: int = 1, roi_s=DEFAULT_ROI_S, hop_s: float = HOP_S) -> np.ndarray:
    """Smooth over the whole window, then zero everything outside the ROI. Idempotent."""
    p = smooth(np.asarray(probs), smooth_k)
    lo, hi = roi_bounds(p.shape[-1], roi_s, hop_s)
    if lo > 0 or hi < p.shape[-1]:
        p[..., :lo] = 0.0
        p[..., hi:] = 0.0
    return p


def true_change_time(labels: np.ndarray, hop_s: float = HOP_S, thr: float = 0.5):
    """(has_change[N] bool, t_change[N] seconds, nan where no change).

    Note: a window whose change lies OUTSIDE the ROI can never be hit when roi_s is set
    (counted as a miss). The current sampler draws t_change in [0.5, 2.5] so this never
    happens with the training/val sets.
    """
    has = labels.max(axis=1) > thr
    t = np.where(has, labels.argmax(axis=1) * hop_s, np.nan)
    return has, t


# ---------------------------------------------------------------------------
# Threshold-free detection metrics
# ---------------------------------------------------------------------------
def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware Average Precision."""
    y = y.astype(bool)
    if not y.any():
        return float("nan")
    if _sklearn_ap is not None:
        return float(_sklearn_ap(y, scores))

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


# ---------------------------------------------------------------------------
# Event-level metrics
# ---------------------------------------------------------------------------
def event_outcomes(probs, has, t_true, thr, collar_s, smooth_k=1, hop_s=HOP_S, roi_s=DEFAULT_ROI_S):
    p = prepare(probs, smooth_k, roi_s, hop_s)
    t_pred = p.argmax(axis=1) * hop_s
    fired = p.max(axis=1) >= thr
    hit = fired & has & (np.abs(t_pred - np.nan_to_num(t_true)) <= collar_s)
    tp = int(hit.sum())
    fp = int((fired & ~hit).sum())   # false alarm, or right window but wrong place
    fn = int((has & ~hit).sum())     # missed, or fired in the wrong place
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                tp=tp, fp=fp, fn=fn, thr=float(thr))


def sweep_threshold(probs, has, t_true, collar_s, grid=None, smooth_k=1, hop_s=HOP_S, roi_s=DEFAULT_ROI_S):
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    p = prepare(probs, smooth_k, roi_s, hop_s)
    return max((event_outcomes(p, has, t_true, t, collar_s, smooth_k=1, hop_s=hop_s, roi_s=roi_s) for t in grid),
               key=lambda d: d["f1"])


def select_threshold(probs, labels, collar_s=0.2, grid=None, smooth_k=1, hop_s=HOP_S, roi_s=DEFAULT_ROI_S) -> float:
    """Select optimal threshold on validation set based on event F1 score."""
    has, t = true_change_time(labels, hop_s)
    best_event = sweep_threshold(probs, has, t, collar_s, grid=grid, smooth_k=smooth_k, hop_s=hop_s, roi_s=roi_s)
    return float(best_event["thr"])


def evaluate_at(probs, labels, thr: float, collar_s=0.2, smooth_k=1, hop_s=HOP_S, roi_s=DEFAULT_ROI_S) -> dict:
    """Evaluate performance using a fixed, pre-selected threshold (e.g., on test set)."""
    has, t = true_change_time(labels, hop_s)
    return event_outcomes(probs, has, t, thr, collar_s, smooth_k=smooth_k, hop_s=hop_s, roi_s=roi_s)


def lag_report(probs, has, t_true, smooth_k=1, hop_s=HOP_S, roi_s=DEFAULT_ROI_S):
    """Peak time minus true change time (ms) over windows that contain a change.

    The median alone is a poor monitor (it sits at 0 for any unbiased model), so MAE is
    reported as well.
    """
    if not has.any():
        return {}
    p = prepare(probs, smooth_k, roi_s, hop_s)
    lag = (p[has].argmax(axis=1) * hop_s - t_true[has]) * 1000.0
    return dict(median_ms=float(np.median(lag)), mae_ms=float(np.mean(np.abs(lag))),
                p10_ms=float(np.percentile(lag, 10)), p90_ms=float(np.percentile(lag, 90)),
                within_100ms=float(np.mean(np.abs(lag) <= 100)),
                within_200ms=float(np.mean(np.abs(lag) <= 200)))


def recall_by_evidence(probs, has, t_true, thr, collar_s, window_s=3.0,
                       edges=(0.0, 0.75, 1.25, 1.75, 2.25, 3.0), smooth_k=1, hop_s=HOP_S, roi_s=DEFAULT_ROI_S):
    """Recall bucketed by seconds of audio AFTER the change (window_s - t_change)."""
    p = prepare(probs, smooth_k, roi_s, hop_s)
    t_pred = p.argmax(axis=1) * hop_s
    hit = (p.max(axis=1) >= thr) & has & (np.abs(t_pred - np.nan_to_num(t_true)) <= collar_s)
    evidence = window_s - np.nan_to_num(t_true)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = has & (evidence > lo) & (evidence <= hi)
        out[f"{lo:.2f}-{hi:.2f}s"] = (int(m.sum()), float(hit[m].mean()) if m.any() else float("nan"))
    return out


# ---------------------------------------------------------------------------
# False alarms
# ---------------------------------------------------------------------------
def false_alarm_by_kind(probs, kinds: np.ndarray, thr: float, smooth_k=1, hop_s=HOP_S, roi_s=DEFAULT_ROI_S) -> dict:
    """
    Fire rate per window type ('nochange', 'fake', 'change'), looking only at the ROI peak
    (default 0.5-2.5 s, matching the real app).
    If fire rate on 'fake' >> 'nochange', the model relies on splice/channel artifacts.
    (For 'change' the "rate" is just the fraction of windows that fire at all, ignoring location.)
    """
    p = prepare(probs, smooth_k, roi_s, hop_s)
    fired = p.max(axis=1) >= thr
    kinds = np.asarray(kinds)

    rates = {}
    for k in np.unique(kinds):
        m = kinds == k
        rates[str(k)] = float(fired[m].mean()) if m.any() else float("nan")
    return rates


def false_alarms_per_minute(fire_rate: float, window_s: float = 3.0, roi_s=DEFAULT_ROI_S) -> float:
    """
    False alarms per minute of no-change audio.

    Assumes windows are stepped by the ROI span (so the ROIs tile the audio with no gaps
    and no double counting): with roi_s=(0.5, 2.5) each window "decides" 2 s of audio,
    i.e. 30 windows/min -> FA/min = fire_rate * 30.  With roi_s=None the whole window_s
    is used (20 windows/min for 3 s).  If the app steps by less than that and does NOT
    merge neighbouring detections, the real number is larger.
    """
    span = (roi_s[1] - roi_s[0]) if roi_s is not None else window_s
    return float(fire_rate * 60.0 / span)


def threshold_for_fa_budget(probs, kinds, labels, max_fa_per_min=1.0, smooth_k=5, collar_s=0.2,
                            hop_s=HOP_S, roi_s=DEFAULT_ROI_S, grid=None):
    """Lowest threshold whose no-change false-alarm rate fits the budget; returns its event metrics.

    Returns None if even the highest grid value exceeds the budget. Use this (not max-F1 on a
    40%-positive synthetic prior) to pick the deployment threshold: real streams contain far fewer
    changes than the synthetic val set, so max-F1 thresholds are too eager.
    """
    p = prepare(probs, smooth_k, roi_s, hop_s)
    score = p.max(axis=1)
    kinds = np.asarray(kinds)
    nc = kinds == "nochange"
    has, t = true_change_time(labels, hop_s)
    grid = np.linspace(0.02, 0.98, 97) if grid is None else grid
    for thr in grid:                      # ascending: fire rate is non-increasing in thr
        rate = float((score[nc] >= thr).mean()) if nc.any() else 0.0
        fa = false_alarms_per_minute(rate, roi_s=roi_s)
        if fa <= max_fa_per_min:
            out = event_outcomes(p, has, t, thr, collar_s, smooth_k=1, hop_s=hop_s, roi_s=roi_s)
            out["fa_per_min_nochange"] = fa
            return out
    return None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def slice_report(probs, labels, mask: np.ndarray, collars_s=(0.1, 0.2, 0.5), smooth_k=1, window_s=3.0,
                 hop_s=HOP_S, roi_s=DEFAULT_ROI_S, kinds=None) -> dict:
    """Run full_report over a specific subset defined by boolean mask (e.g. SNR bin, RIR)."""
    mask = np.asarray(mask, dtype=bool)
    p_sub = probs[mask]
    l_sub = labels[mask]
    if len(p_sub) == 0:
        return {}
    k_sub = None if kinds is None else np.asarray(kinds)[mask]
    return full_report(p_sub, l_sub, collars_s=collars_s, smooth_k=smooth_k, window_s=window_s,
                       hop_s=hop_s, roi_s=roi_s, kinds=k_sub)


def full_report(probs, labels, collars_s=(0.1, 0.2, 0.5), smooth_k=5, window_s=3.0, hop_s=HOP_S,
                roi_s=DEFAULT_ROI_S, kinds=None):
    # Smooth over the full window once, zero outside the ROI once
    p = prepare(probs, smooth_k, roi_s, hop_s)
    has, t = true_change_time(labels, hop_s)
    score = p.max(axis=1)

    rep = dict(n=len(probs), frac_change=float(has.mean()), roi_s=roi_s,
               AP=average_precision(score, has), AUROC=auroc(score, has))
    for c in collars_s:
        rep[f"event@{int(round(c * 1000))}ms"] = sweep_threshold(p, has, t, c, smooth_k=1, hop_s=hop_s, roi_s=roi_s)
    rep["lag"] = lag_report(p, has, t, smooth_k=1, hop_s=hop_s, roi_s=roi_s)
    best = rep["event@200ms"] if "event@200ms" in rep else next(v for k, v in rep.items() if k.startswith("event@"))
    rep["recall_by_evidence@200ms"] = recall_by_evidence(
        p, has, t, best["thr"], 0.2, window_s, smooth_k=1, hop_s=hop_s, roi_s=roi_s)
    rep["best_threshold"] = best["thr"]

    if kinds is not None:
        kinds = np.asarray(kinds)
        rep["fa_by_kind"] = false_alarm_by_kind(p, kinds, best["thr"], smooth_k=1, hop_s=hop_s, roi_s=roi_s)
        rep["fa_per_min"] = {k: false_alarms_per_minute(v, window_s, roi_s)
                             for k, v in rep["fa_by_kind"].items() if k in ("nochange", "fake")}
        rep["mean_score_by_kind"] = {str(k): float(score[kinds == k].mean()) for k in np.unique(kinds)}
    return rep


def print_report(rep):
    print(f"windows={rep['n']}  frac_with_change={rep['frac_change']:.2f}  roi={rep.get('roi_s')}  "
          f"AP={rep['AP']:.3f}  AUROC={rep['AUROC']:.3f}")
    for k, v in rep.items():
        if k.startswith("event@"):
            print(f"  {k:12s} F1={v['f1']:.3f}  P={v['precision']:.3f}  R={v['recall']:.3f}  (thr={v['thr']:.2f})")
    if rep.get("lag"):
        l = rep["lag"]
        print(f"  lag: median={l['median_ms']:+.0f} ms  MAE={l['mae_ms']:.0f}  p10={l['p10_ms']:+.0f}  "
              f"p90={l['p90_ms']:+.0f}  within100={l['within_100ms']:.2f}  within200={l['within_200ms']:.2f}")
    if rep.get("fa_by_kind"):
        fa = ", ".join(f"{k}={v:.3f}" for k, v in rep["fa_by_kind"].items())
        print(f"  fire rate @thr={rep['best_threshold']:.2f} (ROI only): {fa}")
        if rep.get("fa_per_min"):
            print("  FA/min: " + ", ".join(f"{k}={v:.2f}" for k, v in rep["fa_per_min"].items()))
    if rep.get("recall_by_evidence@200ms"):
        print("  recall@200ms by seconds of post-change audio:")
        for k, (n, r) in rep["recall_by_evidence@200ms"].items():
            print(f"    {k:11s} n={n:5d}  recall={r:.2f}")