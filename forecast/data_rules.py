"""Data-aligned rules for the test-phase experiments (exp_data_aligned).

Everything here is a fixed, pre-declared rule or is fitted on the training split
only, so nothing estimated from the validation or test period can leak into
training.

1. V-spike detection.  A row t is flagged when the log move into t exceeds
   `thresh` and the next move reverses at least `reversal` of it.  Such
   one-day round trips are
   either bad prints or off-market block trades near compliance deadlines;
   either way they are not forecastable from the lookback window.
   Detection at row t needs row t+1, so:
     * in the training split the flagged price may be *repaired* (it is all
       observed before training starts);
     * in the validation/test split nothing is altered -- flagged rows are only
       used to report a second, "excluding flagged days" set of metrics.
2. Calendar-gap channel: log1p(calendar days since the previous row).  Fully
   causal: the gap into row t is known at row t.
3. AR(1) on the target's log return, fitted by OLS on the training split, as a
   second benchmark next to the random walk.
"""
import numpy as np
import pandas as pd


def detect_vspikes(y, thresh=0.2, reversal=0.8):
    """Indices t (1..n-2) of one-day round-trip spikes in a positive price series."""
    y = np.asarray(y, dtype=np.float64)
    r = np.diff(np.log(y))                    # r[t-1] = log(y[t] / y[t-1])
    out = []
    for t in range(1, len(y) - 1):
        a, b = r[t - 1], r[t]
        if abs(a) > thresh and a * b < 0 and abs(b) >= reversal * abs(a):
            out.append(t)
    return np.asarray(out, dtype=int)


def repair_train_spikes(y, n_train, thresh=0.2, reversal=0.8):
    """Replace flagged spike prices inside the training split by the geometric
    mean of their two neighbours.  Rows >= n_train are never touched, and a
    spike whose reversal row lies outside the training split is left alone
    (it could not be identified at training time)."""
    y = np.asarray(y, dtype=np.float64).copy()
    spikes = [t for t in detect_vspikes(y, thresh, reversal) if t + 1 < n_train]
    for t in spikes:
        y[t] = np.sqrt(y[t - 1] * y[t + 1])
    return y, spikes


def flagged_rows(y, thresh=0.2, reversal=0.8):
    """Rows whose one-step forecast error is dominated by a spike: the spike row
    itself and the reversal row after it."""
    s = detect_vspikes(y, thresh, reversal)
    return np.unique(np.concatenate([s, s + 1])) if len(s) else np.asarray([], dtype=int)


def calendar_gap_feature(index):
    """log1p(calendar days since the previous row); the first row gets 1 day."""
    dates = pd.to_datetime(pd.Index(index))
    gaps = np.diff(dates.values).astype('timedelta64[D]').astype(np.float64)
    gaps = np.concatenate([[1.0], np.maximum(gaps, 1.0)])
    return np.log1p(gaps)


def fit_ar1(y_train):
    """OLS fit of r_{t+1} = c + phi * r_t on training-split log returns."""
    r = np.diff(np.log(np.asarray(y_train, dtype=np.float64)))
    X = np.column_stack([np.ones(len(r) - 1), r[:-1]])
    c, phi = np.linalg.lstsq(X, r[1:], rcond=None)[0]
    return float(c), float(phi)


def ar1_forecast(raw_y, start_idx, n_windows, horizon, c, phi):
    """Iterated AR(1) price forecast, aligned with tool._naive_forecast:
    window k's origin is row start_idx - 1 + k."""
    out = np.empty(n_windows * horizon)
    for k in range(n_windows):
        o = start_idx - 1 + k
        r_prev = np.log(raw_y[o] / raw_y[o - 1])
        level, j = np.log(raw_y[o]), k * horizon
        for h in range(horizon):
            r_prev = c + phi * r_prev
            level += r_prev
            out[j + h] = np.exp(level)
    return out


