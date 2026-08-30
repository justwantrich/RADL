"""Scalar regression metrics and predictive-accuracy tests for RADL."""
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error


def calculate_ui(y_true, y_pred):
    numerator = np.sqrt(np.sum((y_true - y_pred) ** 2))
    denominator = np.sqrt(np.sum(y_true ** 2))
    return numerator / denominator


def calculate_d(observed, predicted):
    """Willmott index of agreement between observed and predicted series."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mean_observed = np.mean(observed)
    numerator = np.sum(np.abs(observed - predicted))
    denominator = np.sum(np.abs(observed - mean_observed) + np.abs(predicted - mean_observed))
    return 1 - (numerator / denominator)


def regression_metrics(true, pred):
    true = np.asarray(true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    RMSE = np.sqrt(mean_squared_error(true, pred))
    MAE = mean_absolute_error(true, pred)
    MAPE = mean_absolute_percentage_error(true, pred)
    # RMSLE is undefined for non-positive values. A diverged model can emit a
    # negative price; returning NaN keeps a finished training run from dying at
    # the very last step.
    if (true <= 0).any() or (pred <= 0).any():
        RMSLE = np.nan
    else:
        RMSLE = np.sqrt(mean_squared_log_error(true, pred))
    u1 = calculate_ui(true, pred)
    d = calculate_d(true, pred)
    return RMSE, MAE, MAPE * 100, RMSLE * 100, u1 * 100, d


def diebold_mariano(e_model, e_base, h=1, power=2):
    """Diebold-Mariano test of equal predictive accuracy.

    Uses the Harvey, Leybourne & Newbold (1997) small-sample correction and a
    Student-t reference distribution.

    Args:
        e_model: forecast errors (actual - predicted) of the proposed model.
        e_base:  forecast errors of the benchmark.
        h:       forecast horizon; sets the truncation lag of the long-run variance.
        power:   1 for absolute-error loss, 2 for squared-error loss.

    Returns:
        (DM statistic, two-sided p-value).  A negative statistic means the
        proposed model has the lower loss.
    """
    e_model = np.asarray(e_model, dtype=float).ravel()
    e_base = np.asarray(e_base, dtype=float).ravel()
    if e_model.shape != e_base.shape:
        raise ValueError(f"error series must match: {e_model.shape} vs {e_base.shape}")

    d = np.abs(e_model) ** power - np.abs(e_base) ** power
    T = len(d)
    if T < 3:
        return np.nan, np.nan
    d_bar = d.mean()
    dev = d - d_bar
    gamma = [np.sum(dev[k:] * dev[:T - k]) / T for k in range(h)]
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    if not np.isfinite(V_d) or V_d <= 0:
        return np.nan, np.nan

    dm = d_bar / np.sqrt(V_d)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm *= correction
    p = 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))
    return dm, p


def _self_check():
    rng = np.random.default_rng(0)
    n = 400
    good = rng.normal(0, 1.0, n)
    bad = rng.normal(0, 2.0, n)

    dm, p = diebold_mariano(good, bad)
    assert dm < 0 and p < 0.01, f"clearly better model not detected: dm={dm}, p={p}"

    dm, p = diebold_mariano(bad, good)
    assert dm > 0 and p < 0.01, "sign must flip when arguments swap"

    dm, p = diebold_mariano(good, good.copy())
    assert np.isnan(dm), "identical error series has zero variance -> NaN"

    # RMSLE must degrade to NaN rather than raising on a negative prediction.
    m = regression_metrics(np.array([1.0, 2.0]), np.array([-1.0, 2.0]))
    assert np.isnan(m[3]), "RMSLE should be NaN for non-positive input"
    m = regression_metrics(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
    assert np.isfinite(m[3]), "RMSLE should be finite for positive input"

    print("evaluate self-check OK")


if __name__ == "__main__":
    _self_check()
