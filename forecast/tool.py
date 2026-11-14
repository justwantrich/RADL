"""Training loop, evaluation, and export utilities for the RADL pipeline.

Reversible instance normalization now lives inside the network, so the training,
validation, and test paths share one forward contract and cannot desynchronize
the normalization from its inverse.
"""
import os
import time
import torch
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from evaluate import regression_metrics, diebold_mariano
from load_data import reverse_scale, read_data
from data_rules import fit_ar1, ar1_forecast, flagged_rows


def _sanitize_for_filename(s: str) -> str:
    s = str(s)
    for ch in ("\\", "/", " ", ":"):
        s = s.replace(ch, "_")
    return s


def _infer_market_name(args) -> str:
    """Primary market (target column) label used for run tagging."""
    feat_names = getattr(args, "feature_names", None)
    if isinstance(feat_names, (list, tuple)) and len(feat_names) > 0:
        return str(feat_names[0])
    df = read_data(args.data_path, header=0, index_col=0)
    if df is None or len(df.columns) == 0:
        return "UnknownMarket"
    return str(df.columns[0])


def _variant_tag(args) -> str:
    """Compact description of the ablation arm, so parallel runs do not overwrite
    each other's checkpoints and result files."""
    if not getattr(args, "use_revin", False):
        rv = "norv"
    else:
        rv = "rv-" + getattr(args, "revin_anchor", "last")
        if not getattr(args, "revin_affine", True):
            rv += "-noaff"
    parts = [rv]
    if not getattr(args, "use_static_adj", True):
        parts.append("nostatic")
    if not getattr(args, "use_dynamic_adj", True):
        parts.append("nodyn")
    gating = getattr(args, "gating_method", "sigmoid")
    if gating != "sigmoid":
        parts.append(f"gate-{gating}")
    # Only departures from the C2+dx defaults enter the tag, so the default run
    # has a short name and every ablation arm gets its own files.
    for attr, default, label in (
            ("spike_rule", "vspike", "sp-"), ("loss", "huber", "loss-"),
            ("add_dt", True, "dt"), ("add_dx", True, "dx"),
            ("use_state_encoder", True, "se"), ("state_hidden", 16, "sh"),
            ("blocks", 3, "b"), ("layers", 3, "l"), ("early_stop_patience", 60, "es"),
            ("bottleneck_lambda", 0.001, "lam"), ("aux_loss_weight", 0.0, "aux")):
        v = getattr(args, attr, default)
        if v == default or (attr == "bottleneck_lambda" and gating == "none"):
            continue
        if isinstance(v, bool):
            parts.append(label if v else "no" + label)
        else:
            parts.append(f"{label}{v}")
    if getattr(args, "tag", ""):
        parts.append(_sanitize_for_filename(args.tag))
    return "_".join(parts)


def _build_run_tag(args) -> str:
    h = getattr(args, "forecast_horizon", "NA")
    d = getattr(args, "length", "NA")
    market = _sanitize_for_filename(_infer_market_name(args))
    return f"h={h}_d={d}_{market}_{_variant_tag(args)}_seed={getattr(args, 'seed', 'NA')}"


def _checkpoint_path(args) -> str:
    return f"./save_dict/{args.forecast_horizon}/{args.model_name}_{_build_run_tag(args)}.chkpt"


def _output_prefix(args) -> str:
    return f"{args.model_name}_{_build_run_tag(args)}"


def mse(pred, label):
    """Squared error on the headline target (variable 0), which is what every
    reported test metric uses."""
    loss = (pred[:, 0, :] - label[:, 0, :]) ** 2
    return torch.mean(loss)


def target_loss(pred, label, args):
    """Training loss on the headline target: MSE (original) or Huber.

    Huber's threshold is in the same z-units as the loss; with 'auto' it is the
    99th percentile of |daily change| on the training split, so only the
    rarest training days (the early thin-market jumps and residual bad
    prints, ~1% of days but 18-32% of the training SSE) are down-weighted.
    """
    if getattr(args, "loss", "mse") == "huber":
        return torch.nn.functional.huber_loss(pred[:, 0, :], label[:, 0, :],
                                              delta=args.huber_delta_value)
    return mse(pred, label)


def mae(pred, label):
    return torch.mean((pred - label).abs())


def _gate_penalty(model, args, epoch):
    """Sparsity penalty on the gates, ramped in for the sigmoid parameterization."""
    if getattr(model, "st_gating", None) is None:
        return 0.0
    lam = args.bottleneck_lambda
    if args.gating_method == "sigmoid":
        warmup = max(1, getattr(args, "bottleneck_warmup", 100))
        lam *= min(1.0, (epoch + 1) / warmup)
    return lam * model.st_gating.penalty()


def train_epoch(model, training_data, optimizer, device, args, epoch=0):
    model.train()
    total_loss = []
    aux_weight = getattr(args, 'aux_loss_weight', 0.0)
    clip = getattr(args, 'grad_clip', 0.0)

    if getattr(model, "st_gating", None) is not None and args.gating_method == "gumbel":
        # Geometric temperature anneal from tau_start to tau_end over training.
        frac = epoch / max(1, args.epoch - 1)
        ratio = args.gumbel_tau_end / args.gumbel_tau_start
        model.st_gating.tau = args.gumbel_tau_start * (ratio ** frac)

    for eod, gt in training_data:
        Eod, Gt = eod.to(device), gt.to(device)

        optimizer.zero_grad()
        pred = model(Eod).reshape(-1, args.code_num, args.forecast_horizon)

        loss = target_loss(pred, Gt, args)
        if aux_weight > 0:
            loss = loss + aux_weight * ((pred[:, 1:, :] - Gt[:, 1:, :]) ** 2).mean()
        loss = loss + _gate_penalty(model, args, epoch)

        loss.backward()
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss.append(loss.item())

    return sum(total_loss) / len(total_loss)


def _collect(model, loader, device, args):
    """Run the model over a loader and return (ground truth, prediction),
    both (n_windows, code_num, horizon) numpy arrays."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for eod, gt in loader:
            pred = model(eod.to(device)).reshape(-1, args.code_num, args.forecast_horizon)
            preds.append(pred.cpu().numpy())
            gts.append(gt.numpy())
    return np.concatenate(gts, axis=0), np.concatenate(preds, axis=0)


def eval_epoch(model, validation_data, device, args, a=None, b=None):
    """Validation error on the headline target only.

    The previous version averaged over all variables, so checkpoint selection
    optimised the fit to Oil, Coal, Gold and the rest rather than to the carbon
    price that every reported metric is computed on.
    """
    valid_gt, valid_pred = _collect(model, validation_data, device, args)
    y_true = valid_gt[:, 0, :].reshape(-1)
    y_pred = valid_pred[:, 0, :].reshape(-1)
    MSE = mean_squared_error(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    return MSE, MAE, sqrt(MSE)


def test_epoch(model, test_data, device, args):
    return _collect(model, test_data, device, args)


def train(model, training_data, validation_data, optimizer, device, args, a, b, scheduler=None):
    model_path = _checkpoint_path(args)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    best_val_mse = float('inf')
    best_epoch = -1
    total_train_time = 0.0
    patience = getattr(args, 'early_stop_patience', 0)
    epochs_no_improve = 0

    sep = "=" * 90
    print(f"\n{sep}")
    print("                              TRAINING STARTED")
    print(sep)
    aux_w = getattr(args, 'aux_loss_weight', 0.0)
    if aux_w > 0:
        print(f"  [Aux Loss] weight = {aux_w}")
    if patience > 0:
        print(f"  [Early Stop] patience = {patience}")
    if scheduler is not None:
        print(f"  [LR Schedule] CosineAnnealing")
    print(f"{'Epoch':>11s} | {'Train Loss':>11s} | {'Val MSE':>10s} | "
          f"{'Val MAE':>10s} | {'Val RMSE':>10s} | {'LR':>10s} | {'Time':>6s} | Save")
    print("-" * 90)

    for epoch_i in range(args.epoch):
        epoch_start = time.time()

        train_loss = train_epoch(model, training_data, optimizer, device, args=args, epoch=epoch_i)
        valid_mse, valid_mae, valid_rmse = eval_epoch(model, validation_data, device, args, a, b)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        current_lr = optimizer.param_groups[0]['lr']

        saved_flag = ""
        if valid_mse < best_val_mse:
            best_val_mse = valid_mse
            best_epoch = epoch_i
            epochs_no_improve = 0
            torch.save({'model': model.state_dict(), 'settings': args, 'epoch': epoch_i}, model_path)
            saved_flag = "<-- Best"
        else:
            epochs_no_improve += 1

        print(f"[{epoch_i + 1:4d}/{args.epoch:4d}] | "
              f"{train_loss:11.6f} | {valid_mse:10.6f} | "
              f"{valid_mae:10.6f} | {valid_rmse:10.6f} | "
              f"{current_lr:10.6f} | {epoch_time:5.1f}s | {saved_flag}")

        if patience > 0 and epochs_no_improve >= patience:
            print(f"\n  [Early Stop] No improvement for {patience} epochs. Stopping at epoch {epoch_i + 1}.")
            break

    print(sep)
    print(f"  Training complete!")
    print(f"  Total time   : {total_train_time:.1f}s ({total_train_time / 60:.1f} min)")
    print(f"  Best Val MSE : {best_val_mse:.6f} (epoch {best_epoch + 1})")
    print(f"  Model saved  : {os.path.abspath(model_path)}")
    print(sep, "\n")


def _naive_forecast(raw_y, start_idx, n_windows, horizon):
    """Random-walk benchmark: every step of window k is predicted by the last
    observation available at that window's forecast origin."""
    origins = raw_y[start_idx - 1: start_idx - 1 + n_windows]
    return np.repeat(origins, horizon)


def test(model, test_data, optimizer, device, args, a, b):
    model_path = _checkpoint_path(args)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    print(f"[Info] Loaded best model from epoch {checkpoint['epoch'] + 1}")

    test_gt, test_pred = test_epoch(model, test_data, device, args)
    test_pred = reverse_scale(args, test_pred, a, b)
    test_gt = reverse_scale(args, test_gt, a, b)

    df = read_data(args.data_path, header=0, index_col=0)
    if getattr(args, 'drop_cols', None):
        df = df.drop(columns=[c for c in args.drop_cols if c in df.columns])
    dates_index = df.index
    raw_y = df.iloc[:, 0].values.astype(np.float64)

    N, H = test_pred.shape[0], args.forecast_horizon

    # Window k, horizon step h -> data row (valid_index + length + k + h).
    start_idx = args.valid_index + args.length
    max_row_idx = start_idx + (N - 1) + (H - 1)
    if start_idx < 1 or max_row_idx >= len(dates_index):
        raise ValueError(
            f"Date index out of range when constructing test_dates: "
            f"start_idx={start_idx}, max_row_idx={max_row_idx}, len(df)={len(dates_index)}."
        )

    dates_list = [dates_index[start_idx + k + h] for k in range(N) for h in range(H)]
    test_dates = np.asarray(dates_list)

    y_true_1d = test_gt[:, 0, :].reshape(-1)
    y_pred_1d = test_pred[:, 0, :].reshape(-1)
    y_naive_1d = _naive_forecast(raw_y, start_idx, N, H)

    assert len(test_dates) == len(y_true_1d) == len(y_pred_1d) == len(y_naive_1d)
    # The inverse-scaled ground truth must equal the raw panel it came from.
    expected = np.array([raw_y[start_idx + k + h] for k in range(N) for h in range(H)])
    assert np.allclose(y_true_1d, expected, rtol=1e-6, atol=1e-6), \
        "Inverse-scaled ground truth does not match the raw series; window/date alignment is wrong."

    # Second benchmark: AR(1) on the target's log return, fitted on the
    # training split only (the same, possibly spike-repaired, series the
    # network was trained on).
    n_train = int(args.trade_day * args.train_ratio)
    y_train = getattr(args, 'train_target_raw', raw_y[:n_train])
    ar_c, ar_phi = fit_ar1(y_train)
    y_ar1_1d = ar1_forecast(raw_y, start_idx, N, H, ar_c, ar_phi)
    print(f"[Info] AR(1) benchmark fitted on train: c={ar_c:+.5f}, phi={ar_phi:+.4f}")

    # Rows flagged by the V-spike rule (spike row and its reversal row).  Used
    # for reporting only; the test data itself is never altered.
    flag_rows = set(flagged_rows(raw_y, getattr(args, 'spike_thresh', 0.2),
                                 getattr(args, 'spike_reversal', 0.8)).tolist())
    target_rows = np.array([start_idx + k + h for k in range(N) for h in range(H)])
    flagged = np.isin(target_rows, list(flag_rows))

    def _block(mask, sample):
        yt = y_true_1d[mask]
        fc = ((args.model_name, y_pred_1d[mask]), ("Naive-RW", y_naive_1d[mask]),
              ("AR(1)", y_ar1_1d[mask]))
        out = []
        for name, yhat in fc:
            out.append([sample, int(mask.sum()), name, *regression_metrics(yt, yhat)])
        e_m, e_rw, e_ar = yt - fc[0][1], yt - fc[1][1], yt - fc[2][1]
        # Diebold-Mariano: negative statistic + small p = model is more accurate.
        extra = [diebold_mariano(e_m, e_rw, h=H, power=2), diebold_mariano(e_m, e_rw, h=H, power=1),
                 diebold_mariano(e_m, e_ar, h=H, power=2), diebold_mariano(e_m, e_ar, h=H, power=1)]
        out[0] += [v for pair in extra for v in pair]
        for r in out[1:]:
            r += [np.nan] * 8
        return out

    rows = _block(np.ones_like(flagged), "all")
    if flagged.any():
        rows += _block(~flagged, "excl_flagged")
        print(f"[Info] {int(flagged.sum())} test target day(s) flagged by the V-spike rule: "
              f"{[str(d) for d in test_dates[flagged]]}")

    result = pd.DataFrame(rows, columns=["Sample", "n", "Model", "RMSE", "MAE", "MAPE", "RMSLE", "U1", "d",
                                         "DM_vs_RW_sq", "p_RW_sq", "DM_vs_RW_abs", "p_RW_abs",
                                         "DM_vs_AR1_sq", "p_AR1_sq", "DM_vs_AR1_abs", "p_AR1_abs"])
    result["Horizon"] = args.forecast_horizon
    result["Seed"] = args.seed
    result["Variant"] = _variant_tag(args)

    pred_df = pd.DataFrame({
        "Date": test_dates,
        "TRUE": y_true_1d,
        "Pred": y_pred_1d,
        "Naive": y_naive_1d,
        "AR1": y_ar1_1d,
        "Flagged": flagged,
    })

    out_dir = "./output/{}".format(args.forecast_horizon)
    os.makedirs(out_dir, exist_ok=True)
    prefix = _output_prefix(args)
    pred_path = f"{out_dir}/pred_df_{prefix}.csv"
    result_path = f"{out_dir}/{prefix}_result.csv"
    pred_df.to_csv(pred_path, index=False)
    result.to_csv(result_path, index=False)
    print(f"[Info] Predictions saved to {pred_path}")
    print(f"[Info] Results saved to {result_path}")

    return result


def _self_check():
    """Alignment checks for the naive benchmark. Run: python -m tool"""
    raw = np.arange(100, dtype=float)

    # h=1: the benchmark for row r is row r-1.
    nv = _naive_forecast(raw, start_idx=90, n_windows=5, horizon=1)
    assert np.array_equal(nv, [89, 90, 91, 92, 93]), nv

    # h=3: every step of a window uses that window's own origin, never a value
    # observed after it.
    nv = _naive_forecast(raw, start_idx=90, n_windows=2, horizon=3)
    assert np.array_equal(nv, [89, 89, 89, 90, 90, 90]), nv

    print("tool self-check OK")


if __name__ == "__main__":
    _self_check()
