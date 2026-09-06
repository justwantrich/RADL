"""Training loop, evaluation, and export utilities for the RADL pipeline.

Reversible instance normalization now lives inside the network, so the training,
validation, and test paths share one forward contract and cannot desynchronize
the normalization from its inverse.
"""
import os
import time
from collections import deque
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from evaluate import regression_metrics, diebold_mariano
from load_data import reverse_scale, read_data


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
    if getattr(args, 'backbone_norm', 'batch') != 'batch':
        parts.append('norm-' + args.backbone_norm)
    if getattr(args, 'gcn_order', 2) != 2:
        parts.append('order' + str(args.gcn_order))
    if getattr(args, 'refit_train_valid', False):
        parts.append('refit' + str(args.refit_epochs))
    if getattr(args, 'use_movement_gate', False):
        parts.append('movegate')
    if getattr(args, 'checkpoint_metric', 'mse') != 'mse':
        parts.append('select-' + args.checkpoint_metric)
    mae_weight = getattr(args, 'target_mae_weight', 0.0)
    if mae_weight > 0:
        parts.append(f"maew{mae_weight:g}".replace('.', 'p'))
    if getattr(args, 'use_lag_graph', False):
        parts.append('laggraph-' + '-'.join(str(lag) for lag in args.graph_lags))
    att_gain = getattr(args, "att_init_gain", None)
    # Preserve the historical baseline tag for the original Xavier gain;
    # non-baseline ablations get an explicit suffix to prevent collisions.
    if att_gain is not None and abs(float(att_gain) - 1.414) > 1e-12:
        parts.append(f"att-g{att_gain:g}".replace(".", "p"))
    if not getattr(args, "use_static_adj", True):
        parts.append("nostatic")
    gating = getattr(args, "gating_method", "none")
    if gating != "none":
        parts.append(f"gate-{gating}")
    return "_".join(parts)


def _build_run_tag(args) -> str:
    h = getattr(args, "forecast_horizon", "NA")
    d = getattr(args, "length", "NA")
    market = _sanitize_for_filename(_infer_market_name(args))
    return f"h={h}_d={d}_{market}_{_variant_tag(args)}_seed={getattr(args, 'seed', 'NA')}"


def _checkpoint_path(args) -> str:
    return os.path.join(getattr(args, 'save_dir', './save_dict'),
                        str(args.forecast_horizon),
                        f"{args.model_name}_{_build_run_tag(args)}.chkpt")


def _output_prefix(args) -> str:
    return f"{args.model_name}_{_build_run_tag(args)}"


def mse(pred, label):
    """Squared error on the headline target (variable 0), which is what every
    reported test metric uses."""
    loss = (pred[:, 0, :] - label[:, 0, :]) ** 2
    return torch.mean(loss)


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

        loss = mse(pred, Gt)
        mae_weight = getattr(args, 'target_mae_weight', 0.0)
        if mae_weight > 0:
            loss = loss + mae_weight * (pred[:, 0, :] - Gt[:, 0, :]).abs().mean()
        if getattr(model, 'movement_conv', None) is not None:
            origin = Eod[:, 0, -1, 0].unsqueeze(-1)
            movement = ((Gt[:, 0, :] - origin).abs()
                        > getattr(args, 'movement_threshold', 0.02)).to(pred.dtype)
            loss = loss + getattr(args, 'movement_loss_weight', 0.005) * F.binary_cross_entropy_with_logits(
                model.last_movement_logits, movement)
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
    best_score = float('inf')
    selection_metric = getattr(args, 'checkpoint_metric', 'mse')
    best_epoch = -1
    total_train_time = 0.0
    patience = getattr(args, 'early_stop_patience', 0)
    epochs_no_improve = 0
    average_k = max(0, int(getattr(args, 'weight_average_k', 0)))
    recent_states = deque(maxlen=average_k or None)
    parameter_names = set(dict(model.named_parameters()).keys())

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
    if average_k > 0:
        print(f"  [Weight Average] final {average_k} epoch checkpoints")
    print(f"{'Epoch':>11s} | {'Train Loss':>11s} | {'Val MSE':>10s} | "
          f"{'Val MAE':>10s} | {'Val RMSE':>10s} | {'LR':>10s} | {'Time':>6s} | Save")
    print("-" * 90)

    for epoch_i in range(args.epoch):
        epoch_start = time.time()

        train_loss = train_epoch(model, training_data, optimizer, device, args=args, epoch=epoch_i)
        valid_mse, valid_mae, valid_rmse = eval_epoch(model, validation_data, device, args, a, b)

        if scheduler is not None:
            scheduler.step()

        if average_k > 0:
            # Keep CPU snapshots so averaging does not consume GPU memory.
            recent_states.append({
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            })

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        current_lr = optimizer.param_groups[0]['lr']

        saved_flag = ""
        current_score = valid_mse if selection_metric == 'mse' else valid_mae
        if current_score < best_score:
            best_score = current_score
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
    if selection_metric != 'mse':
        print(f"  Selected by  : validation {selection_metric.upper()} = {best_score:.6f}")
    print(f"  Model saved  : {os.path.abspath(model_path)}")
    print(sep, "\n")

    averaged = False
    average_start_epoch = None
    if average_k > 0 and recent_states:
        average_start_epoch = epoch_i - len(recent_states) + 2
        averaged_state = {}
        for name in recent_states[0]:
            values = [state[name] for state in recent_states]
            if name in parameter_names:
                averaged_state[name] = torch.stack(values, dim=0).mean(dim=0).to(values[0].dtype)
            else:
                # Keep non-parameter buffers from the final state. In
                # particular, averaging BatchNorm running statistics is not
                # equivalent to recalibrating them and can hurt evaluation.
                averaged_state[name] = values[-1]
        model.load_state_dict(averaged_state)
        torch.save({
            'model': averaged_state,
            'settings': args,
            'epoch': epoch_i,
            'averaged': True,
            'averaged_start_epoch': average_start_epoch,
            'averaged_end_epoch': epoch_i + 1,
        }, model_path)
        averaged = True
        print(f"  Averaged model : epochs {average_start_epoch}-{epoch_i + 1} ({len(recent_states)} states)")

    return {
        'best_val_mse': best_val_mse,
        'best_epoch': best_epoch,
        'completed_epochs': epoch_i + 1,
        'averaged': averaged,
        'averaged_start_epoch': average_start_epoch,
    }


def train_fixed(model, training_data, optimizer, device, args, scheduler=None):
    """Refit a chosen configuration on all pre-test data for fixed epochs."""
    model_path = _checkpoint_path(args)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for epoch_i in range(args.refit_epochs):
        loss = train_epoch(model, training_data, optimizer, device, args=args, epoch=epoch_i)
        if scheduler is not None:
            scheduler.step()
        print(f"[Refit {epoch_i + 1:4d}/{args.refit_epochs:4d}] loss={loss:.6f}")
    torch.save({'model': model.state_dict(), 'settings': args,
                'epoch': args.refit_epochs - 1, 'refit': True}, model_path)
    print(f"[Info] Refit model saved: {os.path.abspath(model_path)}")


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

    rows = []
    for name, yhat in ((args.model_name, y_pred_1d), ("Naive-RW", y_naive_1d)):
        RMSE, MAE, MAPE, RMSLE, u1, d = regression_metrics(y_true_1d, yhat)
        rows.append([name, RMSE, MAE, MAPE, RMSLE, u1, d])

    # Diebold-Mariano against the random walk. Negative statistic and small
    # p-value = the model is significantly more accurate than persistence.
    dm_mse, p_mse = diebold_mariano(y_true_1d - y_pred_1d, y_true_1d - y_naive_1d, h=H, power=2)
    dm_mae, p_mae = diebold_mariano(y_true_1d - y_pred_1d, y_true_1d - y_naive_1d, h=H, power=1)

    result = pd.DataFrame(rows, columns=["Model", "RMSE", "MAE", "MAPE", "RMSLE", "U1", "d"])
    result["DM_vs_RW_sq"] = [dm_mse, np.nan]
    result["p_sq"] = [p_mse, np.nan]
    result["DM_vs_RW_abs"] = [dm_mae, np.nan]
    result["p_abs"] = [p_mae, np.nan]
    result["Horizon"] = args.forecast_horizon
    result["Seed"] = args.seed
    result["Variant"] = _variant_tag(args)

    pred_df = pd.DataFrame({
        "Date": test_dates,
        "TRUE": y_true_1d,
        "Pred": y_pred_1d,
        "Naive": y_naive_1d,
    })

    out_dir = os.path.join(getattr(args, 'output_dir', './output'),
                           str(args.forecast_horizon))
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
