"""Data loading, scaling, and DataLoader construction for RADL training.

Scaling statistics are estimated on the training split only.  Estimating them
on the full panel, as the previous version did, leaks the test-period level and
dispersion into every training sample -- which is hard to defend in a paper
whose contribution is about handling distribution shift.
"""
import os
import pandas as pd
import torch
import torch.utils.data as Data
import numpy as np

from data_rules import repair_train_spikes, calendar_gap_feature


def read_data(path, **kwargs):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path, **kwargs)
    elif ext in ('.xls', '.xlsx'):
        return pd.read_excel(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .csv or .xlsx")


def scale(args, data):
    """Per-variable z-score using training-split statistics only.

    Args:
        data: (code_num, trade_day) float array, modified in place.

    Returns:
        (scaled data, per-variable means, per-variable stds).
    """
    n_train = int(args.trade_day * args.train_ratio)
    if n_train < 2:
        raise ValueError(f"Training split has {n_train} rows; need at least 2 to estimate scale.")

    a_list, b_list = [], []
    for i in range(args.code_num):
        a = data[i, :n_train].mean()
        b = data[i, :n_train].std()
        if not np.isfinite(b) or b < 1e-8:
            # Constant over the training split: keep the level, drop the scaling.
            b = 1.0
        data[i, :] = (data[i, :] - a) / b
        a_list.append(a)
        b_list.append(b)
    return data, a_list, b_list


def reverse_scale(args, values, a, b):
    """Invert scale() on an array whose axis 1 indexes variables."""
    values = np.asarray(values, dtype=np.float64).copy()
    for i in range(args.code_num):
        values[:, i] = values[:, i] * b[i] + a[i]
    return values


def load_EOD_data(args):
    df = read_data(args.data_path, header=0, index_col=0)

    if getattr(args, 'drop_cols', None):
        df = df.drop(columns=[c for c in args.drop_cols if c in df.columns])

    if df.isna().any().any():
        raise ValueError(f"{args.data_path} contains missing values; fill or drop them before training.")

    args.feature_names = list(df.columns)
    args.code_num = len(df.columns)
    args.trade_day = len(df)
    n_train = int(args.trade_day * args.train_ratio)

    data = df.T.values.astype(np.float64)

    # V-spike repair, training split only (see data_rules).
    args.repaired_spikes = []
    if getattr(args, 'spike_rule', 'none') == 'vspike':
        data[0], spikes = repair_train_spikes(data[0], n_train,
                                              args.spike_thresh, args.spike_reversal)
        args.repaired_spikes = [str(df.index[t]) for t in spikes]
        print(f"[spike_rule] repaired {len(spikes)} training-split spike(s) in "
              f"{df.columns[0]}: {args.repaired_spikes}")
    # Target series as the model sees it on the training split (for the AR(1) fit).
    args.train_target_raw = data[0, :n_train].copy()

    new_data, a, b = scale(args, data)

    # Channel order: 0 price, then calendar gap (if -add_dt), then daily change (if -add_dx).
    channels = [new_data]
    if getattr(args, 'add_dt', False):
        gap = calendar_gap_feature(df.index)
        mu, sd = gap[:n_train].mean(), gap[:n_train].std()
        gap = (gap - mu) / (sd if sd > 1e-8 else 1.0)
        channels.append(np.broadcast_to(gap, (args.code_num, args.trade_day)))
    if getattr(args, 'add_dx', False):
        # Daily change of every variable, divided by its training-split std, so
        # the TCN reads lag-1..3 moves directly instead of differencing levels.
        dx = np.zeros_like(new_data)
        dx[:, 1:] = np.diff(new_data, axis=1)
        sd = dx[:, 1:n_train].std(axis=1, keepdims=True)
        channels.append(dx / np.where(sd > 1e-8, sd, 1.0))
    eod_data = np.stack(channels, axis=-1)
    args.fea_num = eod_data.shape[-1]

    # Huber threshold in the loss's own units (train-split z-scored target):
    # 99th percentile of the |H-step change| on the training split, so the rule
    # scales with the forecast horizon (H=1: |daily change|, as before).
    delta = str(getattr(args, 'huber_delta', 'auto'))
    if delta == 'auto':
        H = max(1, int(getattr(args, 'forecast_horizon', 1)))
        z = new_data[0, :n_train]
        dz = np.abs(z[H:] - z[:-H])
        args.huber_delta_value = float(np.quantile(dz, 0.99))
    else:
        args.huber_delta_value = float(delta)

    return eod_data, a, b


def get_batch(eod_data, offset, seq_len, out_len):
    return (eod_data[:, offset:offset + seq_len, :],
            eod_data[:, offset + seq_len:offset + seq_len + out_len, 0])


def prepare_dataloaders(eod_data, args):
    """Sliding-window tensors wrapped in chronologically split DataLoaders."""
    n_windows = eod_data.shape[1] - args.length - args.forecast_horizon + 1
    EOD = np.empty((n_windows, args.code_num, args.length, args.fea_num), dtype=np.float32)
    GT = np.empty((n_windows, args.code_num, args.forecast_horizon), dtype=np.float32)
    for i in range(n_windows):
        EOD[i], GT[i] = get_batch(eod_data, i, args.length, args.forecast_horizon)

    train_eod, train_gt = EOD[:args.train_index], GT[:args.train_index]
    valid_eod, valid_gt = EOD[args.train_index:args.valid_index], GT[args.train_index:args.valid_index]
    test_eod, test_gt = EOD[args.valid_index:], GT[args.valid_index:]

    for name, part in (("train", train_eod), ("valid", valid_eod), ("test", test_eod)):
        if len(part) == 0:
            raise ValueError(f"The {name} split is empty. Check -length, -forecast_horizon and the split ratios.")

    def loader(x, y, shuffle=False, drop_last=False):
        ds = Data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return Data.DataLoader(dataset=ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last)

    return (loader(train_eod, train_gt, shuffle=True, drop_last=True),
            loader(valid_eod, valid_gt),
            loader(test_eod, test_gt))
