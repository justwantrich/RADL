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
    ratio = args.valid_ratio if getattr(args, 'refit_train_valid', False) else args.train_ratio
    n_train = int(args.trade_day * ratio)
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

    data = df.T.values.astype(np.float64)
    new_data, a, b = scale(args, data)
    eod_data = new_data.reshape(args.code_num, args.trade_day, args.fea_num)

    return eod_data, a, b


def get_batch(eod_data, offset, seq_len, out_len):
    return (eod_data[:, offset:offset + seq_len, :],
            eod_data[:, offset + seq_len:offset + seq_len + out_len, 0])


def make_window_arrays(eod_data, args):
    """Materialise all chronological sliding windows before any split."""
    n_windows = eod_data.shape[1] - args.length - args.forecast_horizon + 1
    EOD = np.empty((n_windows, args.code_num, args.length, args.fea_num), dtype=np.float32)
    GT = np.empty((n_windows, args.code_num, args.forecast_horizon), dtype=np.float32)
    for i in range(n_windows):
        EOD[i], GT[i] = get_batch(eod_data, i, args.length, args.forecast_horizon)
    return EOD, GT


def prepare_dataloaders(eod_data, args):
    """Sliding-window tensors wrapped in chronologically split DataLoaders."""
    EOD, GT = make_window_arrays(eod_data, args)

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


def prepare_refit_loader(eod_data, args):
    """Use all pre-test windows after model choices have been fixed on validation."""
    EOD, GT = make_window_arrays(eod_data, args)
    dataset = Data.TensorDataset(torch.from_numpy(EOD[:args.valid_index]),
                                 torch.from_numpy(GT[:args.valid_index]))
    return Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
