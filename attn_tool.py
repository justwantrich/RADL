"""Attention export utilities: post-hoc inspection of the learned dependency structure.

The dependency matrix is now estimated per window rather than per mini-batch, so
the time-varying export below is a genuine per-day series instead of one point
per batch.  Drive this with attn_load_data.prepare_all_dataloaders.
"""
import numpy as np
import pandas as pd
import torch

from load_data import read_data


def collect_attention(model, loader, device, args):
    """Return (per-window adjacency, predictions).

    adjacency: (n_windows, N, N) with adj[k, i, j] the weight of source
    variable i when aggregating into target variable j for window k.
    """
    model.eval()
    atts, preds = [], []
    with torch.no_grad():
        for eod, _ in loader:
            pred, adj = model(eod.to(device), True)
            preds.append(pred.cpu().numpy().reshape(-1, args.code_num, args.forecast_horizon))
            atts.append(adj.cpu().numpy())
    return np.concatenate(atts, axis=0), np.concatenate(preds, axis=0)


def _column_names(args):
    names = list(getattr(args, 'feature_names', []) or [])
    if len(names) < args.code_num:
        df = read_data(args.data_path, header=0, index_col=0)
        if getattr(args, 'drop_cols', None):
            df = df.drop(columns=[c for c in args.drop_cols if c in df.columns])
        names = list(df.columns)
    return names[:args.code_num]


def export_attention(model, loader, device, args, out_dir=None):
    """Write the mean dependency matrix and the time-varying weights into the
    carbon-price node, and return both as DataFrames."""
    model_path = "./save_dict/{}/{}.chkpt".format(args.forecast_horizon, args.model_name)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    print(f"[Info] Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")

    att, _ = collect_attention(model, loader, device, args)
    names = _column_names(args)

    mean_att = pd.DataFrame(att.mean(axis=0), index=names, columns=names).round(4)
    print("Mean learned dependency matrix (rows = source, columns = target)")
    print(mean_att)

    # Incoming dependency weights of the target variable (column 0), one row per
    # window.  Window k's forecast origin is data row k + length - 1.
    df = read_data(args.data_path, header=0, index_col=0)
    if getattr(args, 'drop_cols', None):
        df = df.drop(columns=[c for c in args.drop_cols if c in df.columns])
    n = att.shape[0]
    dates = df.index[args.length - 1: args.length - 1 + n]
    into_target = pd.DataFrame(att[:, :, 0], columns=names)
    into_target.insert(0, "Date", np.asarray(dates))
    into_target = into_target.set_index("Date")

    if out_dir:
        mean_att.to_csv(f"{out_dir}/avg_att.csv")
        into_target.to_csv(f"{out_dir}/att_over_time.csv")
        print(f"[Info] Attention exports written to {out_dir}")

    return mean_att, into_target
