"""Training entry point for the RADL forecasting implementation.

Loads data through the shared preprocessing path, fits the dependency-aware
backbone selected in config, and writes checkpoints and test metrics using
the existing layout under ./save_dict and ./output.
"""
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from load_data import load_EOD_data, prepare_dataloaders, prepare_refit_loader
from tool import train, train_fixed, test
from config import get_args
from set_model import get_model


def inspect_data(eod_data, train_loader, valid_loader, test_loader, args, a, b):
    sep = "=" * 70
    print(f"\n{sep}")
    print("                    DATA INSPECTION REPORT")
    print(sep)

    print("\n[1] Raw Data Preview (eod_data after train-split standardization)")
    print(f"    eod_data.shape: {eod_data.shape}  "
          f"-> (code_num={args.code_num}, trade_day={args.trade_day}, fea_num={args.fea_num})")
    print(f"    First 5 time steps of variable 0:")
    print(f"      {eod_data[0, :5, 0]}")
    if eod_data.shape[0] > 1:
        print(f"    First 5 time steps of variable 1:")
        print(f"      {eod_data[1, :5, 0]}")

    print("\n[2] Batch Dimension Check")
    for name, loader in [("Train", train_loader), ("Valid", valid_loader), ("Test", test_loader)]:
        for X, y in loader:
            print(f"    {name:5s} -> X.shape: {tuple(X.shape)}, y.shape: {tuple(y.shape)}")
            break
    print(f"    Expected X: (batch, {args.code_num}, {args.length}, {args.fea_num})")
    print(f"    Expected y: (batch, {args.code_num}, {args.forecast_horizon})")

    print("\n[3] Dataset Split")
    print(f"    Train samples : {len(train_loader.dataset):>6d}   batches: {len(train_loader)}")
    print(f"    Valid samples : {len(valid_loader.dataset):>6d}   batches: {len(valid_loader)}")
    print(f"    Test  samples : {len(test_loader.dataset):>6d}   batches: {len(test_loader)}")

    print("\n[4] Alignment Check (first batch)")
    for X, y in train_loader:
        print(f"    Batch dim match (X[0]==y[0]): {X.shape[0] == y.shape[0]}")
        print(f"    Code  dim match (X[1]==y[1]): {X.shape[1] == y.shape[1]}")
        print(f"    X[0, 0, :5, 0] = {X[0, 0, :5, 0].numpy()}")
        print(f"    y[0, 0, :]     = {y[0, 0, :].numpy()}")
        break

    print("\n[5] Statistical Features (per variable, standardized)")
    print(f"    {'Var':>4s}  {'Mean':>9s}  {'Std':>9s}  {'Min':>9s}  {'Max':>9s}  {'NaN':>5s}")
    print(f"    {'-' * 50}")
    for i in range(args.code_num):
        vals = eod_data[i, :, 0]
        print(f"    {i:4d}  {np.mean(vals):+9.4f}  {np.std(vals):9.4f}  "
              f"{np.min(vals):+9.4f}  {np.max(vals):+9.4f}  {np.isnan(vals).sum():5d}")

    print(f"\n[6] Standardization: {args.standardization} on the first "
          f"{int(args.trade_day * args.train_ratio)} rows (training split only)")
    if a and b:
        col_names = getattr(args, 'feature_names', [])
        print(f"    {'Var':>4s}  {'Name':<12s}  {'Mean(a)':>10s}  {'Std(b)':>10s}")
        print(f"    {'-' * 42}")
        for i in range(args.code_num):
            vname = col_names[i] if i < len(col_names) else f"Var_{i}"
            print(f"    {i:4d}  {vname:<12s}  {a[i]:10.4f}  {b[i]:10.4f}")

    print(f"\n{sep}")
    print("                  DATA INSPECTION COMPLETE")
    print(f"{sep}\n")


def print_config(args):
    sep = "=" * 70
    print(f"\n{sep}")
    print("                    TRAINING CONFIGURATION")
    print(sep)
    print(f"  Data path         : {args.data_path}")
    print(f"  Model             : {args.model_name}")
    print(f"  Forecast horizon  : {args.forecast_horizon}")
    print(f"  Sequence length   : {args.length}")
    print(f"  Code num          : {args.code_num}")
    print(f"  Trade days        : {args.trade_day}")
    print(f"  Split ratios      : train {args.train_ratio} / valid {args.valid_ratio}")
    print(f"  Train index       : {args.train_index}")
    print(f"  Valid index       : {args.valid_index}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  Epochs            : {args.epoch}")
    print(f"  Learning rate     : {args.lr}")
    print(f"  Weight decay      : {args.weight_decay} (Adam)")
    print(f"  Grad clip         : {args.grad_clip}")
    print(f"  Dropout           : {args.dropout}")
    print(f"  Standardization   : {args.standardization} (train-split statistics)")
    print(f"  Seed              : {args.seed}")
    print(f"  CUDA              : {args.cuda}")
    print(f"  --- RADL components ---")
    print(f"  RevIN             : {args.use_revin}"
          + (f"  (anchor={args.revin_anchor}, affine={args.revin_affine})" if args.use_revin else ""))
    print(f"  Backbone          : blocks={args.blocks}, layers={args.layers}, "
          f"residual={args.residual_channels}, dilation={args.dilation_channels}, "
          f"skip={args.skip_channels}, end={args.end_channels}, gcn_order={args.gcn_order}, "
          f"att_dim={args.att_dim}")
    print(f"  Attention init    : Xavier uniform (gain={args.att_init_gain})")
    print(f"  Backbone norm     : {getattr(args, 'backbone_norm', 'batch')}")
    if getattr(args, 'use_lag_graph', False):
        print(f"  Lag-aware graph   : lags={args.graph_lags}")
    if getattr(args, 'weight_average_k', 0) > 0:
        print(f"  Weight averaging  : final {args.weight_average_k} epochs")
    print(f"  Static adjacency  : {args.use_static_adj}")
    gating = getattr(args, 'gating_method', 'none')
    print(f"  Gating method     : {gating}")
    if gating != "none":
        print(f"    bottleneck_lambda : {args.bottleneck_lambda}")
        print(f"    gate_lr_scale     : {args.gate_lr_scale}x")
        if gating == "sigmoid":
            print(f"    bottleneck_warmup : {args.bottleneck_warmup} epochs")
        elif gating == "gumbel":
            print(f"    tau               : {args.gumbel_tau_start} -> {args.gumbel_tau_end}")
    if args.aux_loss_weight > 0:
        print(f"  Aux loss weight   : {args.aux_loss_weight}")
    if getattr(args, 'target_mae_weight', 0.0) > 0:
        print(f"  Target MAE weight : {args.target_mae_weight}")
    if getattr(args, 'use_movement_gate', False):
        print(f"  Movement gate     : threshold={args.movement_threshold}, "
              f"loss_weight={args.movement_loss_weight}")
    if args.early_stop_patience > 0:
        print(f"  Early stop        : patience={args.early_stop_patience}")
    else:
        print(f"  Early stop        : disabled")
    if args.use_cosine_lr:
        print(f"  LR schedule       : CosineAnnealing (T_max={args.epoch})")
    print(f"  Checkpoint metric : {getattr(args, 'checkpoint_metric', 'mse')}")
    if getattr(args, 'refit_train_valid', False):
        print(f"  Final refit       : train+validation, {args.refit_epochs} epochs")
    print(sep)


def print_model_info(model, args, device):
    sep = "=" * 70
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{sep}")
    print("                       MODEL SUMMARY")
    print(sep)
    print(f"  Model name      : {args.model_name}")
    print(f"  Device          : {device}")
    print(f"  Total params    : {total_params:>12,d}")
    print(f"  Trainable params: {trainable_params:>12,d}")
    print(f"  Receptive field : {model.receptive_field} steps (lookback {args.length})")
    if model.receptive_field < args.length:
        print(f"  [Warn] Receptive field is shorter than the lookback: the last "
              f"{args.length - model.receptive_field} oldest steps are unused. "
              f"Raise -blocks or -layers to use them.")
    print(sep, "\n")


def build_param_groups(model, args):
    """Adam parameter groups.

    Weight decay is withheld from biases, BatchNorm, the RevIN affine terms and
    the gate logits.  Decaying RevIN's gamma is actively harmful: it is the
    divisor in the inverse mapping, so shrinking it toward zero inflates the
    denormalized forecast.  Decaying the gate logits would pull every gate to 0.5.
    """
    no_decay, decay, gates = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'bottleneck_logits' in name:
            gates.append(p)
        elif name.startswith('revin.') or 'bias' in name or name.startswith('bn.'):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [{'params': decay, 'weight_decay': args.weight_decay},
              {'params': no_decay, 'weight_decay': 0.0}]
    if gates:
        groups.append({'params': gates, 'weight_decay': 0.0,
                       'lr': args.lr * args.gate_lr_scale})
    return groups


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    eod_data, a, b = load_EOD_data(args)
    train_loader, valid_loader, test_loader = prepare_dataloaders(eod_data, args)

    inspect_data(eod_data, train_loader, valid_loader, test_loader, args, a, b)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = get_model(device, args)

    print_model_info(model, args, device)

    optimizer = optim.Adam(build_param_groups(model, args), lr=args.lr)

    scheduler = None
    if args.use_cosine_lr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr * 0.01)

    if getattr(args, 'refit_train_valid', False):
        if args.refit_epochs <= 0:
            raise ValueError('-refit_epochs must be positive when -refit_train_valid is true')
        refit_loader = prepare_refit_loader(eod_data, args)
        train_fixed(model, refit_loader, optimizer, device, args, scheduler=scheduler)
    else:
        train(model, train_loader, valid_loader, optimizer, device, args, a, b, scheduler=scheduler)

    result = test(model, test_loader, optimizer, device, args, a, b)

    if getattr(model, 'st_gating', None) is not None:
        _print_gating_summary(model, test_loader, device, args)

    return result


def _print_gating_summary(model, test_loader, device, args):
    """Average test-set gating weights, ranked as a variable-importance table."""
    model.eval()
    gate_accum, n_samples = None, 0
    with torch.no_grad():
        for eod, _ in test_loader:
            eod = eod.to(device)
            gate_w = model.st_gating(eod).squeeze(-1).squeeze(-1)  # (B, N)
            gate_accum = gate_w.sum(dim=0) if gate_accum is None else gate_accum + gate_w.sum(dim=0)
            n_samples += eod.size(0)
    avg_w = (gate_accum / n_samples).cpu().numpy()

    names = list(getattr(args, 'feature_names', []))
    names += [f"Var_{i}" for i in range(len(names), args.code_num)]
    ranked = sorted(zip(names, avg_w), key=lambda t: t[1], reverse=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"         VARIABLE IMPORTANCE RANKING  [{args.gating_method.upper()} Gate]")
    print(sep)
    print(f"  {'Rank':>4s}  {'Variable':<16s}  {'Avg Gate Weight':>15s}  {'Status':<10s}")
    print(f"  {'-' * 50}")
    for rank, (name, w) in enumerate(ranked, 1):
        print(f"  {rank:4d}  {name:<16s}  {w:15.4f}  {'RETAINED' if w > 0.5 else 'SUPPRESSED':<10s}")
    print(f"\n  Retained / Total: {sum(1 for _, w in ranked if w > 0.5)} / {args.code_num}")
    print(sep, "\n")


if __name__ == '__main__':
    args = get_args()
    print_config(args)
    all_result = main(args)

    sep = "=" * 70
    print(f"\n{sep}")
    print("                    FINAL TEST RESULTS")
    print(sep)
    print(all_result.to_string(index=False))
    print(sep)
