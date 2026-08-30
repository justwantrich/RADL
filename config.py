"""CLI and runtime configuration for the RADL forecasting codebase.

Regime-Aligned Dependency Learning (RADL) = a dependency-aware spatiotemporal
backbone (Graph WaveNet style) with (a) anchored reversible instance
normalization, (b) a per-window dependency matrix, and (c) optional
regime-aligned variable gating.
"""
import argparse
from load_data import read_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def set_split_indices(args):
    """Chronological window split.  Called from every entry point so the split
    used for training can never drift from the one used for scaling."""
    args.train_index = int(args.trade_day * args.train_ratio) - args.length - args.forecast_horizon
    args.valid_index = int(args.trade_day * args.valid_ratio) - args.length - args.forecast_horizon
    return args


def get_args():
    parser = argparse.ArgumentParser(
        description="RADL (Regime-Aligned Dependency Learning): unified multivariate forecasting."
    )

    parser.add_argument('-data_path', type=str, default="./data/GDEA.csv",
                        help="Path to the input data file (.csv or .xlsx)")
    parser.add_argument('-drop_cols', nargs='*', default=[],
                        help="Column names to drop from data (e.g. -drop_cols AQI Temp)")

    parser.add_argument('-length', '-L', type=int, default=21, dest='length',
                        help="Lookback length L (historical time steps per sample)")
    parser.add_argument('-code_num', type=int, default=16,
                        help="Number of variables / nodes (auto-detected from data)")
    parser.add_argument('-trade_day', type=int, default=2209,
                        help="Total trading days (auto-detected from data)")
    parser.add_argument('-fea_num', type=int, default=1,
                        help="Number of input features per variable")

    parser.add_argument('-train_ratio', type=float, default=0.7,
                        help="Fraction of rows used for training")
    parser.add_argument('-valid_ratio', type=float, default=0.8,
                        help="Cumulative fraction of rows up to the end of validation")

    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="Adam decoupled L2 penalty. NOTE: previously this multiplied a "
                             "hand-rolled sum of parameter norms that dominated the data-fit "
                             "term by 40-80x; it is now passed to the optimizer, so the old "
                             "default of 1e-3 is not comparable.")
    parser.add_argument('-grad_clip', type=float, default=5.0,
                        help="Max global gradient norm (0 = disabled)")

    parser.add_argument('-epoch', type=int, default=500, help="Number of training epochs")
    parser.add_argument('-batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('-hidden_size', type=int, default=64, help="Number of hidden units")

    parser.add_argument('-blocks', type=int, default=3, help="Number of dilation blocks")
    parser.add_argument('-layers', type=int, default=3, help="Dilated conv layers per block")
    parser.add_argument('-att_dim', type=int, default=8,
                        help="Embedding width of the latent correlation (attention) layer")
    parser.add_argument('-use_static_adj', type=str2bool, default=True,
                        help="Add a learned static adjacency alongside the per-window one")

    parser.add_argument('-forecast_horizon', type=int, default=1,
                        help="Forecasting horizon in steps")
    parser.add_argument('-standardization', type=str, default="z-score", choices=["z-score"],
                        help="Standardization method (train-split statistics only)")

    parser.add_argument('-model_name', type=str, default="GWNet-Attn",
                        help="Registered backbone name (selects the RADL network factory path)")
    parser.add_argument('-cuda', type=str2bool, default=True, help="Use CUDA if available")
    parser.add_argument('-avg_att', type=str2bool, default=True)
    parser.add_argument('-seed', type=int, default=123123, help="Random seed")

    # Reversible instance normalization
    parser.add_argument('-use_revin', type=str2bool, default=True,
                        help="Per-window reversible instance normalization inside the network")
    parser.add_argument('-revin_anchor', type=str, default="last", choices=["last", "mean"],
                        help="Location statistic: 'last' makes a zero output equal the random walk; "
                             "'mean' reduces to standard RevIN")
    parser.add_argument('-revin_affine', type=str2bool, default=True,
                        help="Learnable per-variable affine term (the RevIN paper's gamma/beta)")

    # Regime-aligned variable emphasis (gating)
    parser.add_argument('-gating_method', type=str, default="none",
                        choices=["none", "sigmoid", "gumbel"],
                        help="Gating parameterization: none | sigmoid | gumbel")
    parser.add_argument('-bottleneck_lambda', type=float, default=0.001,
                        help="Sparsity penalty weight on the gate probabilities")
    parser.add_argument('-gate_lr_scale', type=float, default=10.0,
                        help="Learning rate multiplier for gate parameters")
    parser.add_argument('-bottleneck_warmup', type=int, default=100,
                        help="[sigmoid] Penalty warm-up epochs (linear ramp)")
    parser.add_argument('-gumbel_tau_start', type=float, default=1.0,
                        help="[gumbel] Initial temperature (high = soft sampling)")
    parser.add_argument('-gumbel_tau_end', type=float, default=0.05,
                        help="[gumbel] Final temperature (low = hard selection)")

    # Auxiliary supervision on the non-target variables
    parser.add_argument('-aux_loss_weight', type=float, default=0.0,
                        help="Weight for auxiliary multi-variable MSE (0 = off)")

    # Optimization schedule
    parser.add_argument('-early_stop_patience', type=int, default=60,
                        help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument('-use_cosine_lr', type=str2bool, default=True,
                        help="Use cosine annealing LR schedule")

    args = parser.parse_args()

    data = read_data(args.data_path, header=0, index_col=0)
    if args.drop_cols:
        data = data.drop(columns=[c for c in args.drop_cols if c in data.columns])
    args.trade_day = len(data)
    args.code_num = len(data.columns)

    args.use_feature_bottleneck = args.gating_method != "none"
    set_split_indices(args)

    return args
