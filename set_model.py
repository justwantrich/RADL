"""Factory for the dependency-aware spatiotemporal backbone used in RADL."""
from model.gwnet_att import gwnet


def get_model(device, args):
    if args.model_name != "GWNet-Attn":
        raise ValueError(f"Unknown model_name {args.model_name!r}; only 'GWNet-Attn' is registered.")

    return gwnet(
        args.code_num, args.length,
        dropout=args.dropout,
        in_dim=args.fea_num,
        out_dim=args.forecast_horizon,
        residual_channels=getattr(args, 'residual_channels', 32),
        dilation_channels=getattr(args, 'dilation_channels', 32),
        skip_channels=getattr(args, 'skip_channels', 256),
        end_channels=getattr(args, 'end_channels', 512),
        kernel_size=2,
        blocks=args.blocks,
        layers=args.layers,
        att_dim=args.att_dim,
        att_init_gain=args.att_init_gain,
        use_lag_graph=getattr(args, 'use_lag_graph', False),
        graph_lags=getattr(args, 'graph_lags', (0, 1, 2, 5)),
        gcn_order=getattr(args, 'gcn_order', 2),
        backbone_norm=getattr(args, 'backbone_norm', 'batch'),
        use_movement_gate=getattr(args, 'use_movement_gate', False),
        gating_method=getattr(args, 'gating_method', 'none'),
        use_static_adj=getattr(args, 'use_static_adj', True),
        use_revin=getattr(args, 'use_revin', True),
        revin_anchor=getattr(args, 'revin_anchor', 'last'),
        revin_affine=getattr(args, 'revin_affine', True),
    ).to(device)
