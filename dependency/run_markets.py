"""Auditable carbon-market runs using the downloaded CUTS+ implementation.

No coarse-to-fine grouping. Original model, losses and sampling are retained.
Only batch construction, progress reporting and result export are adapted.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import types

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import cuts_plus as upstream

ROOT = Path(__file__).resolve().parent
MARKETS = ["GDEA", "HBEA", "SHEA", "TJEA"]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def vector_batches(data, observ_mask, bs, n_nodes, input_step, pred_step, block_size=None):
    """Same shuffled full batches and dropped remainder as upstream, no scalar GPU copies."""
    indices = upstream.generate_indices(input_step, pred_step, len(data), block_size)
    # Upstream consumes the shuffled list with pop(), hence the reversal.
    indices = indices[::-1]
    for start in range(0, len(indices) - bs + 1, bs):
        t = torch.tensor(indices[start:start + bs], device=data.device, dtype=torch.long)
        ix = t[:, None] + torch.arange(-input_step, 0, device=data.device)
        iy = t[:, None] + torch.arange(pred_step, device=data.device)
        yield (data[ix].permute(0, 2, 1, 3), data[iy].permute(0, 2, 1, 3), t,
               observ_mask[ix].permute(0, 2, 1, 3), observ_mask[iy].permute(0, 2, 1, 3))


class EpochLogger:
    def __init__(self, directory):
        self.log_dir = str(directory)
        self.values = {}

    def log_metrics(self, metrics_dict, iters):
        for key, value in metrics_dict.items():
            self.values.setdefault(key, []).append(float(value))

    def log_figures(self, figure, **kwargs):
        import matplotlib.pyplot as plt
        plt.close(figure)


def graph_stats(prob, threshold):
    n = len(prob)
    values = prob[~np.eye(n, dtype=bool)]
    selected = values > threshold
    weights = values / values.sum() if values.sum() else np.zeros_like(values)
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-30))) / np.log(len(values))
    return dict(threshold=threshold, n_edges=int(selected.sum()), n_possible_edges=len(values),
                density=float(selected.mean()), sparsity=float(1-selected.mean()),
                weight_mean=float(values.mean()), weight_std=float(values.std()),
                weight_q10=float(np.quantile(values, .1)), weight_median=float(np.median(values)),
                weight_q90=float(np.quantile(values, .9)), weight_min=float(values.min()),
                weight_max=float(values.max()), normalized_weight_entropy=float(entropy),
                target_in_density=float((prob[1:, 0] > threshold).mean()),
                target_in_weight_mean=float(prob[1:, 0].mean()))


def load_panel(args, market):
    path = args.data_dir / f"{market}.csv"
    df = pd.read_csv(path)
    dates = pd.to_datetime(df.iloc[:, 0], errors="raise")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError(f"Unsorted or duplicate dates: {path}")
    raw = df.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError(f"Nonfinite data: {path}")
    count = int(len(raw) * args.train_fraction)
    if args.match_rows:
        count = min(count, args.match_rows)
    raw = raw[:count]
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    if (std < 1e-8).any():
        raise ValueError(f"Constant input variable: {path}")
    return ((raw - mean) / std).astype(np.float32), dict(
        market=market, source=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        total_rows=len(df), used_rows=count, start_date=str(dates.iloc[0].date()),
        end_date=str(dates.iloc[count-1].date()), train_fraction=args.train_fraction,
        columns=list(df.columns[1:]), scaling_mean=mean.tolist(), scaling_std=std.tolist(),
        scaling="per-variable z-score, ddof=0, fitted only on graph-training rows",
        row_axis="source", column_axis="target", diagonal_in_statistics=False)


def train_one(args, market, seed, multiplier):
    label = f"{market}_seed{seed}_lambda{multiplier:g}"
    directory = args.output / "runs" / label
    complete = directory / "complete.json"
    if complete.exists():
        previous = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        current_source = args.data_dir / f"{market}.csv"
        if hashlib.sha256(current_source.read_bytes()).hexdigest() != previous["sha256"]:
            raise RuntimeError(f"Data changed since completed run {label}; use a new output directory")
        for name, digest in previous["upstream_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Upstream source changed since completed run {label}; use a new output directory")
        print(f"SKIP {label} (complete)", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data, metadata = load_panel(args, market)
    cfg = OmegaConf.load(ROOT / "opt" / "lorenz_example.yaml").sota.cuts_plus
    cfg.n_nodes = data.shape[1]
    cfg.n_groups = data.shape[1]
    cfg.group_policy = "None"
    cfg.input_step = args.lookback
    cfg.total_epoch = args.epochs
    cfg.batch_size = args.batch_size
    cfg.supervision_policy = "full"
    cfg.fill_policy = "none"
    cfg.show_graph_every = args.epochs + 1
    cfg.graph_discov.lambda_s_start = args.lambda_start * multiplier
    cfg.graph_discov.lambda_s_end = args.lambda_end * multiplier
    if len(data) - args.lookback < args.batch_size:
        raise ValueError("Not enough training windows for a full batch")
    metadata.update(seed=seed, lambda_multiplier=multiplier,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), platform=platform.platform(),
                    upstream_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (ROOT / "cuts_plus.py", ROOT / "model/cuts_plus_net.py",
                                               ROOT / "utils/gumbel_softmax.py")},
                    notes=["New documented protocol; original carbon-market configuration not recovered.",
                           "Upstream graph-discovery sampling passes [p,1-p] as logits, not log probabilities.",
                           "Upstream graph_discov does not pass its scheduled tau to the sampler; actual tau=1.",
                           "Sampling semantics retained, not silently corrected.",
                           "No coarse-to-fine; all 16 source nodes have separate parameters from epoch 1.",
                           "Edge scores are learned Bernoulli gate probabilities, not signed effect sizes.",
                           "Unequal market sample lengths remain a cross-market comparison limitation."])
    write_json(directory / "metadata.json", metadata)
    OmegaConf.save(cfg, directory / "config.yaml")
    log = EpochLogger(directory)
    model = upstream.MultiCAD(cfg, log, device="cuda")
    if args.cuda_graphs:
        # Capture only deterministic neural forward/backward work. Graph sampling,
        # optimizer steps and schedules stay in the original Python methods.
        # fork_rng prevents the verification inputs from changing training RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            sample_x = torch.randn(cfg.batch_size, cfg.n_nodes, cfg.input_step, 1, device="cuda")
            sample_mask = torch.ones_like(sample_x)
            sample_graph = torch.ones(cfg.batch_size, cfg.n_nodes, cfg.n_nodes, device="cuda", requires_grad=True)
            reference = model.fitting_model(sample_x, sample_mask, sample_graph)
            reference_grads = torch.autograd.grad(reference.sum(), (sample_graph, *model.fitting_model.parameters()))
            reference = reference.detach().clone()
            reference_grads = [g.detach().clone() for g in reference_grads]
            def local_forward_on_device(layer, x):
                # Original allocates on CPU then copies to CUDA, which cannot be captured.
                out = x.new_zeros((x.shape[0], layer.out_channel, x.shape[2]))
                for i in range(x.shape[2]):
                    out[..., i] = layer.conv_list[i](x[..., i].unsqueeze(-1)).squeeze(-1)
                return out
            for layer in model.fitting_model.modules():
                if layer.__class__.__name__ == "LocalConv1D":
                    layer.forward = types.MethodType(local_forward_on_device, layer)
            model.fitting_model = torch.cuda.make_graphed_callables(
                model.fitting_model, (sample_x, sample_mask, sample_graph), allow_unused_input=True)
            actual = model.fitting_model(sample_x, sample_mask, sample_graph)
            actual_grads = torch.autograd.grad(actual.sum(), (sample_graph, *model.fitting_model.parameters()))
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            for actual_g, reference_g in zip(actual_grads, reference_grads):
                torch.testing.assert_close(actual_g, reference_g, rtol=1e-4, atol=1e-5)
            write_json(directory / "cuda_graph_verification.json", dict(forward_and_all_gradients_match=True))
            print(f"{label} CUDA neural graph forward/gradient equivalence PASS", flush=True)
    # Upstream does not initialize G/GT when group_policy=None; initialize the full graph here.
    model.G = torch.eye(cfg.n_nodes, device="cuda")
    model.GT = torch.nn.Parameter(torch.full((cfg.n_nodes, cfg.n_nodes), .5, device="cuda"))
    model.set_graph_optimizer(0)
    history = []
    started = time.monotonic()

    class Progress:
        def __init__(self, total):
            self.epoch = 0

        def set_postfix_str(self, text):
            pass

        def update(self, amount):
            self.epoch += amount
            with torch.no_grad():
                raw = (model.G @ model.GT).cpu().numpy()
                prob = (model.G @ model.GT.sigmoid()).cpu().numpy()
            row = dict(epoch=self.epoch, elapsed_seconds=time.monotonic()-started,
                       lambda_used=args.lambda_start * multiplier * model.lambda_gamma ** (self.epoch-1),
                       tau_used=1.0, tau_scheduled_but_unused=model.gumbel_tau / model.gumbel_tau_gamma)
            row.update({k: float(np.mean(v)) for k, v in log.values.items()})
            row.update({k: v for k, v in graph_stats(prob, .5).items() if k != "threshold"})
            history.append(row)
            log.values.clear()
            pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
            if self.epoch % 25 == 0 or self.epoch == args.epochs:
                np.savez_compressed(directory / f"graph_epoch{self.epoch:03d}.npz",
                                    graph_raw=raw, graph_probability=prob)
                write_json(directory / "progress.json", row)
                print(f"{label} epoch={self.epoch}/{args.epochs} "
                      f"loss={row.get('graph_discov/data_loss', float('nan')):.5f} "
                      f"density@.5={row['density']:.3f} elapsed={row['elapsed_seconds']:.1f}s", flush=True)

    old_batch = upstream.batch_generater
    old_progress = upstream.tqdm.tqdm
    upstream.batch_generater = vector_batches
    upstream.tqdm.tqdm = Progress
    try:
        x = data[:, :, None]
        raw_result = model.train(x, np.ones_like(x), x, true_cm=None)
    finally:
        upstream.batch_generater = old_batch
        upstream.tqdm.tqdm = old_progress
    with torch.no_grad():
        prob = (model.G @ model.GT.sigmoid()).cpu().numpy()
    if not np.isfinite(prob).all():
        raise RuntimeError(f"Nonfinite final graph: {label}")
    np.save(directory / "graph_raw.npy", raw_result)
    np.save(directory / "graph_probability.npy", prob)
    columns = metadata["columns"]
    pd.DataFrame(prob, index=columns, columns=columns).to_csv(directory / "graph_probability.csv", index_label="source")
    pd.DataFrame([graph_stats(prob, t) for t in args.thresholds]).to_csv(directory / "graph_statistics.csv", index=False)
    torch.save(dict(model=model.fitting_model.state_dict(), GT=model.GT.detach().cpu(),
                    G=model.G.detach().cpu(), config=OmegaConf.to_container(cfg)), directory / "checkpoint.pt")
    write_json(complete, dict(market=market, seed=seed, lambda_multiplier=multiplier,
                             epochs=args.epochs, elapsed_seconds=time.monotonic()-started))
    print(f"COMPLETE {label}", flush=True)


def summarize(output, thresholds):
    rows, similarities, groups = [], [], {}
    for path in sorted((output / "runs").glob("*/complete.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        p = np.load(path.parent / "graph_probability.npy")
        for t in thresholds:
            rows.append(dict(**item, **graph_stats(p, t)))
        groups.setdefault((item["market"], item["lambda_multiplier"]), []).append((item["seed"], p))
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "graph_statistics_all_runs.csv", index=False)
    metrics = ["density", "n_edges", "weight_mean", "weight_std", "normalized_weight_entropy",
               "target_in_density", "target_in_weight_mean"]
    summary = frame.groupby(["market", "lambda_multiplier", "threshold"])[metrics].agg(["count", "mean", "std", "min", "max"])
    summary.to_csv(output / "graph_statistics_summary.csv")
    for (market, mult), members in groups.items():
        folder = output / "stability" / f"{market}_lambda{mult:g}"
        folder.mkdir(parents=True, exist_ok=True)
        probs = np.stack([p for _, p in members])
        names = json.loads((output / "runs" / f"{market}_seed{members[0][0]}_lambda{mult:g}" / "metadata.json").read_text(encoding="utf-8"))["columns"]
        seed_sd = probs.std(0, ddof=1) if len(probs) > 1 else np.full_like(probs[0], np.nan)
        for name, matrix in (("probability_mean", probs.mean(0)), ("probability_std", seed_sd)):
            pd.DataFrame(matrix, index=names, columns=names).to_csv(folder / f"{name}.csv", index_label="source")
        off = ~np.eye(probs.shape[1], dtype=bool)
        for t in thresholds:
            freq = (probs > t).mean(0)
            np.fill_diagonal(freq, np.nan)
            pd.DataFrame(freq, index=names, columns=names).to_csv(folder / f"edge_selection_frequency_threshold{t:g}.csv", index_label="source")
            for (s1, p1), (s2, p2) in itertools.combinations(members, 2):
                a, b = p1[off] > t, p2[off] > t
                union = (a | b).sum()
                v1, v2 = p1[off], p2[off]
                corr = float(np.corrcoef(v1, v2)[0, 1]) if v1.std() > 0 and v2.std() > 0 else np.nan
                similarities.append(dict(market=market, lambda_multiplier=mult, threshold=t,
                                         seed_a=s1, seed_b=s2, jaccard=float((a & b).sum()/union) if union else np.nan,
                                         both_empty=bool(not union), weight_correlation=corr))
    pd.DataFrame(similarities).to_csv(output / "seed_pair_stability.csv", index=False)
    write_json(output / "status.json", dict(completed_runs=len(groups) and len(list((output / "runs").glob("*/complete.json"))),
                                            seed_standard_deviation_ddof=1,
                                            interpretation="Seed frequencies quantify optimization variability, not bootstrap confidence or causality."))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output", type=Path, default=ROOT / "result" / "train70_fullgraph")
    p.add_argument("--markets", nargs="+", choices=MARKETS, default=MARKETS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--lambda-multipliers", nargs="+", type=float, default=[1, .5, 2])
    p.add_argument("--lambda-start", type=float, default=.1)
    p.add_argument("--lambda-end", type=float, default=.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lookback", type=int, default=21)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-fraction", type=float, default=.7)
    p.add_argument("--match-rows", type=int, default=0)
    p.add_argument("--thresholds", nargs="+", type=float, default=[.3, .4, .5, .6, .7])
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--cuda-graphs", action="store_true", help="Capture neural forward/backward after numerical equivalence check")
    args = p.parse_args()
    if not 0 < args.train_fraction <= 1 or args.epochs < 1:
        p.error("Invalid train fraction or epoch count")
    if any(v <= 0 for v in [args.lambda_start, args.lambda_end, *args.lambda_multipliers]):
        p.error("Lambda values and multipliers must be positive")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(args.output, args.thresholds)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires the existing Windows CUDA PyTorch environment")
    torch.set_num_threads(4)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(planned_runs=len(args.markets)*len(args.seeds)*len(args.lambda_multipliers),
                    comparison_note="Seeds measure optimizer randomness; lambda sweep measures regularization sensitivity. Neither replaces time-series bootstrap.")
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Output directory has a different protocol; use a new output directory")
    write_json(path, protocol)
    try:
        # Baseline seed 0 for all four markets first, then other seeds and penalties.
        for mult, seed, market in itertools.product(args.lambda_multipliers, args.seeds, args.markets):
            train_one(args, market, seed, mult)
            summarize(args.output, args.thresholds)
    except BaseException as exc:
        write_json(args.output / "error.json", dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__":
    main()
