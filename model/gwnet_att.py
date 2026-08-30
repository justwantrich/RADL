"""Dependency-aware spatiotemporal backbone for RADL.

Backbone: dilated causal convolution stack with gated activations and graph
diffusion, following Graph WaveNet (Wu et al., IJCAI 2019).  The latent
correlation layer follows the self-attention adjacency of StemGNN
(Cao et al., NeurIPS 2020).  Both must be cited in the paper.

RADL adds three things on top of that base:
  1. Reversible instance normalization anchored at the most recent observation,
     applied inside the network so the forward pass cannot desynchronize
     from the inverse mapping.
  2. A per-sample dependency matrix.  The upstream implementation averages the
     attention matrix over the mini-batch, which makes every prediction depend
     on the other windows that happen to share its batch -- at evaluation time
     that reads future observations.  Here each window builds its own graph.
  3. Optional regime-aligned variable gating with a sparsity penalty.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    """Graph diffusion step.  Support `a` is per-sample: (B, N_src, N_dst)."""

    def forward(self, x, a):
        return torch.einsum('ncvl,nvw->ncwl', (x, a)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]

        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class RevIN(nn.Module):
    """Reversible instance normalization over the lookback window.

    Follows Kim et al. (ICLR 2022), with one deviation: `anchor` selects the
    location statistic.  With anchor="last" the window is centred on the most
    recent observation, so a network output of zero reproduces the random-walk
    forecast exactly and the network only has to learn the departure from it.
    With anchor="mean" this reduces to standard RevIN.

    Tensors are (B, N, T, F): batch, variable/node, time, feature.
    """

    def __init__(self, num_nodes, anchor="last", affine=True, eps=1e-5):
        super(RevIN, self).__init__()
        if anchor not in ("last", "mean"):
            raise ValueError(f"anchor must be 'last' or 'mean', got {anchor!r}")
        self.anchor = anchor
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_nodes))
            self.beta = nn.Parameter(torch.zeros(num_nodes))

    def normalize(self, x):
        if self.anchor == "last":
            loc = x[:, :, -1:, :]
        else:
            loc = x.mean(dim=2, keepdim=True)
        # eps is added to the variance, so the effective scale floor is sqrt(eps).
        # Carbon markets have frozen-price stretches where the window variance is
        # exactly zero; the floor keeps those windows finite instead of exploding.
        scale = (x.var(dim=2, keepdim=True, unbiased=False) + self.eps).sqrt()
        x = (x - loc) / scale
        if self.affine:
            x = x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return x, loc, scale

    def denormalize(self, y, loc, scale):
        """Map (B, N, H) forecasts back to the input scale."""
        if self.affine:
            y = (y - self.beta.view(1, -1, 1)) / (self.gamma.view(1, -1, 1) + self.eps)
        return y * scale[:, :, 0, 0].unsqueeze(-1) + loc[:, :, 0, 0].unsqueeze(-1)


class STGating(nn.Module):
    """Regime-aligned variable emphasis: one soft gate per input variable.

    method="sigmoid" gives a deterministic relaxation; method="gumbel" uses the
    binary concrete relaxation with an annealed temperature, which pushes gates
    towards {0, 1} and yields a hard variable-selection story for the paper.
    """

    def __init__(self, num_nodes, method="sigmoid", init_logit=2.0):
        super(STGating, self).__init__()
        self.method = method
        self.tau = 1.0
        # Start near-open (sigmoid(2.0) = 0.88) so the backbone is not starved
        # before the gates have received any gradient.
        self.bottleneck_logits = nn.Parameter(torch.full((num_nodes,), float(init_logit)))

    def forward(self, x):
        """Return gate weights broadcastable over (B, N, T, F)."""
        logits = self.bottleneck_logits.view(1, -1, 1, 1)
        if self.method == "gumbel" and self.training:
            u = torch.rand(x.size(0), logits.size(1), 1, 1, device=x.device).clamp_(1e-6, 1 - 1e-6)
            logistic_noise = torch.log(u) - torch.log1p(-u)
            return torch.sigmoid((logits + logistic_noise) / self.tau)
        return torch.sigmoid(logits).expand(x.size(0), -1, -1, -1)

    def penalty(self):
        """L1 sparsity penalty on the expected number of retained variables."""
        return torch.sigmoid(self.bottleneck_logits).mean()


class gwnet(nn.Module):
    """RADL backbone.  Input (B, N, T, F) -> output (B, N, out_dim)."""

    def __init__(self, num_nodes, seq_length, dropout=0.3, in_dim=1, out_dim=1,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=3, layers=3,
                 att_dim=8, gating_method="none", use_static_adj=True,
                 use_revin=True, revin_anchor="last", revin_affine=True):

        super(gwnet, self).__init__()

        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.out_dim = out_dim
        self.att_dim = att_dim

        self.revin = RevIN(num_nodes, anchor=revin_anchor, affine=revin_affine) if use_revin else None
        self.st_gating = STGating(num_nodes, method=gating_method) if gating_method != "none" else None

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        # Data-dependent adjacency from the latent correlation layer, plus an
        # optional learned static adjacency (Graph WaveNet's adaptive matrix).
        # The static term is what stabilises the per-sample attention graph.
        self.weight_key = nn.Parameter(torch.zeros(size=(self.seq_length, att_dim)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.seq_length, att_dim)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.supports_len = 1

        if use_static_adj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes) * 0.1)
            self.supports_len += 1
        else:
            self.nodevec1 = None
            self.nodevec2 = None

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=(1, new_dilation)))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=(1, new_dilation)))
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=self.out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def latent_correlation_layer(self, input_data):
        """Per-sample dependency matrix, (B, N, N).

        attention[b, i, j] weights source variable i when aggregating into
        target variable j, so the softmax runs over the source axis (dim=1) to
        match the summation axis of nconv.
        """
        input_data = input_data.mean(dim=1)                      # (B, N, T)
        key = torch.matmul(input_data, self.weight_key)          # (B, N, att_dim)
        query = torch.matmul(input_data, self.weight_query)      # (B, N, att_dim)
        attention = torch.bmm(key, query.permute(0, 2, 1).contiguous())
        return F.softmax(attention / math.sqrt(self.att_dim), dim=1)

    def forward(self, my_input, return_att=False):
        # my_input: (B, N, T, F)
        loc = scale = None
        if self.revin is not None:
            my_input, loc, scale = self.revin.normalize(my_input)
        if self.st_gating is not None:
            my_input = my_input * self.st_gating(my_input)

        my_input = my_input.permute(0, 3, 1, 2)  # (B, N, T, F) -> (B, F, N, T)

        adp = self.latent_correlation_layer(my_input)
        new_supports = [adp]
        if self.nodevec1 is not None:
            # Column-stochastic so incoming weights sum to one over source nodes.
            static = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=0)
            new_supports.append(static.unsqueeze(0).expand(my_input.size(0), -1, -1))

        in_len = my_input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(my_input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = my_input
        x = self.start_conv(x)                        # (B, C, N, T)

        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x
            my_filter = self.filter_convs[i](residual)
            my_filter = torch.tanh(my_filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = my_filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip
            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # Keep only the final temporal position: it is the one whose receptive
        # field ends at the forecast origin.  Without this slice any lookback
        # longer than the receptive field emits T'>1 predictions per window,
        # which the downstream reshape silently misaligns.
        x = F.relu(skip[..., -1:])
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)                        # (B, out_dim, N, 1)
        out = x.squeeze(-1).transpose(1, 2)           # (B, N, out_dim)

        if self.revin is not None:
            out = self.revin.denormalize(out, loc, scale)

        if return_att:
            return out, adp
        return out


def _self_check():
    """Shape and invariant checks.  Run: python -m model.gwnet_att"""
    torch.manual_seed(0)
    B, N, F_in = 4, 16, 1

    # 1. Output shape holds for lookbacks both under and over the receptive field.
    for L in (10, 21, 30, 60):
        for H in (1, 3, 7, 21):
            m = gwnet(N, L, out_dim=H, blocks=3, layers=3).eval()
            y = m(torch.randn(B, N, L, F_in))
            assert y.shape == (B, N, H), f"L={L} H={H} -> {tuple(y.shape)}"

    # 2. compact backbones work too (previously emitted T'=15 and crashed the loss).
    m = gwnet(N, 21, out_dim=1, blocks=2, layers=2).eval()
    assert m(torch.randn(B, N, 21, F_in)).shape == (B, N, 1)

    # 3. No batch leakage: a sample's prediction must not depend on its batch mates.
    m = gwnet(N, 21, out_dim=1).eval()
    x = torch.randn(8, N, 21, F_in)
    with torch.no_grad():
        solo, full = m(x[:1]), m(x)
    assert torch.allclose(solo[0], full[0], atol=1e-6), \
        f"batch leakage: max diff {(solo[0] - full[0]).abs().max():.3e}"

    # 4. anchor='last' + zero output reproduces the random walk exactly.
    m = gwnet(N, 21, out_dim=1, revin_anchor="last", revin_affine=False).eval()
    nn.init.zeros_(m.end_conv_2.weight)
    nn.init.zeros_(m.end_conv_2.bias)
    x = torch.randn(B, N, 21, F_in)
    with torch.no_grad():
        y = m(x)
    assert torch.allclose(y, x[:, :, -1, :], atol=1e-5), "zero output must equal last observation"

    # 5. RevIN round-trips: normalize -> denormalize is the identity on the anchor.
    rv = RevIN(N, anchor="mean", affine=True)
    x = torch.randn(B, N, 21, F_in)
    xn, loc, sc = rv.normalize(x)
    back = rv.denormalize(xn[:, :, 0, 0].unsqueeze(-1), loc, sc)
    assert torch.allclose(back.squeeze(-1), x[:, :, 0, 0], atol=1e-4), "RevIN round-trip failed"

    # 6. Adjacency rows are normalised over the source axis that nconv sums over.
    m = gwnet(N, 21, out_dim=1).eval()
    with torch.no_grad():
        _, att = m(torch.randn(B, N, 21, F_in), True)
    assert torch.allclose(att.sum(dim=1), torch.ones(B, N), atol=1e-5), "support not column-stochastic"

    # 7. Gating produces per-variable weights in (0, 1).
    for meth in ("sigmoid", "gumbel"):
        m = gwnet(N, 21, out_dim=1, gating_method=meth)
        g = m.st_gating(torch.randn(B, N, 21, F_in))
        assert g.shape == (B, N, 1, 1) and (g > 0).all() and (g < 1).all()
        assert m(torch.randn(B, N, 21, F_in)).shape == (B, N, 1)

    print("gwnet_att self-check OK")


if __name__ == "__main__":
    _self_check()
