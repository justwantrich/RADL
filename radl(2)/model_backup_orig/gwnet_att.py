import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, a):
        x = torch.einsum('ncvl,vw->ncwl', (x, a))
        return x.contiguous()


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
        c_in = (order*support_len+1)*c_in
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


class gwnet(nn.Module):
    """Dependency-aware spatiotemporal backbone with adaptive structure inside RADL."""

    def __init__(self, num_nodes, seq_length, dropout=0.3, in_dim=1, out_dim=1, residual_channels=32,
                 dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=3, layers=3,
                 gating_method="none"):

        super(gwnet, self).__init__()

        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.out_dim = out_dim

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        receptive_field = 1
        self.supports_len = 0
        self.supports = []
        self.out_dim_2_em_dim = {
            1: 2,
            2: 1,
            3: 6,
            4: 6,
            5: 6
        }
        self.weight_key = nn.Parameter(torch.zeros(size=(self.seq_length, self.out_dim_2_em_dim[self.out_dim])))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.seq_length, self.out_dim_2_em_dim[self.out_dim])))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=(new_dilation,)))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=(new_dilation,)))
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
        input_data = input_data.squeeze(1)
        key = torch.matmul(input_data, self.weight_key)
        query = torch.matmul(input_data, self.weight_query)
        attention = torch.bmm(key, query.permute(0, 2, 1).contiguous())
        attention = F.softmax(attention/math.sqrt(self.out_dim_2_em_dim[self.out_dim]), dim=1)
        attention = torch.mean(attention, dim=0)
        return attention

    def forward(self, my_input, return_att):
        my_input = my_input.permute(0, 3, 1, 2)  # (B, N, T, F) -> (B, F, N, T)

        adp = self.latent_correlation_layer(my_input)
        new_supports = [adp]

        in_len = my_input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(my_input, (self.receptive_field-in_len, 0, 0, 0))
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
                skip = skip[:, :, :,  -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip
            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x.transpose(1, 3)
        if return_att:
            return x, adp
        return x
