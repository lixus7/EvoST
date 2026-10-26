"""Standalone Graph WaveNet for g36 (needs only torch).

The classes below are copied verbatim from the project's code as it is in the baseline repository:
  BatchGCNConv              src/model/gcn_conv.py (sha256 995c7ef49120f330d7611dd60e46bcb0e48e55eb22eb6b922a0e222d9acfe549, characters 88-1321, block sha256 5ba07934255e088de14f3ba3f11b10b2636d722b715a58d15af1f72442f023be)
  GWN_Backbone, GWN_Model   src/model/model.py    (sha256 8d162f1d482aa6f5145e4eedd23dded7c7da9d7bd1cba34f65ec5ba714a71145, characters 51960-57929, block sha256 42ba53a22cfed3580325d30e96678d77f3f25e18ba7c9b81b8966ecf3ae16274)
They are used unchanged for training. `GwnModel` runs the same backbone code with the two per-sensor embedding matrices
E1, E2 (N x 10 each; they generate the adaptive adjacency softmax(relu(E1 E2^T)), which the backbone averages with the
static adjacency) passed in as one table [N, 20], so that the table can be replaced without touching the shared weights.
`check_against_original` compares the two on random inputs.
"""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_params_log(self):                                           # referenced by the copied class; logging only
    return sum(p.numel() for p in self.parameters())


class BatchGCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super(BatchGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()



    def forward(self, x, adj):
        # x: [bs, N, in_features], adj: [N, N]
        input_x = torch.matmul(adj, x)              # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        output = self.weight_neigh(input_x)             # [bs, N, in_features] * [in_features, out_features] = [bs, N, out_features]
        if self.weight_self is not None:
            output += self.weight_self(x)               # [bs, N, out_features]
        return output


class GWN_Backbone(nn.Module):
    """Graph WaveNet (Wu et al. IJCAI 2019), retrain-mode reimplementation.

    Faithful to the GWN architecture:
      - 1x1 start conv embeds the raw time history into residual channels.
      - Stacked ST-blocks. Each block applies a *gated dilated TCN on the
        time axis* (filter = tanh, gate = sigmoid, multiplied), then a GCN
        with an adaptive adjacency `softmax(relu(E1 @ E2.T))` blended with
        the input static adjacency, then a residual + BN.
      - A skip path from each block accumulates into the final output, which
        goes through two 1x1 convs to produce the prediction tensor.

    Simplifications vs the original (kept within the existing parameter
    budget of ~10-30k for fair baseline comparison):
      - 2 ST-blocks with dilations [1, 2] instead of 4-8 blocks with
        dilations [1,2,1,2,1,2,1,2].
      - residual_channels = hidden_channel // 2.

    Streaming-mode caveat:
      `E1, E2 ∈ R^{args.graph_size × adp_dim}` and N may change year-to-year.
      This model is therefore only safe under `strategy: retrain` (fresh
      model each period). The existing trainer guarantees this when
      `args.init == False` (see default_trainer.py:117).
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        T_out = args.gcn["out_channel"]
        residual_channels = max(args.gcn["hidden_channel"] // 2, 16)
        skip_channels = args.gcn["hidden_channel"]
        kernel_size = 2
        dilations = [1, 2]
        adp_dim = 10
        N = int(getattr(args, "graph_size", 0))
        if N <= 0:
            raise ValueError(
                "GWN_Backbone needs args.graph_size set at __init__ time "
                "(per-year node count). The main.py loop sets it before "
                "instantiating the model."
            )

        # Learnable adaptive-adjacency embeddings (the GWN signature feature).
        self.E1 = nn.Parameter(torch.randn(N, adp_dim) * 0.01)
        self.E2 = nn.Parameter(torch.randn(N, adp_dim) * 0.01)

        # [B, 1, N, T] -> [B, C_r, N, T]
        self.start_conv = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.bn = nn.ModuleList()

        receptive_field = 1
        for d in dilations:
            self.filter_convs.append(
                nn.Conv2d(residual_channels, residual_channels,
                          kernel_size=(1, kernel_size), dilation=(1, d))
            )
            self.gate_convs.append(
                nn.Conv2d(residual_channels, residual_channels,
                          kernel_size=(1, kernel_size), dilation=(1, d))
            )
            self.skip_convs.append(
                nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1))
            )
            self.gcn_layers.append(
                BatchGCNConv(residual_channels, residual_channels, bias=True, gcn=False)
            )
            self.bn.append(nn.BatchNorm2d(residual_channels))
            receptive_field += (kernel_size - 1) * d

        self.end_conv1 = nn.Conv2d(skip_channels, skip_channels, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(skip_channels, T_out, kernel_size=(1, 1))
        self.receptive_field = receptive_field

    def forward(self, x, adj):
        # x: [B, N, T]  T = args.gcn["in_channel"] = 12
        T = x.shape[-1]
        x = x.unsqueeze(1)  # [B, 1, N, T]
        if T < self.receptive_field:
            x = F.pad(x, (self.receptive_field - T, 0))

        x = self.start_conv(x)  # [B, C_r, N, T_pad]

        # Adaptive adjacency, blended with the input static adj.
        adp = F.softmax(F.relu(self.E1 @ self.E2.T), dim=1)
        adj_eff = 0.5 * (adj + adp)

        skip = None
        for i in range(len(self.filter_convs)):
            residual = x
            f = torch.tanh(self.filter_convs[i](x))
            g = torch.sigmoid(self.gate_convs[i](x))
            x = f * g  # [B, C_r, N, T_block]

            # Skip path: project to skip_channels, keep last time step.
            s = self.skip_convs[i](x)[..., -1:]  # [B, skip, N, 1]
            skip = s if skip is None else skip + s

            # GCN over space, batched across time steps:
            #   [B, C_r, N, T_block] -> [B*T_block, N, C_r] -> gcn -> reshape back
            B_, C_, N_, Tn = x.shape
            x_g = x.permute(0, 3, 2, 1).reshape(B_ * Tn, N_, C_)
            x_g = self.gcn_layers[i](x_g, adj_eff)
            x_g = x_g.reshape(B_, Tn, N_, C_).permute(0, 3, 2, 1)

            # Residual (trim earlier time steps so shapes match).
            x = x_g + residual[..., -Tn:]
            x = self.bn[i](x)

        # End convs: [B, skip, N, 1] -> [B, T_out, N, 1] -> [B, N, T_out]
        x = F.relu(skip)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)
        return x.squeeze(-1).permute(0, 2, 1)


class GWN_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = GWN_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x


ADP = 10                                                               # width of E1 and of E2
E1, E2 = 'backbone.E1', 'backbone.E2'


class Args:
    """The attributes the classes above read; values are the project's configuration."""

    def __init__(self, nodes, dropout=0.0):
        self.graph_size = int(nodes)
        self.gcn = dict(in_channel=12, out_channel=12, hidden_channel=64)
        self.dropout = float(dropout)
        self.y_len = 12


def original(nodes, dropout=0.0):
    """The project's GWN_Model for a graph of `nodes` sensors."""
    return GWN_Model(Args(nodes, dropout))


def forward_original(model, x, adj):
    """GWN_Model.forward on dense inputs x [B, N, 12] and the row-normalised static adjacency [N, N] -> [B, N, 12]."""
    b, n, _ = x.shape
    return model(types.SimpleNamespace(x=x.reshape(b * n, 12)), adj).reshape(b, n, 12)


class GwnModel(nn.Module):
    """GWN_Model without E1 and E2, which are passed to the forward functions as one table [N, 20] = [E1 | E2]."""

    def __init__(self, dropout=0.0):
        super().__init__()
        assert float(dropout) == 0., 'the project trains Graph WaveNet with dropout 0; an output dropout would have to be mirrored'
        self.backbone = GWN_Backbone(Args(1, dropout))
        del self.backbone.E1, self.backbone.E2                         # set per call, as plain tensors
        self.backbone.E1 = self.backbone.E2 = None
        self.fc = nn.Linear(12, 12)
        self.activation = nn.GELU()

    def load_shared(self, state):
        """state: {original parameter or buffer name: tensor} without E1 and E2."""
        own = self.state_dict()
        assert set(state) == set(own), (sorted(set(state) ^ set(own))[:6],)
        self.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
        return self.eval()

    def predict(self, x, table, adj):
        """x [B, N, 12] normalised inputs, table [N, 20], adj [N, N] -> [B, N, 12]; gradients flow to `table`."""
        bb = self.backbone
        assert table.shape == (x.shape[1], 2 * ADP) and adj.shape == (x.shape[1], x.shape[1]) and not self.training
        bb.E1, bb.E2 = table[:, :ADP].contiguous(), table[:, ADP:].contiguous()   # the memory layout of the original parameters
        try:
            feat = bb(x, adj)                                          # the copied forward, unchanged
        finally:
            bb.E1 = bb.E2 = None
        return self.fc(self.activation(feat + x))

    @torch.no_grad()
    def forward(self, x, table, adj):
        return self.predict(x, table, adj)


def split_state(state):
    """(shared weights and buffers, table [N, 20]) of a GWN_Model state dict."""
    table = torch.cat([torch.as_tensor(state[E1]), torch.as_tensor(state[E2])], 1)
    return {k: v for k, v in state.items() if k not in (E1, E2)}, table


def check_against_original(state, dropout=0.0, windows=2, seed=0):
    """Largest absolute difference between GwnModel and the project's GWN_Model (eval mode) on random inputs."""
    state = {k: torch.as_tensor(v) for k, v in state.items()}
    shared, table = split_state(state)
    n = table.shape[0]
    ref = original(n, dropout)
    ref.load_state_dict(state)
    ref.eval()
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(windows, n, 12, generator=gen)
    adj = torch.rand(n, n, generator=gen)
    adj = adj / adj.sum(1, keepdim=True)
    with torch.no_grad():
        a, b = forward_original(ref, x, adj), GwnModel(dropout).load_shared(shared)(x, table.to(torch.float32), adj)
    return float((a - b).abs().max())
