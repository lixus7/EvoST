"""Standalone STID for g35 (needs only torch).

The two classes below are copied verbatim from the project's `src/model/model.py` as it is in the baseline repository
(sha256 8d162f1d482aa6f5145e4eedd23dded7c7da9d7bd1cba34f65ec5ba714a71145, characters 57929-62671, block sha256 0b11b7c5def9982a85f878966848d13469a7688c453e8d8eca76841d6a00baad).
They are used unchanged for training. `StidModel` mirrors `STID_Model.forward` in eval mode with the node-embedding table
passed in, so that the table can be replaced without touching the shared weights; `check_against_original` compares the
two on random inputs.
"""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_params_log(self):                                           # referenced by the copied class; logging only
    return sum(p.numel() for p in self.parameters())


class STID_Backbone(nn.Module):
    """STID (Shao et al. CIKM 2022), retrain-mode reimplementation.

    Faithful to the STID core idea: combine a time-series embedding with a
    *node-identity embedding* (the paper's key contribution — `E_node ∈
    R^{N×D}`, looked up per node), then process the concatenation with a
    stack of residual MLP blocks.

    Adaptation (vs the original paper):
      - The original additionally concatenates a time-of-day embedding
        (`R^{288×D}`) and a day-of-week embedding (`R^{7×D}`), looked up
        from per-sample timestamp metadata. This pipeline's dataloader
        (`SpatioTemporalDataset`) does not surface timestamps, so we omit
        these two embeddings. Documenting this as a deliberate limitation
        — re-adding them would require dataloader changes.
      - 3 residual MLP blocks (matches the original default).

    Streaming-mode caveat:
      `nn.Embedding(N, D_node)` is sized with `args.graph_size`, so this
      model is only safe under `strategy: retrain` (fresh model each year).
      Incremental loading across years would break because the embedding
      row count would change.
    """

    def __init__(self, args):
        super().__init__()
        H = args.gcn["hidden_channel"]                                  # 64
        D_node = getattr(args, "stid_node_dim", max(H // 2, 16))        # 32
        D_total = H + D_node
        N = int(getattr(args, "graph_size", 0))
        if N <= 0:
            raise ValueError(
                "STID_Backbone needs args.graph_size set at __init__ time. "
                "Use strategy=retrain so the model is freshly instantiated "
                "for each year's N."
            )

        self.time_embed = nn.Linear(args.gcn["in_channel"], H)
        self.node_embed = nn.Embedding(N, D_node)
        # Fixed-std init instead of xavier-by-fan-in: xavier on shape [N, D_node]
        # gives scale sqrt(6/(N+D)), which collapses to ~0.002 once N>~10k
        # (PEMS03 from 2018 onward), making the node-identity signal
        # numerically invisible vs the H=64 time embedding.
        nn.init.normal_(self.node_embed.weight, mean=0.0, std=0.02)
        # Buffer the node-id range so .to(device) follows the model.
        self.register_buffer("node_ids", torch.arange(N), persistent=False)

        num_blocks = getattr(args, "stid_blocks", 3)
        dropout = getattr(args, "dropout", 0.15)
        # Pre-norm residual MLP block. Without LayerNorm the residual stream
        # variance compounds across blocks and combined with lr=0.03 + MSE on
        # raw-scale traffic flow this stalls the optimizer (epoch-30 train
        # loss ~5k on some seed/year combos, never reaching the ~10 MAE basin
        # within the 100-epoch budget).
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(D_total),
                    nn.Linear(D_total, D_total),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(D_total, D_total),
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(D_total)
        self.out_proj = nn.Linear(D_total, args.gcn["out_channel"])

    def forward(self, x, adj):
        # x: [B, N, in_channel]
        B, N, _ = x.shape
        time_h = self.time_embed(x)                              # [B, N, H]
        ids = self.node_ids[:N]                                  # [N]
        node_h = self.node_embed(ids).unsqueeze(0).expand(B, -1, -1)  # [B, N, D_node]
        h = torch.cat([time_h, node_h], dim=-1)                  # [B, N, D_total]
        for block in self.blocks:
            h = h + block(h)
        h = self.final_norm(h)
        return self.out_proj(h)


class STID_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = STID_Backbone(args)
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


class Args:
    """The attributes the classes above read; values are the project's configuration."""

    def __init__(self, nodes, dropout=0.0):
        self.graph_size = int(nodes)
        self.gcn = dict(in_channel=12, out_channel=12, hidden_channel=64)
        self.dropout = float(dropout)
        self.y_len = 12


NODE_TABLE = 'backbone.node_embed.weight'


def original(nodes, dropout=0.0):
    """The project's STID_Model for a graph of `nodes` sensors."""
    return STID_Model(Args(nodes, dropout))


def forward_original(model, x):
    """STID_Model.forward on dense inputs x [B, N, 12] -> [B, N, 12] (the model reads only adj.shape[0])."""
    b, n, _ = x.shape
    out = model(types.SimpleNamespace(x=x.reshape(b * n, 12)), types.SimpleNamespace(shape=(n, n)))
    return out.reshape(b, n, 12)


class StidModel(nn.Module):
    """STID_Model without the node-embedding table, which is passed to the forward functions as [N, 32]."""

    def __init__(self, dropout=0.0):
        super().__init__()
        assert float(dropout) == 0., 'the project trains STID with dropout 0; an output dropout would have to be mirrored'
        self.backbone = STID_Backbone(Args(1, dropout))
        self.fc = nn.Linear(12, 12)
        self.activation = nn.GELU()

    def load_shared(self, state):
        """state: {original parameter name: tensor} without the node table."""
        own = self.state_dict()
        expected = {k for k in own if k != NODE_TABLE}
        assert set(state) == expected, (sorted(set(state) ^ expected)[:6],)
        own.update({k: torch.as_tensor(v) for k, v in state.items()})
        self.load_state_dict(own)
        return self.eval()

    def predict(self, x, table):
        """x [B, N, 12] normalised inputs, table [N, 32] -> [B, N, 12]; gradients flow to `table`."""
        bb = self.backbone
        assert table.shape[0] == x.shape[1] and not self.training
        h = torch.cat([bb.time_embed(x), table.unsqueeze(0).expand(x.shape[0], -1, -1)], dim=-1)
        for block in bb.blocks:
            h = h + block(h)
        feat = bb.out_proj(bb.final_norm(h))
        return self.fc(self.activation(feat + x))

    @torch.no_grad()
    def forward(self, x, table):
        return self.predict(x, table)


def split_state(state):
    """(shared weights, node table) of a STID_Model state dict."""
    return {k: v for k, v in state.items() if k != NODE_TABLE}, state[NODE_TABLE]


def check_against_original(state, dropout=0.0, windows=3, seed=0):
    """Largest absolute difference between StidModel and the project's STID_Model on random inputs."""
    shared, table = split_state({k: torch.as_tensor(v) for k, v in state.items()})
    n = table.shape[0]
    ref = original(n, dropout)
    ref.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    ref.eval()
    x = torch.randn(windows, n, 12, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        a, b = forward_original(ref, x), StidModel(dropout).load_shared(shared)(x, table.to(torch.float32))
    return float((a - b).abs().max())
