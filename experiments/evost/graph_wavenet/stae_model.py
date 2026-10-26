"""Standalone STAEFormer backbone for g30 (needs only torch).

The three classes below are copied verbatim from the project's `src/model/model.py`
(sha256 8d162f1d482aa6f5145e4eedd23dded7c7da9d7bd1cba34f65ec5ba714a71145, characters 76237-84419, block sha256 7a13fe53ea20982e1d5a8bab3c667fe789cd623bbc111a430744b5db59eb51c8); `StaeModel` mirrors
`STAEFORMER_Model.forward` in eval mode. The export job checks this copy against the original
classes on stored reference windows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _STAE_AttentionLayer(nn.Module):
    """Multi-head attention along the -2 dim of an n-D tensor.

    Mathematically equivalent to STAEformer/model/STAEformer.py::AttentionLayer
    but the QK^T attention matrix is never explicitly materialized — we route
    through `F.scaled_dot_product_attention` (PyTorch 2.0+), which uses
    FlashAttention on CUDA and brings memory from O(L^2) down to O(L).

    Why: spatial attention on xxltrafficdata has L = N (per-year node count,
    up to 3000+). The naive QK^T tensor of shape (heads, B, T, N, N) is
    O(N^2) and OOMs hard at N≥1000 even on 24GB GPUs — for N=3000 the
    attention matrix alone is ~200 GB. SDPA tiles the computation and keeps
    peak memory linear in N.
    """

    def __init__(self, model_dim, num_heads=4):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "STAEformer port requires torch.nn.functional.scaled_dot_product_attention "
                "(PyTorch >= 2.0). Upgrade torch or remove STAEFORMER from --method."
            )
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Inputs: (..., L, model_dim). SDPA expects (..., heads, L, head_dim);
        # broadcasting over any leading dims is fine.
        q = self.FC_Q(query)
        k = self.FC_K(key)
        v = self.FC_V(value)

        L_q = q.shape[-2]
        L_k = k.shape[-2]
        leading = q.shape[:-2]

        # Reshape last dim into (heads, head_dim), then move heads to dim -3
        # so SDPA attends over L_q/L_k (which sits at dim -2 after the move).
        q = q.reshape(*leading, L_q, self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.reshape(*leading, L_k, self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.reshape(*leading, L_k, self.num_heads, self.head_dim).transpose(-3, -2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        # (..., heads, L_q, head_dim) -> (..., L_q, heads*head_dim)
        out = out.transpose(-3, -2).reshape(*leading, L_q, self.model_dim)
        return self.out_proj(out)


class _STAE_SelfAttentionLayer(nn.Module):
    """Pre-LN-style transformer encoder block (LN-after-add variant, matching
    upstream). `dim` argument selects which axis to attend across (1=time,
    2=node).
    """

    def __init__(self, model_dim, feed_forward_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = _STAE_AttentionLayer(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        out = out.transpose(dim, -2)
        return out


class STAEFORMER_Backbone(nn.Module):
    """STAEformer (Liu et al., CIKM 2023), retrain-mode reimplementation.

    Architecture (faithful to upstream `model/STAEformer.py`):
      - Input projection: Linear(1 → input_embedding_dim).
      - Spatio-temporal adaptive embedding `E ∈ R^{T × N × adp_dim}` (the
        paper's key contribution), xavier-initialized.
      - Concatenate [input_emb, adp_emb] along feature axis →
        model_dim = input_embedding_dim + adp_embedding_dim.
      - `num_layers` stacked temporal-attention blocks (attend across T),
        then `num_layers` spatial-attention blocks (attend across N).
      - Mixed projection: reshape (B, N, T*model_dim) → Linear → (B, N, T_out).

    Differences vs upstream defaults (kept reasonable for the existing
    parameter budget across baselines; tunable via args.staeformer_*):
      - tod_embedding_dim=0, dow_embedding_dim=0 — the dataloader does not
        surface timestamps in this pipeline (same omission as STID).
      - spatial_embedding_dim=0 (matches upstream default).
      - input_embedding_dim=24, adaptive_embedding_dim=80 → model_dim=104.
      - num_heads=4, num_layers=3, feed_forward_dim=256, dropout=0.1.

    Streaming-mode caveat:
      `adaptive_embedding` and the mixed-projection output head are both
      sized to args.graph_size at __init__. N changes year-to-year on
      xxltrafficdata, so this model is retrain-only (strategy: retrain),
      matching GWN / STID / iTransformer / DLinear / STNORM.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        N = int(getattr(args, "graph_size", 0))
        if N <= 0:
            raise ValueError(
                "STAEFORMER_Backbone needs args.graph_size set at __init__ time. "
                "Use strategy=retrain so the model is freshly instantiated "
                "for each year's N."
            )
        in_steps = args.gcn["in_channel"]   # 12
        out_steps = args.gcn["out_channel"]  # 12
        input_dim = 1
        input_embedding_dim = int(getattr(args, "staeformer_input_emb", 24))
        adaptive_embedding_dim = int(getattr(args, "staeformer_adp_emb", 80))
        feed_forward_dim = int(getattr(args, "staeformer_ff_dim", 256))
        num_heads = int(getattr(args, "staeformer_num_heads", 4))
        num_layers = int(getattr(args, "staeformer_num_layers", 3))
        dropout = float(getattr(args, "dropout", 0.1))

        model_dim = input_embedding_dim + adaptive_embedding_dim
        if model_dim % num_heads != 0:
            raise ValueError(
                f"STAEFORMER: model_dim={model_dim} not divisible by "
                f"num_heads={num_heads}. Tune staeformer_input_emb / "
                f"staeformer_adp_emb / staeformer_num_heads."
            )

        self.in_steps = in_steps
        self.out_steps = out_steps
        self.model_dim = model_dim
        self.num_nodes = N

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.adaptive_embedding = nn.Parameter(
            torch.empty(in_steps, N, adaptive_embedding_dim)
        )
        nn.init.xavier_uniform_(self.adaptive_embedding)

        self.attn_layers_t = nn.ModuleList(
            [
                _STAE_SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                _STAE_SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Mixed projection (the upstream `use_mixed_proj=True` head).
        self.output_proj = nn.Linear(in_steps * model_dim, out_steps)

    def forward(self, x, adj):
        # x: [B, N, T_in] — adj ignored (STAEformer is non-graph).
        B, N, T = x.shape
        # Reshape to upstream format: (B, T_in, N, input_dim=1)
        z = x.transpose(1, 2).unsqueeze(-1)
        z = self.input_proj(z)  # (B, T_in, N, input_emb)
        adp = self.adaptive_embedding.unsqueeze(0).expand(B, -1, -1, -1)
        z = torch.cat([z, adp], dim=-1)  # (B, T_in, N, model_dim)

        for attn in self.attn_layers_t:
            z = attn(z, dim=1)  # attend across time
        for attn in self.attn_layers_s:
            z = attn(z, dim=2)  # attend across nodes

        # Mixed projection: (B, T_in, N, D) -> (B, N, T_in*D) -> Linear -> (B, N, T_out)
        out = z.transpose(1, 2).reshape(B, N, self.in_steps * self.model_dim)
        out = self.output_proj(out)
        return out  # (B, N, T_out)


_functional = F


class _FlatAttention:
    """`F` as seen by the classes above.

    Dimensions in front of (heads, L, E) are independent batch dimensions, so they are merged into one before the
    call. PyTorch's memory-efficient attention kernels accept only 4-D inputs; with the 5-D inputs of the spatial
    layers it falls back to materialising an N x N matrix per head, input step and window. The result is the same
    attention up to floating-point rounding.
    """

    @staticmethod
    def scaled_dot_product_attention(q, k, v, dropout_p=0.0):
        lead = q.shape[:-3]
        out = _functional.scaled_dot_product_attention(q.reshape(-1, *q.shape[-3:]), k.reshape(-1, *k.shape[-3:]),
                                                      v.reshape(-1, *v.shape[-3:]), dropout_p=dropout_p)
        return out.reshape(*lead, *out.shape[-3:])


F = _FlatAttention()


class _Args:
    """The attributes STAEFORMER_Backbone reads; values are the project's configuration."""

    def __init__(self, nodes, dropout=0.0):
        self.graph_size = int(nodes)
        self.gcn = dict(in_channel=12, out_channel=12, hidden_channel=64)
        self.dropout = float(dropout)


SHARED_PREFIXES = ('backbone.input_proj.', 'backbone.attn_layers_t.', 'backbone.attn_layers_s.', 'backbone.output_proj.', 'fc.')
EMBEDDING = 'backbone.adaptive_embedding'


class StaeModel(nn.Module):
    """STAEFORMER_Model without the per-sensor embedding table, which is passed to forward()."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.backbone = STAEFORMER_Backbone(_Args(1, dropout))
        self.fc = nn.Linear(12, 12)
        self.activation = nn.GELU()

    def load_shared(self, state):
        """state: {original parameter name: tensor} without the embedding table."""
        own = self.state_dict()
        expected = {k for k in own if k != EMBEDDING}
        assert set(state) == expected, (sorted(set(state) ^ expected)[:6],)
        own.update({k: torch.as_tensor(v) for k, v in state.items()})
        self.load_state_dict(own)
        return self.eval()

    @torch.no_grad()
    def forward(self, x, table):
        """x [B, N, 12] normalised inputs, table [12, N, 80] embedding of exactly these sensors -> [B, N, 12]."""
        assert table.shape[1] == x.shape[1] and not self.training
        self.backbone.adaptive_embedding = nn.Parameter(table, requires_grad=False)
        feat = self.backbone(x, None)
        return self.fc(self.activation(feat + x))


def mapped_table(embedding, index):
    """Embedding table of an old checkpoint in the sensor order of the target year.

    embedding [12, N_old, D]; index [N_new] holds each target sensor's position in the old year or -1.
    Sensors the checkpoint does not know receive the mean embedding over its sensors, per input step.
    """
    index = torch.as_tensor(index, dtype=torch.long)
    known = index >= 0
    table = embedding.mean(1, keepdim=True).expand(-1, len(index), -1).clone()
    table[:, known] = embedding[:, index[known]]
    return table, known
