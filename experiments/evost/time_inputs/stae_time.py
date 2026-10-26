"""STAEFormer with and without the time-of-day / day-of-week inputs of the original (Liu et al., CIKM 2023), for g39.

The project's STAEFormer (stae_model.py, copied from the benchmark repository) sets tod_embedding_dim = dow_embedding_dim = 0
because its data loader carries no timestamps. The original concatenates, per input step, an embedding of the time of day
(steps_per_day x 24) and of the day of the week (7 x 24) in front of the adaptive embedding:
[input projection 24 | time of day 24 | day of week 24 | adaptive 80] -> model_dim 152. TimeStae builds either variant on the
project's layers (stae_model._STAE_SelfAttentionLayer); with time=False it is the project's model (model_dim 104), which is
checked against stae_model.StaeModel on the same weights.

Calendar (checked on the data, g39 PRESPEC.md): every yearly matrix of the benchmark starts on 1 January at 00:00 and the
benchmark uses its first 31 days at five-minute resolution; row r of a year is time of day r mod 288 and day
(weekday(1 January) + r // 288) mod 7, Monday = 0. The splits are consecutive: train rows [0, L_train), validation
[L_train, L_train + L_val), test after that.
"""
import datetime
import numpy as np
import torch
import torch.nn as nn
import stae_model as sm

STEPS_PER_DAY = 288
TOD_DIM = DOW_DIM = 24
INPUT_DIM, ADP_DIM, FF_DIM, HEADS, LAYERS = 24, 80, 256, 4, 3
STEPS = 12


def calendar(year, rows):
    """Time of day and day of week (Monday = 0) of absolute rows of a year's matrix."""
    rows = np.asarray(rows, np.int64)
    first = datetime.date(int(year), 1, 1).weekday()
    return rows % STEPS_PER_DAY, (first + rows // STEPS_PER_DAY) % 7


def window_time(year, first_row, starts):
    """tod, dow [len(starts), 12] of the input steps of windows that start at the given positions of a split beginning at first_row."""
    rows = first_row + np.asarray(starts, np.int64)[:, None] + np.arange(STEPS)[None]
    tod, dow = calendar(year, rows)
    return torch.as_tensor(tod), torch.as_tensor(dow)


class TimeBackbone(nn.Module):
    """STAEFORMER_Backbone of the project with the original's time-of-day and day-of-week embeddings (time=True) or without them."""

    def __init__(self, nodes, time=True, dropout=0., time_init_std=None):
        super().__init__()
        self.time = bool(time)
        self.in_steps, self.out_steps = STEPS, STEPS
        self.model_dim = INPUT_DIM + ADP_DIM + (TOD_DIM + DOW_DIM if self.time else 0)
        self.input_proj = nn.Linear(1, INPUT_DIM)
        if self.time:
            self.tod_embedding = nn.Embedding(STEPS_PER_DAY, TOD_DIM)                  # the original's initialisation: N(0, 1)
            self.dow_embedding = nn.Embedding(7, DOW_DIM)
            if time_init_std is not None:                                              # development probe only
                nn.init.normal_(self.tod_embedding.weight, std=float(time_init_std))
                nn.init.normal_(self.dow_embedding.weight, std=float(time_init_std))
        self.adaptive_embedding = nn.Parameter(torch.empty(STEPS, int(nodes), ADP_DIM))
        nn.init.xavier_uniform_(self.adaptive_embedding)
        self.attn_layers_t = nn.ModuleList([sm._STAE_SelfAttentionLayer(self.model_dim, FF_DIM, HEADS, dropout) for _ in range(LAYERS)])
        self.attn_layers_s = nn.ModuleList([sm._STAE_SelfAttentionLayer(self.model_dim, FF_DIM, HEADS, dropout) for _ in range(LAYERS)])
        self.output_proj = nn.Linear(STEPS * self.model_dim, STEPS)

    def forward(self, x, tod=None, dow=None, table=None):
        """x [B, N, 12] normalised inputs; tod / dow [B, 12] integer indices (time=True); table [12, N, 80] replaces the adaptive embedding."""
        B, N, T = x.shape
        z = self.input_proj(x.transpose(1, 2).unsqueeze(-1))                     # [B, T, N, 24]
        parts = [z]
        if self.time:
            parts.append(self.tod_embedding(tod)[:, :, None, :].expand(B, T, N, TOD_DIM))
            parts.append(self.dow_embedding(dow)[:, :, None, :].expand(B, T, N, DOW_DIM))
        adp = self.adaptive_embedding if table is None else table
        parts.append(adp.unsqueeze(0).expand(B, -1, -1, -1))
        z = torch.cat(parts, dim=-1)
        for attn in self.attn_layers_t:
            z = attn(z, dim=1)
        for attn in self.attn_layers_s:
            z = attn(z, dim=2)
        return self.output_proj(z.transpose(1, 2).reshape(B, N, T * self.model_dim))


class TimeStae(nn.Module):
    """STAEFORMER_Model of the project (backbone, then fc(GELU(backbone + x))) on TimeBackbone."""

    def __init__(self, nodes, time=True, dropout=0., time_init_std=None):
        super().__init__()
        self.backbone = TimeBackbone(nodes, time, dropout, time_init_std)
        self.fc = nn.Linear(STEPS, STEPS)
        self.activation = nn.GELU()

    def forward(self, x, tod=None, dow=None, table=None):
        return self.fc(self.activation(self.backbone(x, tod, dow, table) + x))


EMBEDDING = 'backbone.adaptive_embedding'


def shared_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k != EMBEDDING}


def check_against_project(seed=0, nodes=7, windows=3):
    """With time=False, TimeStae on the weights of a fresh project StaeModel must give the project's forecast."""
    torch.manual_seed(seed)
    mine = TimeStae(nodes, time=False).eval()
    ref = sm.StaeModel(dropout=0.).eval()
    state = {('backbone.' + k[len('backbone.'):] if k.startswith('backbone.') else k): v for k, v in mine.state_dict().items() if k != EMBEDDING}
    ref.load_shared(state)
    x = torch.randn(windows, nodes, STEPS, generator=torch.Generator().manual_seed(seed + 1))
    table = mine.backbone.adaptive_embedding.detach()
    with torch.no_grad():
        return float((mine(x, table=table) - ref(x, table)).abs().max())
