"""E23 (g42): refreshing the per-sensor identity against EAC's node prompt, on a five-year-old frozen STAEFormer, g29 protocol.

EAC (Chen et al., ICLR 2025, "Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph
Forecasting"; the benchmark repository's src/model/model.py, class EAC_Model, configuration conf/PEMS/eac.json) freezes the
backbone and tunes a low-rank prompt per node that is added to the node's input window, x <- x + U V with U [N, rank],
V [rank, 12], rank 6, both initialised uniform(-0.1, 0.1); the model's residual path keeps the raw input; a node that arrives
gets a new row of U. Here that prompt is put on the five-year-old STAEFormer of g33 / g37 and given what g37 gave every other
parameter group: the same fit windows, holdout, epochs, patience, batches and order of windows (e15_maintain.MaintainBundle),
masked L1, Adam, its rate chosen among candidates on the fit's holdout, the model in evaluation mode.

  same, stale5o     references (g37)
  fit_emb           g33's identity refit: the embedding table only, rate 3e-3 (one of fit_emb_sel's candidates)
  fit_emb_sel       g37's: the embedding table only, rate chosen from e15_maintain.EMB_RATES
  fit_prompt        EAC's prompt only: table (the stale table; the mean row for sensors the old checkpoint does not know) and
                    shared weights frozen, U and V trained, rate chosen from PROMPT_RATES (EAC's own rate, 0.03, is the largest)
  fit_emb_prompt    table and prompt together: the table at the rate fit_emb_sel chose in this run, the prompt's rate from PROMPT_RATES
  fit_scratch       a new STAEFormer of the project's class (random initialisation, seeded) trained on the same fit windows,
                    every parameter and a new table, rate chosen from SCRATCH_RATES, at most SCRATCH_EPOCHS epochs with
                    patience SCRATCH_PATIENCE (the repository's recipe; a model trained from nothing needs more epochs than a
                    refit), masked L1 like every fit here: "why not train a new model on the new data?"

No validation or test value is read by a fit. Rules E1-E3, S1 in ../PRESPEC.md.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn as nn
import e3_confirm as e3
import e5_stae as e5
import e10_refresh as e10
import e15_maintain as e15
import g28_common as g
import stae_model as sm

RANK = 6                                                                 # EAC's configuration (conf/PEMS/eac.json)
PROMPT_RATES = (3e-3, 1e-2, 3e-2)
SCRATCH_RATES = (3e-4, 1e-3, 3e-3)
SCRATCH_EPOCHS = int(os.environ.get('G42_SCRATCH_EPOCHS', '100'))        # the override is for bounded smokes only
SCRATCH_PATIENCE = 5
e15.ARMS = {'fit_emb_sel': e15.ARMS['fit_emb_sel']}                       # g37's arm that this package keeps; the others are in g37's delivery
NEW = ('fit_prompt', 'fit_emb_prompt', 'fit_scratch')


def eac_prompt(n, generator, device):
    """U [n, RANK], V [RANK, 12], uniform(-0.1, 0.1) as EAC_Model initialises them."""
    U = (torch.rand(n, RANK, generator=generator, dtype=torch.float32) * .2 - .1).to(device)
    V = (torch.rand(RANK, 12, generator=generator, dtype=torch.float32) * .2 - .1).to(device)
    return nn.Parameter(U), nn.Parameter(V)


def run_model(model, table, prompt, x):
    """The project's STAEFormer forward with an optional EAC prompt on the backbone's input; the residual keeps the raw input."""
    model.backbone.adaptive_embedding = table if isinstance(table, nn.Parameter) else nn.Parameter(table, requires_grad=False)
    xin = x if prompt is None else x + prompt
    return model.fc(model.activation(model.backbone(xin, None) + x))


class EacBundle(e15.MaintainBundle):
    def __init__(self, path):
        super().__init__(path)
        self._new = {}

    def tune_new(self, device, name, rate, table_rate):
        """One fit of a new arm at one candidate rate; returns (model, table, prompt or None, record). Mirrors e15_maintain.tune."""
        began, raw = time.time(), self.z['raw_train']
        n_fit = min(e3.FIT_WINDOWS, g.n_windows(raw))
        first = g.n_windows(raw) - n_fit
        stride = e10.fit_stride(self.n)
        train = np.arange(first, first + n_fit - e10.FIT['holdout'], stride)
        hold = np.arange(first + n_fit - e10.FIT['holdout'], first + n_fit, stride)
        gen = torch.Generator().manual_seed(1000003 * self.seed + 17 * self.year + 7)
        start = e5.StaeBundle.table(self, self.old, device)
        if name == 'fit_scratch':
            torch.manual_seed(1000003 * self.seed + 17 * self.year + 11)
            model = sm.StaeModel(dropout=0.).to(device).eval()
            table = nn.Parameter(nn.init.xavier_uniform_(torch.empty(12, self.n, 80)).to(device))
            mean, std = self.norm[self.year]                                     # a model of this year: this year's train statistics
            prompt_U = prompt_V = None
            params = [dict(params=[table] + list(model.parameters()), lr=rate)]
            epochs, patience = SCRATCH_EPOCHS, SCRATCH_PATIENCE
        else:
            model = self.model(self.old, device)
            mean, std = self.norm[self.old]
            table = nn.Parameter(start.clone(), requires_grad=(name == 'fit_emb_prompt'))
            prompt_U, prompt_V = eac_prompt(self.n, gen, device)
            params = [dict(params=[prompt_U, prompt_V], lr=rate)]
            if name == 'fit_emb_prompt':
                params.append(dict(params=[table], lr=table_rate))
            epochs, patience = e10.FIT['epochs'], e10.FIT['patience']
        for p in model.parameters():
            p.requires_grad_(name == 'fit_scratch')
        trained = int(sum(p.numel() for group in params for p in group['params']))

        def batch(idx):
            x = np.stack([np.array(g.get_windows(raw, int(i), 1)[0][0], dtype=np.float32) for i in idx])
            y = g.clean_y(np.stack([np.array(g.get_windows(raw, int(i), 1)[1][0]) for i in idx])).to(device)
            return g.normalize_x(x, mean, std).to(device), y, g.target_valid(y)

        def prompt():
            return None if prompt_U is None else prompt_U @ prompt_V

        def predict(x):
            return run_model(model, table, prompt(), x)

        size = [e10.fit_batch(self.n)]

        def guarded(fn):
            while True:
                try:
                    return fn(size[0])
                except torch.cuda.OutOfMemoryError:
                    assert size[0] > 1, 'out of GPU memory with one window per step'
                    size[0] = max(1, size[0] // 2)
                    torch.cuda.empty_cache()

        def holdout():
            def run(b):
                total = count = 0.
                with torch.no_grad():
                    for s in range(0, len(hold), b):
                        x, y, valid = batch(hold[s:s + b])
                        total += float(((predict(x) - y).abs() * valid).sum())
                        count += float(valid.sum())
                return total / max(count, 1.)
            return guarded(run)

        def snapshot():
            return (table.detach().clone(), None if prompt_U is None else prompt().detach().clone(),
                    copy.deepcopy(model.state_dict()) if name == 'fit_scratch' else None)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            self.peak_seen = max(self.peak_seen, torch.cuda.max_memory_allocated() / 2 ** 30)
            torch.cuda.reset_peak_memory_stats()
        optim = torch.optim.Adam(params)
        order_gen = torch.Generator().manual_seed(7919 * self.seed + self.year)     # g33's order of windows
        history = [holdout()]
        best = (history[0], snapshot(), 0)
        epoch_seconds = []
        for epoch in range(epochs):
            t0 = time.time()
            order = train[torch.randperm(len(train), generator=order_gen).numpy()]
            s = 0
            while s < len(order):
                def step(b):
                    x, y, valid = batch(order[s:s + b])
                    with torch.enable_grad():
                        loss = ((predict(x) - y).abs() * valid).sum() / valid.sum().clamp_min(1)
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                    optim.step()
                    return b
                s += guarded(step)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            epoch_seconds.append(time.time() - t0)
            history.append(holdout())
            if history[-1] < best[0]:
                best = (history[-1], snapshot(), epoch + 1)
            elif epoch + 1 - best[2] >= patience:
                break
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else None
        self.peak_seen = max(self.peak_seen, peak or 0.)
        best_table, best_prompt, best_state = best[1]
        if best_state is not None:
            model.load_state_dict(best_state)
        for p in model.parameters():
            p.requires_grad_(False)
        if name != 'fit_scratch':
            model.backbone.adaptive_embedding = nn.Parameter(start.clone(), requires_grad=False)
        record = dict(arm=name, lr=rate, table_lr=table_rate if name == 'fit_emb_prompt' else (rate if name == 'fit_scratch' else None), rank=RANK if name != 'fit_scratch' else None,
                      max_epochs=epochs, patience=patience, stride=stride, train_windows=int(len(train)), holdout_windows=int(len(hold)), batch_windows=size[0],
                      holdout_mae=history, best_epoch=best[2], epochs_run=len(history) - 1, trained_values=trained,
                      mean_abs_change_table=float((best_table - start).abs().mean()) if name != 'fit_scratch' else None,
                      prompt_mean_abs=float(best_prompt.abs().mean()) if best_prompt is not None else None, seconds=time.time() - began,
                      seconds_per_epoch=float(np.mean(epoch_seconds)) if epoch_seconds else None, peak_gpu_gib=peak)
        return model, best_table.contiguous(), (best_prompt.contiguous() if best_prompt is not None else None), record

    def tuned_new(self, name, device):
        """(model, table, prompt) of a new arm: one fit per candidate rate, the lowest holdout error kept (the smaller rate on a tie)."""
        if name not in self._new:
            rates = SCRATCH_RATES if name == 'fit_scratch' else PROMPT_RATES
            table_rate = None
            if name == 'fit_emb_prompt':
                self.tuned('fit_emb_sel', device)
                table_rate = self.info['fit_emb_sel']['selected_lr']
            best, runs = None, {}
            for rate in rates:
                model, table, prompt, record = self.tune_new(device, name, rate, table_rate)
                runs['%g' % rate] = record
                if best is None or min(record['holdout_mae']) < min(best[3]['holdout_mae']):
                    best = (model, table, prompt, record)
            self._new[name] = best[:3]
            self.info[name] = dict(best[3], selected_lr=best[3]['lr'], candidates=runs, seconds=sum(r['seconds'] for r in runs.values()),
                                   peak_gpu_gib=max((r['peak_gpu_gib'] or 0.) for r in runs.values()) if device.type == 'cuda' else None)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        return self._new[name]

    def _forward(self, key, x_raw, device, sub):
        base, ckpt, norm, variant = key
        if variant not in NEW:
            return super()._forward(key, x_raw, device, sub)
        model, table, prompt = self.tuned_new(variant, device)
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        with torch.no_grad():
            out = torch.cat([run_model(model, table, prompt, x[i:i + sub]) for i in range(0, len(x), sub)])
        if variant != 'fit_scratch':                                             # the shared old model goes back to its own table
            model.backbone.adaptive_embedding = nn.Parameter(e5.StaeBundle.table(self, self.old, device).clone(), requires_grad=False)
        return out.to(torch.float32).to(self.cache_device(device))


def arms_of(B, seed):
    B.seed = seed
    arm = lambda name, ckpt, variant: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant)
    arms = [arm('same', B.year, 'full'), arm('stale5o', B.old, 'full'), arm('fit_emb', B.old, 'fit_emb'), arm('fit_emb_sel', B.old, 'fit_emb_sel'),
            arm('fit_prompt', B.old, 'fit_prompt'), arm('fit_emb_prompt', B.old, 'fit_emb_prompt'), arm('fit_scratch', B.year, 'fit_scratch')]
    return arms


e3.Bundle = EacBundle
e3.arms_of = arms_of
e3.check_reference = e15.check_reference

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    B = e10.RefreshBundle.last
    peak = max(B.peak_seen, torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None
    B.info.update(sensors=B.n, peak_gpu_gib_run=peak, rank=RANK, prompt_values=int(B.n * RANK + RANK * 12),
                  table_values=int(np.prod(B.z['emb/%d/y%d' % (B.seed, B.old)].shape)),
                  shared_values=int(sum(p.numel() for name, p in B.model(B.old, torch.device('cpu')).named_parameters() if name != 'backbone.adaptive_embedding')),
                  rates=dict(emb=e15.EMB_RATES, prompt=PROMPT_RATES, scratch=SCRATCH_RATES), scratch_epochs=SCRATCH_EPOCHS, scratch_patience=SCRATCH_PATIENCE)
    (out / 'e10_info.json').write_text(json.dumps(B.info, indent=2) + '\n')
