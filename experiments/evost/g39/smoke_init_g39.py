"""g39, revised smoke (2026-10-15): what the training code computes before training has changed anything, for an exact
comparison between GPU types. Writes only the --out file.

  python smoke_init_g39.py --dataset PEMS05 --variant tm --year 2025 --head HEAD_DIR --bundles BUNDLE_DIR --conf stae_conf.json --out OUT.json [--seed 42] [--steps 20]

Why: the first smoke (virga_g39.py) compared the validation error after one epoch of training with nitro's, tolerance 5%. On
virga's H100 it differed by 1.0% without and by 9.5% with time inputs while data, environment and code hashes agreed:
training amplifies the rounding differences between GPU types. What an untrained model computes does not. This script runs
train_stae.train_one unchanged and stops it after its --steps-th update; on the way it records
  - checksums (float64 sums on the CPU) of the initial parameters and of the normalised train / validation inputs and their
    time of day / day of week: model construction, initialisation, data, normalisation and calendar, exact on any device;
  - the validation loss of the untrained model, by train_stae.validation_loss just before the first update;
  - the training loss of every update up to --steps (the first is the loss of the untrained model on the first batch).
The revised entry (virga_g39b.py) compares the checksums, the untrained validation loss and the first loss with nitro's; the
later losses are reported to show how fast two GPUs part.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as func
import stae_time as st
import stae_time_settings as settings
import train_stae as T

LOSSES = dict(mse='mse_loss', huber='smooth_l1_loss', mae='l1_loss')


class Stop(Exception):
    pass


def total(t):
    return float(t.detach().to('cpu', torch.float64).sum())


def main():
    ap = argparse.ArgumentParser()
    for k in ('--dataset', '--head', '--bundles', '--conf', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--variant', choices=('tm', 'nt'), required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=20)
    opt = ap.parse_args()
    conf = json.loads(Path(opt.conf).read_text())[opt.dataset]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seen = dict(model=None, splits=[], pieces=[], steps=[])

    class Model(st.TimeStae):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen['model'] = self
            seen['init'] = dict(parameters=sum(total(p) for p in self.parameters()), parameters_abs=sum(total(p.abs()) for p in self.parameters()),
                                time_embeddings=sum(total(p.abs()) for n, p in self.named_parameters() if 'tod_' in n or 'dow_' in n))

    class Split(T.Split):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen['splits'].append(self)

    original = getattr(func, LOSSES[conf['loss']])

    def loss(*a, **k):
        value = original(*a, **k)
        seen['pieces'].append((float(value), int(a[0].shape[0])))
        return value

    class AdamW(torch.optim.AdamW):
        def step(self, closure=None):
            if not seen['steps']:
                m = seen['model']
                m.eval()
                seen['init_validation_loss'] = T.validation_loss(m, seen['splits'][1], int(conf['batch_size']))
                m.train()
            n = sum(c for _, c in seen['pieces'])
            seen['steps'].append(sum(v * c for v, c in seen['pieces']) / n)
            seen['pieces'] = []
            out = super().step(closure)
            if len(seen['steps']) >= opt.steps:
                raise Stop
            return out

    st.TimeStae, T.Split, torch.optim.AdamW = Model, Split, AdamW
    setattr(func, LOSSES[conf['loss']], loss)
    try:
        T.train_one(opt.dataset, opt.variant, opt.year, opt.seed, conf, Path(opt.head).resolve(), Path(opt.bundles).resolve(), device, 1,
                    settings.TIME_INIT_STD)
        raise SystemExit('the epoch ended before %d updates' % opt.steps)
    except Stop:
        pass
    tr, va = seen['splits'][:2]
    data = {'%s_%s' % (name, part): total(getattr(s, part).float()) for name, s in (('train', tr), ('val', va)) for part in ('x', 'tod', 'dow')}
    out = dict(dataset=opt.dataset, variant=opt.variant, year=opt.year, seed=opt.seed, steps=len(seen['steps']), checksums=dict(seen['init'], **data),
               init_validation_loss=seen['init_validation_loss'], first_loss=seen['steps'][0], step_losses=seen['steps'], micro_batch=T.MICRO[0],
               time_init_std=settings.TIME_INIT_STD, torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
               matmul_precision=torch.get_float32_matmul_precision())
    Path(opt.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(variant=opt.variant, init_validation_loss=out['init_validation_loss'], first_loss=out['first_loss'], last_loss=out['step_losses'][-1],
                          micro_batch=out['micro_batch'])), flush=True)


if __name__ == '__main__':
    main()
