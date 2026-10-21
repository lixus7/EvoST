"""E11 (g34 B): identity refresh of a frozen STAEFormer at checkpoint ages of 5, 10 and 15 years, under the g29 protocol.

g33 refreshed the per-sensor embedding table of a five-year-old checkpoint. Here the same fit (e10_refresh, unchanged
settings) is applied to every older checkpoint a bundle holds: 5 years for target year 2015, 5 and 10 for 2020,
5, 10 and 15 for 2025. Arms (base forecast; each uncalibrated and with the calibrator):

  same        same-year checkpoint (reference)
  same_mean   same-year checkpoint, every sensor gets the mean embedding
  stale<k>o   checkpoint that is k years old, its own embeddings (mean embedding for sensors it does not know) and its
              own input statistics
  mean<k>     the same checkpoint, mean embedding for every sensor                      (do the shared weights age?)
  fit<k>      the same checkpoint, shared weights frozen, embedding table fitted on the fit windows of the target year
              starting from the stale<k>o table

At age 5 the arms stale5o, mean5 and fit5 are g33's stale5o, stale5o_mean and fit_emb.
"""
import json
from pathlib import Path
import sys
import torch
import e3_confirm as e3
import e10_refresh as e10


def arms_of(B, seed):
    B.seed = seed
    arm = lambda name, ckpt, variant: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant)
    arms = [arm('same', B.year, 'full'), arm('same_mean', B.year, 'mean')]
    for y in B.ckpt_years[1:]:
        k = B.year - y
        arms += [arm('stale%do' % k, y, 'full'), arm('mean%d' % k, y, 'mean'), arm('fit%d' % k, y, 'fit')]
    return arms


e3.arms_of = arms_of

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    B = e10.RefreshBundle.last
    if 'fit' in B.info:
        B.info['fit5'] = B.info.pop('fit')                              # e10 names the fit of the five-year-old checkpoint 'fit'
    B.info.update(sensors=B.n, known={str(y): int(B.known(y, torch.device('cpu')).sum()) for y in B.ckpt_years[1:]})
    (out / 'e10_info.json').write_text(json.dumps(B.info, indent=2) + '\n')
