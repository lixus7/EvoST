"""Read-only: the arrival experiments' shared calibrator scale s as computed by e3_confirm.run (all sensors, full
training tail) against the same statistic restricted to what the arrival scenario may see.

Usage: python experiments/evost/sensor_arrival/scale_check.py --bundles bundles/stae --out scale_check.json
The twelve main-panel maintenance bundles are read; results/evost/sensor_arrival/scale_check.json is the output for the paper's bundles.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g28_common as g

H, FIT = 64, 2048                                      # e18_handover.H, e3_confirm.FIT_WINDOWS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bundles', required=True, help='directory of the maintenance bundles <DATASET>_<year>.npz')
    ap.add_argument('--out', required=True)
    opt = ap.parse_args()
    out = []
    for ds in ('PEMS03', 'PEMS06', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12'):
        for year in (2020, 2025):
            z = np.load(Path(opt.bundles) / ('%s_%d.npz' % (ds, year)))
            raw = z['raw_train']
            known = np.asarray(z['known/y%d' % (year - 5)], bool)
            nw = g.n_windows(raw)
            fit_n = min(FIT, nw)
            first = nw - fit_n
            _, y = g.get_windows(raw, first, fit_n)
            y = g.clean_y(y)                                 # [fit_n, N, 12] float32
            valid = g.target_valid(y)
            s_all = float(y[valid].std())                    # exactly e3_confirm.run
            arrival = nw - H                                 # e17_coldstart.ColdBundle.arrival(H)
            idx = torch.arange(first, first + fit_n)
            before = idx < arrival
            new = torch.from_numpy(~known)
            legal = valid.clone()
            legal[before[:, None] & new[None, :]] = False    # new sensors only from their arrival on (the calibrator's own mask)
            s_legal = float(y[legal].std())
            kn = torch.from_numpy(known)
            s_est = float(y[valid & kn[None, :, None]].std())  # established sensors only
            row = dict(setting='%s/%d' % (ds, year), sensors=int(len(known)), new=int((~known).sum()),
                       s_all=s_all, s_legal=s_legal, s_established=s_est,
                       legal_over_all=s_legal / s_all, established_over_all=s_est / s_all,
                       valid_targets_all=int(valid.sum()), valid_targets_legal=int(legal.sum()))
            out.append(row)
            print(json.dumps(row), flush=True)
    r1 = np.array([r['legal_over_all'] for r in out]); r2 = np.array([r['established_over_all'] for r in out])
    print('legal/all: min %.4f max %.4f geo %.4f' % (r1.min(), r1.max(), np.exp(np.log(r1).mean())))
    print('established/all: min %.4f max %.4f geo %.4f' % (r2.min(), r2.max(), np.exp(np.log(r2).mean())))
    Path(opt.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
