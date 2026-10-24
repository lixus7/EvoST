# Annual-benchmark runner

Source of the annual EvoST runs: EvoST and its matched controls on each year's STAEFormer checkpoint, for nine EvoXXLTraffic districts and TFNSW. The protocol, the results and the paper tables they feed are described in [docs/evost_annual.md](../../../docs/evost_annual.md).

| File | Role |
|---|---|
| `e24_annual.py` | Entry point: one run = one network, year and evaluation seed |
| `e3_confirm.py` | Per-sensor RLS calibrator, its selection on validation windows, and the side-output replay |
| `e15_maintain.py`, `e10_refresh.py` | Identity refresh with learning-rate selection on the holdout |
| `e5_stae.py`, `stae_model.py` | Standalone STAEFormer and the bundle reader |
| `g28_common.py` | Windows, masks, metrics and file helpers |
| `cases-stage1.json`, `cases-stage2.json` | The 350 and 815 runs: checkpoint path and SHA-256, sensor-ID policy, and reference values for the acceptance gates |
| `runner_b_to_c.diff` | Difference between runner b (1,160 runs) and runner c (the five PEMS06/2005 runs with the zero-label fallback) |
| `MANIFEST.json` | SHA-256 of every file and of its source; the only edits are a machine name in two comments of `e5_stae.py` and neutral cluster labels in the cases files |

The modules other than `e24_annual.py` and `e3_confirm.py` are the same as in the maintenance experiments (`../g37`, `../g42`). The files here are runner c; applying `runner_b_to_c.diff` in reverse gives runner b.

## Usage

```bash
python experiments/evost/annual/e24_annual.py \
  --bundle BUNDLE.npz --seed 51 \
  --cases experiments/evost/annual/cases-stage2.json --role formal \
  --out NEW_OUTPUT_DIR --device cuda:0
```

| Option | Meaning |
|---|---|
| `--bundle` | Annual bundle of one network-year ([format](../../../docs/evost_bundle_format.md#annual-benchmark-bundles)) |
| `--seed` | Evaluation seed, 51–55 (checkpoint seed = evaluation seed − 9) |
| `--cases` | Cases file; the acceptance gates compare the run with its entry |
| `--role formal` | Requires the cases file, complete splits, the annual mode and no test-only environment overrides |
| `--max-windows N` | Bounded debugging run; marked as a smoke test and never accepted |
| `--mode regression` | Reproduces a maintenance-experiment cell instead of the annual arms; not a result |

The output directory must be new. A run writes:

- `e24_record.json`: the main result per stream (Frozen, Frozen+RLS, Refresh, EvoST), the cumulative and per-step metrics, the selected calibration settings, the identity-fit record, the gates and the accepted/rejected status;
- `e24_sums.npz`: float64 error sums per sensor and horizon;
- `e3_selection.json`, `e3_metrics.jsonl`, `e3_sums.npz`: the calibration selection and the validation-plus-test replay kept as a side output;
- `completed.json`.

Large bundles and checkpoints are not part of this repository. The per-run results of all 1,165 runs are in [results/evost/annual/](../../../results/evost/annual/README.md).
