# Annual-benchmark results

All 1,165 accepted runs of the annual benchmark (ten networks; [protocol](../../../docs/evost_annual.md)). The files are derived from the formal scoring of the run records; host names and file-system paths are not included. `summary.json` records the SHA-256 of the two scoring outputs it was derived from.

Recompute every summary value from the per-run rows:

```bash
python scripts/render_annual_results.py --check
```

## cells.csv

One row per run and stream: 1,165 runs × 4 streams = 4,660 rows.

| Column | Meaning |
|---|---|
| `stage` | `stage1` (PEMS03, PEMS04, TFNSW) or `stage2` (the other seven districts) |
| `source` | `stage` for runner b; `repair` for the five PEMS06/2005 runs of runner c (zero-label fallback) |
| `dataset`, `year`, `seed`, `nodes` | Network, year, evaluation seed (checkpoint seed = seed − 9) and number of sensors |
| `stream` | `frozen`, `frozen+RLS`, `refit` (Refresh in the paper) or `EvoST` |
| `Avg_MAE`, `Avg_RMSE`, `Avg_MAPE` | Mean of the twelve cumulative-horizon errors; MAPE in percent |
| `H3_*`, `H6_*`, `H12_*` | Cumulative errors over steps 1–3, 1–6 and 1–12 |
| `identity_lr`, `identity_best_epoch`, `identity_epochs_run` | Selected identity learning rate; the number of completed epochs at its lowest holdout error (0 = the checkpoint's own identities were best); the epochs it ran. `skipped (no valid fitting label)` for the fallback runs |
| `rls_frozen`, `rls_evost` | Selected ridge and forgetting factor of the calibrator on the frozen and on the refreshed checkpoint |
| `gpu`, `seconds` | GPU type and wall time of the whole run |

All metrics are the strict values; no run had a non-finite forecast.

## per_step.csv

For each network, stream (`method`: `evost`, `evost_frozen`, `evost_frozen_rls`, `evost_refit`) and seed, errors averaged over the network's years (`year_count`) with equal weight. `horizon` is `step1`–`step12` for single-step errors, `1`–`12` for cumulative horizons, or `Avg`. These are the values behind the per-step figures (Figure 11 of the submission; Figures 7 and 12 of the extended version).

## summary.json

| Key | Content |
|---|---|
| `table` | For each `network/stream`: years, seeds, and the mean and sample SD (ddof = 1) over seeds of `Avg_*`, `3_*`, `6_*` and `12_*` |
| `matched_controls` | Per-network and ten-network relative reductions (positive = lower error) with the number of seeds in which the cross-year mean is lower |
| `per_step` | Counts of the 360 network-metric-step combinations in which EvoST is lower than Frozen or Frozen+RLS, and the mean reductions |
| `protocol` | Learning-rate selection counts, skipped fits and GPU-hours of both stages |
| `ranking` | Ranking of EvoST among the reported non-foundation methods in the Avg tables (30 comparisons) and the per-horizon tables (120) |
