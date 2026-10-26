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

## rls_selection.csv and identity_lr.csv

The selection data behind Figure 6(a, b) of the paper.

- `rls_selection.csv`: for every run, the validation H12 MAE of the eight calibrator settings of the EvoST stream (`ridge` ∈ {1, 100}, `forget` ∈ {1, 0.999, 0.995, 0.99}); `chosen` marks the selected one. 1,165 runs × 8 rows.
- `identity_lr.csv`: for every run with an identity fit, the lowest holdout MAE reached with each candidate learning rate; `selected` marks the chosen rate. 1,160 fits × 3 rows. The five PEMS06/2005 runs have no fit (zero-label fallback).

## new_sensors.csv and new_sensors_years.csv

The values behind the new-sensor figure (Figure 4 of the submission, Figure 5 of the extended version): H12 errors on the sensors absent from the preceding year's graph, on PEMS07 and TFNSW.

- `new_sensors.csv`: one row per network, configuration and seed. `H12_MAE`, `H12_RMSE` and `H12_MAPE` are averaged over that seed's arrival years (`years`); the plotted bars are the means of the five seeds. Frozen, Frozen+RLS and EvoST come from the runs of this benchmark, pooled over the new sensors of each year from the per-sensor error sums. The other configurations come from the benchmark's new-sensor evaluation of its baselines, which uses each split's own input statistics.
- `new_sensors_years.csv`: sensors, new sensors and removed sensors of each year; `arrival_year` marks the 19 years per network that add sensors.

These are growth cohorts: the sensors are absent from the previous year's graph but are present when the current-year checkpoint is trained. They are distinct from the unseen-sensor arrival experiment. Most configurations cover 19 arrival years per network; PEMS07 STID covers 13–17 years per seed. ST-TTC is omitted because its archived new-sensor rows duplicate Retrain. The CSV contains the 21 plotted configurations per network. The per-year per-sensor error sums and the baselines' new-sensor evaluation script are not included, so this release verifies the exported figure values rather than rebuilding the cohort evaluation from raw predictions.

Recompute the figure's values and the numbers the paper quotes:

```bash
python scripts/check_paper_numbers.py sensitivity new-sensors
```
