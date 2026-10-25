# Annual benchmark: EvoST on ten networks

The annual benchmark evaluates every yearly snapshot of the nine EvoXXLTraffic districts (PEMS03–PEMS08, PEMS10–PEMS12) and of TFNSW. Each year has its own STAEFormer checkpoint, trained by the benchmark on that year's training split. EvoST starts from that same-year checkpoint. Four streams share the checkpoint, the windows and the seeds:

| Stream | Identities | Per-sensor RLS calibrator |
|---|---|---|
| Frozen | checkpoint | no |
| Frozen+RLS | checkpoint | yes |
| Refresh | refreshed | no |
| EvoST | refreshed | yes |

These are the matched controls of the paper (Figure 9 of the submission; Table 9 and Figure 12 of the [extended version](../paper/EvoST_extended.pdf)). The EvoST rows of the result tables (Table 1; Tables 5–8 of the extended version) and the per-step figures (Figure 10 of the submission; Figures 6 and 11 of the extended version) come from the same runs.

## Runs

| Part | Networks | Network-years | Runs |
|---|---|---:|---:|
| Stage 1 | PEMS03 (25 years), PEMS04 (24), TFNSW (21) | 70 | 350 |
| Stage 2 | PEMS05 (21), PEMS06 (21), PEMS07 (25), PEMS08 (25), PEMS10 (20), PEMS11 (27), PEMS12 (24) | 163 | 815 |
| Total | ten networks | 233 | 1,165 |

Each network-year runs with evaluation seeds 51–55. They use the benchmark's checkpoints with seeds 42–46, identified by relative path and SHA-256 in [cases-stage1.json](../experiments/evost/annual/cases-stage1.json) and [cases-stage2.json](../experiments/evost/annual/cases-stage2.json). All 1,165 runs passed the acceptance gates below, and none produced a non-finite forecast.

## Protocol of one run

The runner is [e24_annual.py](../experiments/evost/annual/e24_annual.py); one run is one network, year and evaluation seed.

- **Inputs.** Validation and test inputs are rebuilt from raw windows and standardized with the statistics of the year's training split ([window_stats.py](../utils/window_stats.py)). Residual non-finite inputs are set to zero, and invalid targets are masked.
- **Identity refresh.** Only the identity table of the checkpoint (`12 × N × 80`) is fitted, on up to the latest 2,048 training windows. The final 512 window starts are held out for early stopping and for selecting the learning rate from `{0.001, 0.003, 0.01}`. Each candidate runs at most 16 epochs with patience 3. Fitting and holdout use every second window (every fourth above 1,600 sensors). On TFNSW in 2026, the training split has only 1,704 windows, which leaves 596 fitting and 256 holdout windows after sampling.
- **Calibration.** Each sensor's RLS calibrator is fitted on the training tail. Its ridge `{1, 100}` and forgetting factor `{1, 0.999, 0.995, 0.99}` are selected on up to the first 1,024 validation windows. The resulting states are discarded. New calibrators are then fitted on the training tail and run over the test portion only. They use each target once it is released 12 steps later and update every 64 windows. Validation labels therefore select calibration settings but do not reach the test stream.
- **Metrics.** The benchmark's masked MAE, RMSE and MAPE (in percent) are computed over all sensors of the test portion. They are reported for cumulative horizons 1–12, for single steps 1–12, and as Avg, the mean of the twelve cumulative values. A non-finite forecast would make a metric undefined ("strict" metrics); none occurred.
- **Acceptance gates.**
  - The checkpoint's path and SHA-256 equal the cases file.
  - The training statistics equal those of an independent re-evaluation of the same checkpoint.
  - The 75 uncalibrated Frozen test values lie within `|new − old| ≤ 1e-4 + 1e-5 |old|` of that re-evaluation.
  - For graphs with at most 3,000 sensors, the standalone backbone reproduces the original model's forecasts on stored reference windows.

### Zero-label fallback (PEMS06, 2005)

On PEMS06 in 2005, the fitting windows contain no valid target. For this year, the five seeds keep the checkpoint identities. The calibrators take the standard deviation of the year's training inputs as their shared scale and start from zero residual coefficients. Validation only selects the calibration settings, and the test state is initialized again and absorbs only labels released under the same rule. Refresh therefore coincides with Frozen, and EvoST with Frozen+RLS, in this year. The fallback is runner c. Stages 1 and 2 ran runner b, which differs only by this branch ([runner_b_to_c.diff](../experiments/evost/annual/runner_b_to_c.diff)). The fallback is triggered only when the fitting window has no valid target, a condition no other network-year meets.

## Aggregation

For each network, stream and seed, the yearly values are averaged with equal weight. The tables report the mean and the sample standard deviation (ddof = 1) over the five seeds. Ten-network averages weight networks equally. A relative reduction is `(reference − EvoST) / reference`, computed per network and then averaged.

## Results

Avg errors (MAE ± SD / RMSE / MAPE), mean over five seeds:

| Network | Years | Frozen | Frozen+RLS | Refresh | EvoST |
|---|---:|---|---|---|---|
| PEMS03 | 25 | 11.76 ± 0.15 / 19.72 / 23.75 | 11.01 ± 0.04 / 18.58 / 22.82 | 11.26 ± 0.09 / 19.24 / 21.66 | 10.92 ± 0.05 / 18.47 / 22.52 |
| PEMS04 | 24 | 18.57 ± 0.91 / 29.49 / 17.55 | 16.63 ± 0.15 / 27.17 / 15.53 | 18.18 ± 1.22 / 29.32 / 16.88 | 16.58 ± 0.21 / 27.14 / 15.52 |
| PEMS05 | 21 | 10.05 ± 0.04 / 15.79 / 23.81 | 9.53 ± 0.03 / 15.17 / 21.72 | 9.68 ± 0.05 / 15.49 / 21.90 | 9.48 ± 0.03 / 15.12 / 21.77 |
| PEMS06 | 21 | 12.70 ± 0.50 / 19.21 / 20.95 | 11.57 ± 0.20 / 17.96 / 19.28 | 12.35 ± 0.49 / 18.87 / 19.59 | 11.51 ± 0.20 / 17.87 / 19.18 |
| PEMS07 | 25 | 14.62 ± 0.31 / 25.98 / 21.39 | 13.56 ± 0.07 / 24.65 / 20.29 | 13.96 ± 0.11 / 25.47 / 19.30 | 13.45 ± 0.06 / 24.56 / 19.85 |
| PEMS08 | 25 | 12.35 ± 0.07 / 20.58 / 18.08 | 11.63 ± 0.04 / 19.66 / 17.24 | 11.88 ± 0.04 / 20.17 / 16.54 | 11.55 ± 0.04 / 19.57 / 16.89 |
| PEMS10 | 20 | 10.19 ± 0.06 / 16.94 / 28.49 | 9.82 ± 0.03 / 16.34 / 27.81 | 9.83 ± 0.04 / 16.78 / 26.64 | 9.76 ± 0.03 / 16.30 / 27.56 |
| PEMS11 | 27 | 16.04 ± 0.46 / 27.38 / 25.05 | 14.00 ± 0.08 / 25.89 / 22.81 | 14.87 ± 0.36 / 25.78 / 22.24 | 13.70 ± 0.09 / 25.24 / 22.27 |
| PEMS12 | 24 | 12.54 ± 0.14 / 22.42 / 24.80 | 11.85 ± 0.06 / 21.45 / 23.32 | 12.10 ± 0.09 / 22.08 / 21.87 | 11.76 ± 0.06 / 21.36 / 22.73 |
| TFNSW | 21 | 83.54 ± 1.68 / 176.72 / 59.00 | 65.54 ± 0.35 / 138.29 / 53.33 | 81.93 ± 1.37 / 176.64 / 57.26 | 65.16 ± 0.34 / 137.77 / 52.89 |

- Among the reported non-foundation methods of the benchmark tables, EvoST has the lowest Avg MAE, RMSE and MAPE in all 30 network-metric comparisons, with no ties.
- Relative to Frozen, averaged over the ten networks:
  - EvoST lowers Avg-MAE, Avg-RMSE and Avg-MAPE by 9.4%, 7.4% and 8.1%.
  - Frozen+RLS lowers them by 8.7%, 6.9% and 6.9%.
  - Refresh lowers them by 3.7%, 1.9% and 7.8%.

  With a separate checkpoint for each year, the calibrator gives most of the reduction.
- Relative to Frozen+RLS, EvoST lowers the three errors by 0.8%, 0.6% and 1.2%.
  - Its Avg-MAE is lower on every network, by 0.3% on PEMS04 to 2.1% on PEMS11.
  - The cross-year means are lower in all five paired seeds except on PEMS04 (three).
  - On PEMS04, its Avg-RMSE and Avg-MAPE are lower in only two of the five seeds; on PEMS05, its Avg-MAPE is 0.2% higher.
- Calibration lowers MAE and RMSE of both the frozen and the refreshed model on every network, and MAPE of the frozen model on every network. For the refreshed model, it raises MAPE on PEMS03, PEMS07, PEMS08, PEMS10, PEMS11 and PEMS12.
- Per step, EvoST has lower error than Frozen in all 360 combinations of network, metric and step, and lower error than Frozen+RLS in 343. The 17 exceptions are:
  - MAPE at steps 1–7 and 9–12 on PEMS05;
  - MAPE at steps 4–6 on PEMS04;
  - RMSE at the first step on PEMS04 and PEMS05;
  - MAPE at the last step on TFNSW.
- Compute: the 1,165 runs took 100.3 H100 GPU-hours, including candidate fits, calibration selection and diagnostic replays. The lowest candidate learning rate was selected in 1,100 of the 1,160 runs with an identity fit. The selected candidate ran all 16 epochs in 82 runs (47 of them on TFNSW).

## Reproduce the numbers

[results/evost/annual/](../results/evost/annual/README.md) holds one row per run and stream, the per-step errors and the summary. The script below recomputes the table above, the matched-control reductions and the per-step counts from the per-run rows and compares them with the summary:

```bash
python scripts/render_annual_results.py --check
```

It needs only the Python standard library.

## Run a network-year

A run needs an annual bundle: the `g30-stae` layout of the [bundle guide](evost_bundle_format.md#annual-benchmark-bundles) with only the same-year checkpoint. The bundles and checkpoints are not part of this repository.

```bash
python experiments/evost/annual/e24_annual.py \
  --bundle artifacts/annual/PEMS05_2005.npz --seed 51 \
  --cases experiments/evost/annual/cases-stage2.json --role formal \
  --out outputs/annual/PEMS05_2005/seed51 --device cuda:0
```

See [experiments/evost/annual/README.md](../experiments/evost/annual/README.md) for the outputs and options.
