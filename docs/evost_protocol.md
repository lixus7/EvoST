# EvoST implementation and evaluation protocol

EvoST maintains a forecaster by refreshing its sensor identities while freezing shared spatio-temporal weights. A per-sensor recursive least-squares (RLS) calibrator adjusts the forecasts when labels become available. New sensors borrow identities from established sensors with similar recent observations. A scheduled refit updates all identities and reinitializes all calibrators after sufficient feedback has accumulated.

![EvoST method](assets/evost_method.png)

[Vector method figure](assets/evost_method.pdf).

## Two evaluation settings

| Setting | Checkpoint and observations | Main question |
|---|---|---|
| Checkpoint maintenance | A model trained five years earlier; recent target-year observations are available for all evaluated sensors | Which parameter groups need updating? |
| Sensor arrival | Incoming sensors have 64 post-arrival fitting windows before the stream; their earlier records are masked in sensor-specific fits | How should new identities be initialized and jointly maintained? |

The maintenance setting includes sensors absent from the old checkpoint. Their recent observations are available during the initial fit. The arrival setting imposes a separate observation boundary; the two settings should retain distinct labels in result tables.

## Identity fitting

The identity blocks are `12 × N × 80` in STAEFormer, `N × 32` in STID, and two `N × 10` tables in the Graph WaveNet implementation. The last two tables define the learned adjacency through row-wise `softmax(ReLU(E1 @ E2.T))`. The static adjacency is retained.

The main STAEFormer comparison uses the latest 2,048 training windows and reserves the final quarter for early stopping. Identity learning rates are selected from `{0.001, 0.003, 0.01}`. Fits run for at most 16 epochs with patience 3. Shared weights, shared temporal parameters, output layers, and normalization statistics remain frozen during identity-only maintenance; the model runs in evaluation mode. The full-parameter, shared-weight, output-head, and prompt controls have their own documented learning-rate grids in the source.

The EAC-style prompt control uses a rank-six node prompt on the same frozen STAEFormer and recent observations. It tests the prompt mechanism under the maintenance protocol. EAC itself is a baseline of the annual benchmark ([tables](../tables/README.md), [baseline guide](baselines.md)).

## Arrival and joint refitting

The initial identity is the mean of five behaviorally similar established-sensor identities. Matching uses finite raw observations, including zeros. A donor must have observations at 20% of the available times and overlap with the incoming sensor at 20% of those times. If fewer than five donors qualify, the complete five-donor set is sampled from established sensors with the run's fixed seed.

In the code, the final arrival configuration is the branch `a_prof0_r` of [e21_handover2.py](../experiments/evost/g41/e21_handover2.py). At zero-based stream index 1,024, it refits every identity and restarts every sensor's RLS calibrator. With a 12-step label delay, stream windows 0 through 1,012 are available. These 1,013 windows and the last 11 eligible post-arrival prestream windows form the 1,024-window refit pool. The event follows that step's RLS update and precedes its forecast. Arrival fits use learning rate 0.003.

## Delayed feedback and calibration

Input length and forecast horizon are both 12. PeMS observations are five minutes apart, so the forecast window covers the next hour. RLS updates every 64 windows and uses a forecast window only after all 12 labels have been released. A node-window enters calibration only when all targets are finite and nonzero. Identity fitting masks invalid targets element by element.

Each sensor has a 29-dimensional feature vector: the scaled base forecast, normalized input history, its last value, mean, population standard deviation, slope, and a constant. RLS starts from an exact ridge fit, uses float64 recursion, and symmetrizes the inverse covariance after each update. Ridge `{1, 100}` and forgetting factor `{1, 0.999, 0.995, 0.99}` are selected separately for each branch on the first 1,024 validation windows. The calibrator is then initialized again and validation and test are replayed chronologically. Reported results use the test portion.

The shared output scale is the standard deviation of valid targets in the initial training tail. In the archived arrival implementation, this statistic is computed before sensor-specific arrival masking and stays fixed. Donor observations and sensor-specific fitting records follow the arrival boundary. This scale convention is retained in the released code and results.

## Metrics and aggregation

The primary panel is PeMS03, PeMS06, PeMS08, PeMS10, PeMS11, and PeMS12 in 2020 and 2025: twelve region-year settings, with evaluation seeds 51, 52, and 53. The corresponding STAEFormer checkpoints use seeds 42, 43, and 44. We first average errors over seeds within each setting, divide by the matched reference, and take the geometric mean across settings. Confidence intervals use 10,000 district bootstrap draws, keeping years from the same district together.

The primary endpoint is cumulative H12 MAE over the one-hour forecast horizon. `H3_MAE`, `H6_MAE`, and `H12_MAE` aggregate valid targets across steps 1–3, 1–6, and 1–12, respectively. `Avg_MAE` averages the twelve cumulative-horizon MAEs. These definitions match the cumulative metrics in the annual benchmark. Per-step horizon curves are explicitly identified as such. Checkpoint ages, seed counts, observation budgets, and dataset scopes still differ between the two experiments.

All ratios identify their reference. The maintenance comparison uses a calibrated target-year model. After arrival, new/all-sensor ratios use the calibrated target-year model; established-sensor ratios use the same mature model without the arrivals. Checkpoint-age groups contain 18, 12, and 6 settings at ages 5, 10, and 15 years. They are different cohorts.

## Backbone implementations

These experiments use the repository's backbone implementations. The Graph WaveNet variant has two temporal blocks and blends learned and static adjacency. The initial STID implementation uses node embeddings without timestamp embeddings. A separate STAEFormer time-input study is included ([e19_timeinput.py](../experiments/evost/g39/e19_timeinput.py)). Model definitions, checkpoint preparation helpers, and numerical equivalence checks are included with the experiment source.

The source folders keep the identifiers under which the runs were made, so that the saved results stay traceable. [The experiment guide](../experiments/evost/README.md) maps each folder to its study, and [Running EvoST](run_evost.md) gives the commands. [The results page](evost_results.md) reports the main comparisons.
