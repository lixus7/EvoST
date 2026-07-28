# A2TTA Knockout Ablation

Cells are mean ± std over five seeds, with years averaged within each seed. All rows use the strict chronological protocol: `eval_batch_size=1`, `label_delay_steps=12`, `adapt_every_windows=64`, training-only normalization, and metadata sensor identities.

Start from full A2TTA and remove or replace one component while keeping the remaining stack fixed.

| Knockout row | Method | Calibrator | Change |
|---|---|---|---|
| **Ours (full)** | tta_ctx_local | film | FiLM + online TTA + local clone |
| − local clone | tta_all | film | remove local clone |
| − FiLM → affine | tta_ctx_local | affine | replace FiLM with static affine |
| − online TTA | calibrator | film | disable online updates |
| − calibration stack → backbone | backbone | — | frozen backbone only |

## Backbone = Online-AN (TrafficStream)

### PEMS03
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 11.18±0.03 | 12.10±0.04 | 13.74±0.07 | **12.19±0.04** | — |
| − local clone | 11.21±0.03 | 12.16±0.04 | 13.91±0.06 | 12.26±0.04 | +0.07 |
| − FiLM → affine | 11.47±0.05 | 12.48±0.06 | 14.42±0.09 | 12.61±0.06 | +0.42 |
| − online TTA | 11.19±0.03 | 12.13±0.04 | 13.85±0.06 | 12.22±0.04 | +0.04 |
| − calibration stack → backbone | 11.76±0.09 | 12.75±0.09 | 14.67±0.12 | 12.89±0.09 | +0.70 |

### PEMS04
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 16.56±0.04 | 18.01±0.04 | 20.62±0.08 | **18.15±0.05** | — |
| − local clone | 16.63±0.04 | 18.15±0.06 | 20.96±0.12 | 18.32±0.07 | +0.16 |
| − FiLM → affine | 16.99±0.07 | 18.65±0.10 | 21.86±0.19 | 18.88±0.11 | +0.72 |
| − online TTA | 16.58±0.03 | 18.08±0.04 | 20.85±0.08 | 18.24±0.04 | +0.09 |
| − calibration stack → backbone | 17.41±0.14 | 19.05±0.14 | 22.23±0.17 | 19.27±0.15 | +1.12 |

### PEMS05
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 9.63±0.01 | 10.21±0.03 | 11.28±0.05 | **10.27±0.03** | — |
| − local clone | 9.71±0.02 | 10.32±0.03 | 11.46±0.06 | 10.39±0.03 | +0.12 |
| − FiLM → affine | 9.92±0.04 | 10.58±0.05 | 11.88±0.08 | 10.67±0.05 | +0.41 |
| − online TTA | 9.70±0.01 | 10.31±0.03 | 11.45±0.05 | 10.38±0.03 | +0.11 |
| − calibration stack → backbone | 10.11±0.08 | 10.75±0.09 | 12.04±0.12 | 10.85±0.09 | +0.58 |

### PEMS06
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 11.50±0.14 | 12.31±0.16 | 13.77±0.18 | **12.38±0.15** | — |
| − local clone | 11.64±0.15 | 12.53±0.21 | 14.22±0.30 | 12.63±0.21 | +0.25 |
| − FiLM → affine | 11.83±0.22 | 12.72±0.25 | 14.42±0.31 | 12.82±0.25 | +0.44 |
| − online TTA | 11.50±0.10 | 12.33±0.14 | 13.87±0.18 | 12.41±0.13 | +0.03 |
| − calibration stack → backbone | 12.49±0.58 | 13.31±0.55 | 14.89±0.54 | 13.40±0.55 | +1.02 |

## Backbone = STAEFormer

### PEMS03
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 10.57±0.03 | 11.08±0.03 | 11.93±0.05 | **11.10±0.04** | — |
| − local clone | 10.61±0.03 | 11.14±0.03 | 12.01±0.06 | 11.16±0.04 | +0.06 |
| − FiLM → affine | 10.96±0.09 | 11.47±0.08 | 12.33±0.09 | 11.51±0.09 | +0.40 |
| − online TTA | 10.64±0.03 | 11.18±0.04 | 12.05±0.06 | 11.20±0.04 | +0.09 |
| − calibration stack → backbone | 11.23±0.14 | 11.72±0.12 | 12.57±0.13 | 11.76±0.13 | +0.66 |

### PEMS04
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 15.81±0.07 | 16.65±0.11 | 18.06±0.18 | **16.70±0.11** | — |
| − local clone | 15.94±0.10 | 16.81±0.13 | 18.28±0.22 | 16.86±0.14 | +0.16 |
| − FiLM → affine | 17.32±0.91 | 18.14±0.92 | 19.54±0.96 | 18.20±0.92 | +1.50 |
| − online TTA | 15.91±0.08 | 16.78±0.11 | 18.24±0.20 | 16.82±0.12 | +0.13 |
| − calibration stack → backbone | 17.78±0.76 | 18.50±0.81 | 19.84±0.90 | 18.57±0.82 | +1.87 |

### PEMS05
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 9.20±0.02 | 9.50±0.02 | 10.01±0.04 | **9.52±0.02** | — |
| − local clone | 9.25±0.02 | 9.55±0.02 | 10.07±0.04 | 9.57±0.03 | +0.05 |
| − FiLM → affine | 9.57±0.04 | 9.84±0.03 | 10.35±0.04 | 9.87±0.04 | +0.35 |
| − online TTA | 9.28±0.01 | 9.59±0.01 | 10.11±0.03 | 9.61±0.02 | +0.09 |
| − calibration stack → backbone | 9.75±0.04 | 10.01±0.03 | 10.51±0.03 | 10.05±0.03 | +0.52 |

### PEMS06
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 11.73±0.57 | 12.15±0.58 | 12.88±0.58 | **12.18±0.58** | — |
| − local clone | 12.25±0.56 | 12.68±0.58 | 13.42±0.63 | 12.70±0.59 | +0.53 |
| − FiLM → affine | 12.06±0.47 | 12.43±0.46 | 13.11±0.45 | 12.47±0.46 | +0.29 |
| − online TTA | 11.00±0.21 | 11.43±0.22 | 12.18±0.24 | 11.46±0.22 | -0.72 |
| − calibration stack → backbone | 12.33±0.46 | 12.67±0.45 | 13.30±0.44 | 12.70±0.45 | +0.53 |

Across the four datasets, the full stack lowers Avg-MAE relative to the frozen backbone by 6.06% with Online-AN and 6.26% with STAEFormer. Replacing FiLM with affine costs 3.75% and 4.68%, while removing the local clone costs 1.17% and 1.60%, respectively. Freezing FiLM after warm-up is worse in seven of eight backbone–dataset settings, although it improves STAEFormer on PEMS06.
