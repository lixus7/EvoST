# Case study (PEMS12, 2020)

The values behind the case-study figure (Figure 7 of the submission, Figure 9 of the extended version). Evaluation seed 51; the old checkpoint is the 2015 STAEFormer checkpoint, the reference the 2020 one; all forecasts are calibrated.

| File | Content |
|---|---|
| `sensors.csv` | Panel (a): for each of the 2,171 sensors with valid test targets, whether the old checkpoint knows it (`known`), and the test H12 MAE (steps 1–12) of the old checkpoint and of EvoST relative to the target-year checkpoint. From the main maintenance evaluation ([parameter_groups](../parameter_groups)). |
| `forecasts.csv` | Panel (b): observed flow and the 60-minute-ahead (step 12) forecasts of the target-year checkpoint, the old checkpoint and EvoST for three sensors over one day, every 5 minutes. From a separate run of the same protocol with the same seed and sensors, which records the forecasts. |
| `case.json` | Network, seed, checkpoint years, the day shown and the three sensors |

The p50 and p90 sensors have the nearest-rank median and 90th-percentile relative MAE reductions among the 1,126 known sensors with at least 95% valid test targets; the unseen sensor has the highest share of valid targets, ties broken by sensor ID. The sensors were fixed before the run of panel (b). The day is the first complete weekday by 60-minute target time.

```bash
python scripts/check_paper_numbers.py case-study
```
