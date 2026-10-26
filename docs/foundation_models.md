# Foundation-model baselines

The benchmark tables report time-series foundation models as references, each zero-shot (ZS) and fine-tuned per year (FT). Table 1 shows Chronos-2, TimesFM 2.5 and Moirai-2.0. The per-horizon tables of the [extended version](../paper/EvoST_extended.pdf) (Tables 7 and 8) add Moirai-MoE. The other baselines are covered in [the baseline guide](baselines.md).

## How they stay comparable

- **Same test windows.** Each entry point rebuilds raw-scale context windows from `RawData/<year>.npz` with the indexing that built `FastData/<year>.npz` and checks them against the cached test targets before evaluating.
- **Raw scale.** The models normalize each series internally, so they receive raw inputs and return raw forecasts.
- **Same metrics.** Masked MAE, RMSE and MAPE (in percent) at cumulative horizons 3, 6 and 12, computed by the benchmark's metric code. Avg is the arithmetic mean of the twelve cumulative-horizon values for horizons 1 through 12.
- **Fine-tuning data.** FT variants train on a 20% slice of each year, whereas the trained spatio-temporal baselines use the 60% training split. The foundation models are therefore references and are not ranked.

## Environments

Their dependencies conflict with the baseline environment (`stg`) and with each other, so each family runs in its own conda environment. Checkpoints are downloaded from Hugging Face into `HF_HOME` (the setup scripts use `./hf_cache`). Later runs can work offline with `HF_HUB_OFFLINE=1`.

| Models | Setup | Environment |
|---|---|---|
| TimesFM 2.5 | `bash scripts/setup_timesfm_env.sh` | `timesfm`, with TimesFM 2.5 installed from source and the checkpoint `google/timesfm-2.5-200m-pytorch` |
| Moirai-2.0, Moirai-MoE | `bash scripts/setup_moirai_env.sh` | `moirai`, with `uni2ts==2.0.0`, PyTorch 2.4.1 and the checkpoints `Salesforce/moirai-2.0-R-small` and `Salesforce/moirai-moe-1.0-R-base` |
| Chronos-2 | install by hand | `chronos`, with the `chronos-forecasting` package (Chronos-2 support) and `peft` for LoRA |

## TimesFM 2.5

```bash
conda activate timesfm
python timesfm_main.py --conf conf/PEMS05/timesfm_pems05.json --gpuid 0                         # zero-shot, all years
python timesfm_finetune_main.py --conf conf/PEMS05/timesfm_ft_pems05.json --gpuid 0 --seed 51   # per-year LoRA fine-tuning
```

- Zero-shot forecasts are deterministic, so one seed suffices.
- The 12-step inputs are padded to a context of 32, which TimesFM masks.
- `scripts/gen_timesfm_configs.py` and `scripts/gen_timesfm_ft_configs.py` regenerate the configurations from each dataset's retrain configuration. `scripts/timesfm_run.sh` loops over datasets on one GPU.

## Chronos-2

```bash
conda activate chronos
python chronos_baseline.py --dataset PEMS05 --gpuid 0                                                  # zero-shot
python chronos_finetune.py --dataset PEMS05 --mode multivariate --grouping graph --gpuid 0 --seed 51   # per-year LoRA
```

- The paper's Chronos-2 runs forecast the sensors of each year jointly, grouped by the sensor graph (multivariate graph-grouped inference). This is the zero-shot script's default. The fine-tuning script defaults to univariate mode, so pass `--mode multivariate --grouping graph` to match the paper.
- `scripts/chronos_run.sh` runs the zero-shot sweep on one GPU. The PBS scripts `scripts/hpc_chronos_ft_*.pbs`, submitted by `scripts/submit_chronos_ft_mv_queues.sh`, are examples of the fine-tuning sweep on a cluster.
- Results go to `run_logs/chronos/<dataset>/<dataset>_results.csv` (fine-tuning: `run_logs/chronos_ft/`), one row per year.

## Moirai-2.0 and Moirai-MoE

```bash
conda activate moirai
python moirai_main.py --conf conf/PEMS05/moirai_moirai2_pems05.json --gpuid 0                          # Moirai-2.0, zero-shot
python moirai_main.py --conf conf/PEMS05/moirai_moe_pems05.json --gpuid 0                              # Moirai-MoE, zero-shot
python moirai_finetune_main.py --conf conf/PEMS05/moirai_ft_moirai2_pems05.json --gpuid 0 --seed 51    # per-year fine-tuning
```

- Moirai-2.0 forecasts the 0.5 quantile. Moirai-MoE averages `num_samples` sample paths.
- `uni2ts` provides no LoRA, so FT updates all parameters of Moirai-2.0. The Moirai-MoE configuration freezes its encoder (`freeze_encoder: 1`) and trains the input and output projections, the scaler and the distribution head.
- `scripts/gen_moirai_configs.py` and `scripts/gen_moirai_ft_configs.py` regenerate the configurations.

## Seeds, outputs and quick checks

- FT results are mean ± standard deviation over seeds 51, 52 and 53.
- TimesFM and Moirai log per-year metrics to `log/<DATASET>/<logname>-<seed>/<logname>.log`, as the other baselines do. `--csv_path FILE` also writes one CSV row per year.
- For a quick check, the TimesFM and Moirai entry points take `--fast_dev_run N` (first `N` years only) and `--max_samples N` (at most `N` test windows per year, or `--max_test_samples N` when fine-tuning). The Chronos-2 zero-shot script takes `--year` and `--max-windows`.

The wrappers are [src/model/timesfm_wrapper.py](../src/model/timesfm_wrapper.py) and [src/model/moirai_wrapper.py](../src/model/moirai_wrapper.py). TimesFM uses [timesfm_revin_patch.py](../timesfm_revin_patch.py) to align input normalization.
