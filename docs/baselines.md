# Running the baselines

This guide covers the conventional forecasters, continual methods, test-time calibration and time-series foundation models of the annual benchmark (Table 1 of the paper and Tables 5–8 of the [extended version](../paper/EvoST_extended.pdf)). The EvoST runs of the same benchmark are described in [Running EvoST](run_evost.md#annual-benchmark) and [evost_annual.md](evost_annual.md).

## Environment

```bash
conda env create -f environment.yaml
conda activate stg
```

Core dependencies: `python`, `pytorch`, `torch-geometric`, `networkx`, `scipy`, `numpy`, `tqdm`. A single CUDA GPU is enough; CPU works for debugging (`--gpuid -1`).

## Data

The processed tensors are large and are released separately; see [data/README.md](../data/README.md). Place tensors under `data/<dataset>/{RawData,FastData,graph}/` and the stable-ID sidecars under `data/sensor_metadata/<dataset>/<year>/sensor_metadata.csv`; the configs already point to the tensor directories.

The archived baseline rows use the input preprocessing recorded for those runs, including split-specific cached statistics for the conventional baselines. The current data builder uses training-split statistics for all splits. Rebuilding the cache and retraining therefore produces a new benchmark run and is not an exact reconstruction of those archived rows. The four matched annual streams of EvoST use training-split statistics.

## Running the baselines

All baselines reported in the benchmark tables are launched from `scripts/`. Runner options differ. The per-dataset `pemsXX_run.sh` scripts use fixed dataset and seed lists and accept `GPU` and `NOHUP` (`NOHUP=1` runs in the background with a timestamped log under `run_logs/`). The sweep scripts document their supported `DATASETS`, `METHODS` and `SEEDS` overrides in their headers. Use the direct Python entry point to run one method with a chosen seed.

| Group (table column) | Methods | How to run |
|---|---|---|
| Naïve schemes | Pretrain, Retrain, Online-NN, Online-AN | `bash scripts/pemsXX_run.sh` (steps 1–5) |
| Evolving-graph continual | TrafficStream, STKEC, EAC | `bash scripts/pemsXX_run.sh` (steps 6–8) |
| Static STGNN backbones | STGNN, DCRNN, ASTGNN, TGCN | `bash scripts/baselines_pems_run.sh` |
| Retrieval / continual (STGNN backbone) | PECPM, STRAP | `bash scripts/baselines_pems_run.sh` |
| Test-time calibration | ST-TTC | `bash scripts/sttc_run.sh` |
| Other static backbones | GWN, STID, iTransformer, DLinear, STNorm, STAEFormer | `bash scripts/extra_baselines_run.sh` |
| Foundation models (ZS / FT) | TimesFM-2.5, Chronos-2, Moirai-MoE, Moirai-2.0 | see [Foundation-model baselines](#foundation-model-baselines) |

`scripts/pemsXX_run.sh` runs the full per-dataset pipeline end to end (Retrain → auto-link → Pretrain → Online-NN → Online-AN → TrafficStream → STKEC → EAC) and is the recommended starting point.

```bash
# Full pipeline on PEMS05 (naive and continual baselines); without GPU, its three stages use GPUs 2, 0 and 1
GPU=0 bash scripts/pems05_run.sh

# STRAP-paper backbones + PECPM + STRAP, two datasets, on GPU 0
DATASETS="PEMS04 PEMS05" GPU=0 bash scripts/baselines_pems_run.sh

# Extra static backbones, only GWN + STID, all datasets
METHODS="gwn stid" bash scripts/extra_baselines_run.sh

# ST-TTC on a subset
DATASETS="pems05 pems06" GPU=0 bash scripts/sttc_run.sh

# Run a single method directly via main.py
python main.py --conf conf/PEMS05/eac.json --gpuid 0 --seed 51
```

Per-year metrics for every baseline are written to `log/<DATASET>/<logname>-<seed>/<logname>.log`.

The STAEFormer configuration (`conf/<DATASET>/retrain_staeformer_<dataset>.json`, run by `scripts/extra_baselines_run.sh`) also writes the per-year checkpoints that the annual EvoST runs start from: `log/<DATASET>/retrain_staeformer_<dataset>-<seed>/<year>/<loss>.pkl`, with checkpoint seeds 42–46. TFNSW checkpoints follow the same layout. Of the TFNSW configurations, the repository includes only the STAEFormer one (`conf/TFNSW/retrain_staeformer_tfnsw.json`). The [cases files](../experiments/evost/annual/) list the checkpoint of every run with its SHA-256.

## Foundation-model baselines

The zero-shot (ZS) and per-year fine-tuned (FT) columns of TimesFM 2.5, Chronos-2, Moirai-2.0 and Moirai-MoE run from their own entry points, each in a separate conda environment because their dependencies conflict with the baseline stack. [The foundation-model guide](foundation_models.md) covers their setup, commands and comparability. They use the same test windows and metric code as the other methods, but the FT variants train on a 20% per-year slice instead of the 60% training split, so their columns are references and are excluded from best and second-best highlighting.

## Acknowledgements

This benchmark builds on the data and code of several prior works:

- TrafficStream (IJCAI'21): [paper](https://arxiv.org/abs/2106.06273) · [repo](https://github.com/AprLie/TrafficStream)
- EAC (ICLR'25): [paper](https://openreview.net/pdf?id=FRzCIlkM7I) · [code](https://github.com/Onedean/EAC)
- STKEC (TITS'23): [paper](https://ieeexplore.ieee.org/document/10101714/) · [repo](https://github.com/wangbinwu13116175205/STKEC)
- ST-TTC (NeurIPS'25): [paper](https://arxiv.org/pdf/2506.00635) · [repo](https://github.com/Onedean/ST-TTC)
- STRAP (NeurIPS'25): [paper](https://arxiv.org/abs/2505.19547/) · [repo](https://github.com/HoweyZ/STRAP)
- PECPM (KDD'23)
- the conventional backbones GWN, STID, iTransformer, DLinear, ST-Norm and STAEFormer

Foundation-model baselines:

- TimesFM: [paper](https://arxiv.org/abs/2310.10688) · [repo](https://github.com/google-research/timesfm)
- Chronos: [paper](https://arxiv.org/abs/2403.07815) · [repo](https://github.com/amazon-science/chronos-forecasting)
- Moirai / Moirai-MoE (uni2ts): [paper](https://arxiv.org/abs/2402.02592) · [repo](https://github.com/SalesforceAIResearch/uni2ts)
