# Running EvoST

This guide covers the EvoST experiments of the paper: the maintenance studies on old checkpoints (Sections 6.3–6.5 and 6.8) and the annual benchmark (Section 6.6). Their protocols are described in [evost_protocol.md](evost_protocol.md) and [evost_annual.md](evost_annual.md). The baselines have [their own guide](baselines.md).

The repository supports three uses, which need different inputs:

- **Check the saved results.** The records of the paper's runs are in [results/evost/](../results/evost). The check scripts below recompute the reported comparisons from them on a CPU, without data or checkpoints.
- **Reproduce archived runs.** This needs the original per-year checkpoints, whose paths and SHA-256 values are listed in the annual cases files. These checkpoints are not included in this repository.
- **Make new runs.** Train per-year checkpoints with the baseline pipeline and export bundles from them. Newly trained checkpoints will not generally match the archived hashes, even with the same seeds, so these runs give new results rather than an exact reproduction of the archived ones.

## Requirements

- Python with NumPy and PyTorch. The EvoST experiments do not need PyTorch Geometric or the baseline environment.
- A CUDA GPU for full runs. An A5000 run on PEMS04 in 2025 (4,110 sensors) peaked at 19.4 GiB. Memory also depends on the cache policy and the GPU class: the full H100 sweep included runs above 48 GiB, so this measurement is not a general requirement.
- A CPU is enough for the self-check and for the scripts that check the saved results:

```bash
python experiments/evost/selfcheck.py
python scripts/render_evost_results.py --check
python scripts/render_annual_results.py --check
python scripts/check_paper_numbers.py
python experiments/evost/test_export_bundles.py
```

The self-check uses small synthetic arrays. It tests the calibrator against batch ridge regression and checks that labels not yet released cannot change earlier forecasts. It does not train a traffic model. The last command tests the bundle comparison tool on synthetic bundles.

## Inputs: checkpoint bundles

Every run reads one bundle, a NumPy `.npz` file for one network-year. It holds the raw chronological data splits, the stable sensor IDs and the STAEFormer checkpoints of that network, with every sensor axis in the target year's order. The maintenance bundles contain the target-year checkpoint and the older checkpoints being maintained. The annual bundles contain only the same-year checkpoint. [The bundle format](evost_bundle_format.md) lists every array.

[experiments/evost/export_bundles.py](../experiments/evost/export_bundles.py) writes the bundles from two inputs:

1. **Data.** The EvoXXLTraffic tensors ([download and layout](../data/README.md)), including the `FastData` cache that the benchmark builds on its first run, and the sensor metadata.
2. **Per-year STAEFormer checkpoints.** `METHODS="staeformer" bash scripts/extra_baselines_run.sh` trains them and writes `log/<DATASET>/retrain_staeformer_<dataset>-<seed>/<year>/<loss>.pkl` for the checkpoint seeds 42–46 used in the paper ([baseline guide](baselines.md)).

```bash
python experiments/evost/export_bundles.py stae --dataset PEMS03 --out bundles/stae      # maintenance studies
python experiments/evost/export_bundles.py graph --dataset PEMS03 --out bundles/gcn      # Graph WaveNet and geographic controls
python experiments/evost/export_bundles.py annual --dataset PEMS03 --out bundles/annual  # annual benchmark, new runs
```

- **Where it reads.** By default `data/`, `log/` and `data/sensor_metadata/` of this repository. `--data-root`, `--model-root` and `--metadata-root` point elsewhere, and `--help` lists the other options.
- **Which checkpoints.** Without a cases file, the exporter takes the checkpoint with the lowest loss in its file name for each seed and year, as the paper's maintenance bundles did. This is the mode for new runs. The maintenance bundles hold the target year's checkpoint and those of the earlier panel years (2010, 2015, 2020 and 2025).
- **Archived annual checkpoints.** With `--cases experiments/evost/annual/cases-stage1.json --ckpt-root ORIGINAL_CHECKPOINT_ROOT`, the annual exporter takes exactly the files the cases file lists and checks their SHA-256. `ORIGINAL_CHECKPOINT_ROOT` is the directory containing the `eac/log/...` paths recorded in the cases file.
- **Resources.** The exporter only reads data and checkpoints and runs on a CPU with NumPy and PyTorch.
- **Check.** On the original data and checkpoints, ten sampled exports matched all 4,643 compared numerical and sensor-ID arrays byte for byte (PyTorch 2.2.1, NumPy 1.26.4). The sample covers three maintenance, three graph and four annual bundles from both annual stages. The graph check covers `adj_norm` and `sensor_ids`. This is an array-level check, not byte identity of the bundle files. Metadata are compared separately after removing paths and library versions, and two maintenance bundles differ in a diagnostic of the reference check. `export_bundles.py digest` writes the SHA-256 of every array of a set of bundles, and `compare` checks a new export against such a file.

Bundles and checkpoints are large and are not stored in Git.

The checkpoint-preparation scripts of the STID, Graph WaveNet and time-input studies ([train_stid.py](../experiments/evost/stid/train_stid.py), [train_gwn.py](../experiments/evost/graph_wavenet/train_gwn.py), [train_stae.py](../experiments/evost/time_inputs/train_stae.py)) also need training-prefix files (`<DATASET>_<year>_head.npz`) and their own configuration and reference files, which `export_bundles.py` does not write. The driver that recorded the case study's forecasts and some figure renderers are not included either. The committed metrics and figure data can be checked without these inputs.

## Maintenance experiments

All maintenance experiments share one command line:

```bash
python experiments/evost/run.py STUDY --bundle BUNDLE.npz --seed SEED --out NEW_DIR --device cuda:0
```

| `STUDY` | Paper | What it runs |
|---|---|---|
| `maintenance` | Section 6.3, Figure 3, fitting cost (Figure 12) | The old checkpoint, output-head tuning, shared-weight tuning, identity refresh (EvoST) and full fine-tuning, each with the RLS calibrator, against the target-year checkpoint |
| `prompt` | Section 6.3, Figure 3 | An EAC-style node prompt, identities plus prompt, and a model trained from scratch on the same recent windows |
| `age` | Section 6.4, Figure 6(d), STAEFormer row of Table 2 | Further districts and checkpoints that are five, ten and fifteen years old |
| `stid`, `gwn` | Section 6.4, Table 2 | Identity refresh of the STID node embeddings and of the node vectors behind Graph WaveNet's adaptive adjacency |
| `budget` | Section 6.8, Figure 6(c) | Identity refresh with 64 to 2,048 recent fitting windows. Leave `--budgets` unset for the reported sweep: a subset is a debugging run and is marked `bounded_smoke`, even when full data splits are used |
| `arrival` | Section 6.5, Figure 13 | New sensors that borrow identities at arrival, followed by a refit at stream window 1,024 of either the new identities only or all identities and calibrators (EvoST) |

Section numbers of the main text are the same in the submitted and the extended paper. Figure and table numbers are those of the submission; [the main README](../README.md#where-each-result-comes-from) also lists the extended version's.

- **Settings and seeds.** The main panel is PEMS03, PEMS06, PEMS08, PEMS10, PEMS11 and PEMS12 in 2020 and 2025, with evaluation seeds 51, 52 and 53. Evaluation seed `s` uses checkpoint seed `s − 9`.
- **Output directory.** `--out` must not exist yet. The drivers refuse to overwrite earlier results.
- **Smoke runs.** `--max-windows N` limits each split to `N` windows for debugging and marks the outputs as a smoke test. Omit it for full runs.
- **STID and Graph WaveNet.** Add `--ckpt DIR`. The directory holds the per-year weights `DIR/<DATASET>/<DATASET>_<year>_s<seed>.npz` written by [train_stid.py](../experiments/evost/stid/train_stid.py) or [train_gwn.py](../experiments/evost/graph_wavenet/train_gwn.py). Graph WaveNet also reads the target year's adjacency from a `gcn/<DATASET>_<year>.npz` bundle next to the `stae/` directory.
- **Development switches.** Some modules read environment variables named `G3x_*`, such as `G37_ARMS`, to shorten development runs. Their defaults are the reported settings, so leave them unset.

Five experiments are run through their own scripts. They take the arguments of `run.py` (`--bundle`, `--seed`, `--out`, `--device`):

- the borrow-and-keep arm of the arrival study, `experiments/evost/new_sensor_refit/e18_handover.py`
- the recalibration control of Section 6.5, which rebuilds every calibrator in the new-only refit, `experiments/evost/sensor_arrival/e23_recalall.py`
- the arrival runs with the calibrators' scale restricted to the targets the arrival masks allow (Appendix A.3), `experiments/evost/sensor_arrival/e25_scale.py`, which also needs `--scale-rule full` or `--scale-rule arrival`. `experiments/evost/sensor_arrival/scale_check.py --bundles DIR --out FILE` computes the two scales from the bundles.
- the STAEFormer time-input study of Section 6.4, `experiments/evost/time_inputs/e19_timeinput.py`, which also needs `--models DIR` with the per-year STAEFormer checkpoints that [train_stae.py](../experiments/evost/time_inputs/train_stae.py) retrains with and without time inputs
- the earlier fixed-rate refresh and donor comparisons, `experiments/evost/refresh_and_donors/e10_refresh.py`

## Annual benchmark

One run evaluates one network, year and evaluation seed with the four matched streams Frozen, Frozen+RLS, Refresh and EvoST on that year's own checkpoint.

The commands below reproduce an archived annual run. They require the original checkpoint files whose paths and SHA-256 values are listed in the cases file, and these checkpoints are not included in this repository. Newly trained checkpoints will not generally match those hashes, even with the same seeds. Exporting without `--cases` selects the lowest-loss checkpoint available for each seed and year and is intended for new runs, not for exact reproduction of the archived results. Without a cases file, the runner writes metrics but reports the missing acceptance checks and a `rejected` cell status.

```bash
python experiments/evost/export_bundles.py annual \
  --dataset PEMS05 --years 2005 --out bundles/annual \
  --cases experiments/evost/annual/cases-stage2.json \
  --ckpt-root ORIGINAL_CHECKPOINT_ROOT

python experiments/evost/annual/e24_annual.py \
  --bundle bundles/annual/PEMS05_2005.npz --seed 51 \
  --cases experiments/evost/annual/cases-stage2.json --role formal \
  --out outputs/annual/PEMS05_2005/seed51 --device cuda:0
```

`ORIGINAL_CHECKPOINT_ROOT` is the directory containing the `eac/log/...` paths recorded in the cases file.

- **Seeds.** Evaluation seeds 51–55, with checkpoint seeds 42–46.
- **Cases file.** `cases-stage1.json` lists the runs on PEMS03, PEMS04 and TFNSW, and `cases-stage2.json` those on the other seven districts. Each entry fixes the checkpoint by path and SHA-256 and holds the reference values for the run's acceptance checks.
- **`--role formal`** requires the cases file and complete splits and refuses development switches.
- **Cost.** The paper's 1,165 runs took 100.3 H100 GPU-hours in total.

[The runner's guide](../experiments/evost/annual/README.md) lists the output files, and [evost_annual.md](evost_annual.md) the protocol and the results.

## Outputs and saved results

Each run writes its metrics, the selected calibration settings, the identity-fit record and a completion receipt to `--out`. The records of the paper's runs are in [results/evost/](../results/evost), one folder per study, in [results/evost/annual/](../results/evost/annual/README.md) and in [results/evost/case_study/](../results/evost/case_study/README.md). The render scripts recompute the reported comparisons from these records, and `scripts/render_evost_results.py` also writes [the maintenance results page](evost_results.md).

## Source folders

Each maintenance study has its source in `experiments/evost/<folder>/` and its records in `results/evost/<folder>/`. `run.py` maps each study name to its folder:

| Folder | Study | `run.py` name |
|---|---|---|
| `parameter_groups` | Parameter groups | `maintenance` |
| `prompt_and_scratch` | Prompt and scratch controls | `prompt` |
| `checkpoint_ages` | Further districts and checkpoint ages | `age` |
| `stid` | STID | `stid` |
| `graph_wavenet` | Graph WaveNet | `gwn` |
| `time_inputs` | STAEFormer with time inputs | |
| `new_sensor_refit` | Scheduled refitting of new identities, including the borrow-and-keep arm of the arrival study | |
| `sensor_arrival` | Joint refitting, the final arrival design, the recalibration control and the scale check | `arrival` |
| `history_budget` | History budgets and new-sensor initialization | `budget` |
| `refresh_and_donors` | Fixed-rate identity refresh and donor comparisons | |
| `annual` | Annual benchmark | |

Each folder holds the source as it ran. Module names and source comments use the study numbers of the development record (g33 to g42), which [the experiments README](../experiments/evost/README.md#study-numbers-in-the-source) maps to these folders.
