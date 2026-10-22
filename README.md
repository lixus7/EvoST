# EvoST: Adapting Traffic Forecasters to Evolving Sensor Graphs

EvoST maintains spatio-temporal forecasters as traffic sensor networks evolve. It freezes shared weights, refreshes sensor identities using recent observations, and calibrates forecasts as delayed labels arrive. New sensors borrow identities from established sensors with similar traffic patterns; a later joint refit updates all identities and recalibrates their forecasts.

![EvoST overview](docs/assets/fig2_01_blue_yellow_apricot.png)

[Method and evaluation protocol](docs/evost_protocol.md) · [Results](docs/evost_results.md) · [Experiment code](experiments/evost/README.md) · [Extended appendix](docs/extended_experiments.md)

## Results

Across twelve PeMS region-year settings, identity refresh reduces the relative MAE gap over the one-hour forecast horizon between five-year-old STAEFormer checkpoints and calibrated target-year models from **17.7% to 2.5%**. It captures **95%** of the improvement achieved by full-parameter fine-tuning on the same recent data. The Graph WaveNet implementation recovers **88%** of its five-year MAE gap by refreshing the embeddings that define learned adjacency. STID and older STAEFormer checkpoints provide further evaluations.

| Evidence | Read or reproduce |
|---|---|
| Parameter updates, prompt controls, and absolute errors | [EvoST results](docs/evost_results.md) |
| New-sensor initialization and joint refitting | [Arrival results](docs/evost_results.md#sensor-arrival-and-joint-refitting) |
| Per-run metrics, calibration selections, and summaries | [Saved measurements](results/evost) |
| Complete original baseline comparisons | [Annual benchmark table](tables/main_table.md) |
| Original newly added sensor comparisons | [New-sensor table](tables/tsas_new_sensors.md) |
| Original main results, delay sweep, ablations, and cases | [Annual benchmark analysis](docs/annual_benchmark_analysis.md) |
| Per-horizon curves, drift, cost, sensitivity, and case studies | [Extended experiments](docs/extended_experiments.md) |
| Current figures and editable graph/age plots | [Figure assets](docs/assets) · [plot sources](figures/evost) |

The annual benchmark uses the original A²TTA method, including its FiLM calibrator and local clone. Its complete baselines, results, and runnable entry points remain in this repository. EvoST's maintenance experiments use different checkpoint ages, seed counts, and observation budgets, described in the [protocol](docs/evost_protocol.md). Both studies define cumulative-horizon metrics explicitly.

## Paper and extended material

The WWW draft allocates space according to the claims each experiment supports. The conference allows eight main-text pages and twelve pages in total, including references and appendix. [WWW 2027 formatting requirements](https://www2027.thewebconf.org/research-track-papers/).

| Location | Evidence retained |
|---|---|
| Main text | Absolute MAE for all twelve primary settings; matched parameter, prompt, and scratch controls; STID and Graph WaveNet; checkpoint ages; sensor arrival and joint refitting. |
| PDF appendix | Additional horizons and RMSE; fitting and delayed-feedback details; history budgets and fitting costs; the original baseline table for three representative networks; graph statistics and extended related work. |
| Online record | Complete ten-network baseline tables and horizon curves; annual new-sensor cohorts; the original A²TTA delay sweep, ablations, drift analyses, long-term checks, cases, and mechanism visualizations; source and per-run records. |

The main text contains the evidence for EvoST's core claims. The online material preserves the complete experimental record, including the earlier output-adaptation results under their original names and protocols. Start with [EvoST results](docs/evost_results.md), [annual benchmark analysis](docs/annual_benchmark_analysis.md), or the [extended appendix](docs/extended_experiments.md).

## Code and reproduction

The new experiment source is under [`experiments/evost/`](experiments/evost). It contains identity maintenance for STAEFormer, STID, and Graph WaveNet; per-sensor RLS; sensor-arrival refitting; and the matched parameter, prompt, and scratch controls.

```bash
# Read entry-point options.
python experiments/evost/run.py --help

# Small numerical and delayed-feedback checks on CPU.
python experiments/evost/selfcheck.py

# Rebuild result documentation from committed metric records; no training.
python scripts/render_evost_results.py
```

The experiment modules need NumPy and PyTorch. The saved-result renderer needs NumPy; plot generation additionally needs Matplotlib. See the [experiment guide](experiments/evost/README.md) for full run commands and the [bundle format](docs/evost_bundle_format.md) for data and checkpoint inputs. Large maintenance bundles and checkpoints are not included in Git. Saved-result analysis is available directly from this checkout.

The original benchmark retains its existing environment and commands:

```bash
conda env create -f environment.yaml
conda activate stg
```

Use the [A²TTA and baseline guide](docs/a2tta.md) for the original pipeline. It covers static forecasters, continual methods, STRAP, ST-TTC, and the time-series foundation-model baselines. The existing `main.py`, `a2tta_main.py`, `src/`, `conf/`, and baseline scripts are preserved.

## Data and supplementary details

[EvoXXLTraffic data and downloads](data/README.md) · [data processing](data/processing/README.md) · [original figure notebooks](notebook/README.md).

The [extended appendix](docs/extended_experiments.md) provides the full result tables, all-network horizon plots, sensor-cohort analyses, drift diagnostics, long-term checks, hyperparameter studies, and mechanism visualizations. The [LaTeX appendix and table sources](docs/paper_sources) and [bibliography](docs/paper_sources/references.bib) retain the detailed source material. The [EvoST protocol](docs/evost_protocol.md) specifies fitting windows, delayed feedback, arrival handling, and aggregation.

## Acknowledgements and license

The benchmark builds on TrafficStream, EAC, STKEC, PECPM, STRAP, ST-TTC, and the forecasting backbones and foundation models listed in the [original guide](docs/a2tta.md). Source comments preserve model provenance and implementation differences.

[Apache-2.0 license](LICENSE).
