# EvoST experiments

This folder holds the source of the EvoST experiments in the paper. [Running EvoST](../../docs/run_evost.md) explains the inputs and gives the command for each experiment. This page explains how the folder is organized.

| Path | Content |
|---|---|
| `run.py` | Entry point of the maintenance studies: `python experiments/evost/run.py STUDY ...` |
| `selfcheck.py` | CPU self-check of the RLS calibrator and of the delayed label release |
| `g33/` to `g42/` | One folder per maintenance study, with the source as it ran |
| `export_bundles.py` | Writes the checkpoint bundles the experiments read ([Running EvoST](../../docs/run_evost.md#inputs-checkpoint-bundles)) |
| [annual/](annual/README.md) | The annual-benchmark runner, with its own manifest |

## Studies

| Study | Paper | `run.py` name | Folder | Main module |
|---|---|---|---|---|
| Parameter groups: identities, shared weights, output head, all parameters | Section 6.3, Figure 3 | `maintenance` | `g37` | `e15_maintain.py` |
| EAC-style prompt and recent-data scratch controls | Section 6.3, Figure 3 | `prompt` | `g42` | `e23_eac.py` |
| Further districts and checkpoint ages | Section 6.4, Table 2, Figure 6(d) | `age` | `g34` | `e11_ages.py` |
| STID | Section 6.4, Table 2 | `stid` | `g35` | `e13_stid.py` |
| Graph WaveNet | Section 6.4, Table 2 | `gwn` | `g36` | `e14_gwn.py` |
| STAEFormer with time inputs | Section 6.4 | | `g39` | `e19_timeinput.py` |
| Scheduled refitting of new identities, including borrow and keep | Section 6.5 | | `g40` | `e18_handover.py` |
| Joint refitting and calibrator restart, the final arrival design | Section 6.5, Figure 13 | `arrival` | `g41` | `e21_handover2.py` |
| Recalibration control: the new-only refit with every calibrator rebuilt | Section 6.5 | | `g41` | `e23_recalall.py` |
| Calibrator scale restricted to the targets the arrival masks allow | Appendix A.3 | | `g41` | `e25_scale.py`, `scale_check.py` |
| History budgets and new-sensor initialization | Section 6.8, Figure 6(c) | `budget` | `g38` | `e16_budget.py`, `e17_coldstart.py` |
| Fixed-rate identity refresh and donor comparisons | supporting analyses | | `g33` | `e10_refresh.py` |
| Annual benchmark | Section 6.6 | | `annual` | `e24_annual.py` |

Figure and table numbers are those of the submission. [The overview in the main README](../../README.md#where-each-result-comes-from) also gives the extended version's.

## Why the folders keep their identifiers

`g33` to `g42` are the identifiers under which the runs were made. Each folder is a snapshot of the code that produced the records in `results/evost/<folder>/`, and [the release manifest](../../results/evost/manifest.json) records the SHA-256 of these files. The manifest covers the original g33–g42 source and result snapshot; it does not cover the supplementary files added later (the bundle exporter, the recalibration and scale-check runs, and the extra result files). Renaming the folders or editing the files would break that link. The folders therefore keep their names, and the module names and source comments still use the study and rule identifiers of the development record. The only edits are a few docstrings in which a machine name was replaced by a generic reference.

Within `g41`, the final EvoST arrival configuration is the branch `a_prof0_r`. The other branches are controls and diagnostic variants. `e23_recalall.py` and `e25_scale.py` run `e21_handover2.py` unchanged with one added arm or one changed scale, after the main arrival runs; their module docstrings were shortened for the release, and their records are in `results/evost/g41/recal_V/` and `results/evost/g41/scale_V/`. `scale_check.py` computes `results/evost/g41/scale_check.json` from the twelve main-panel bundles.

## Saved records

`results/evost/<folder>/results_V/` keeps each completed run's metrics, completion receipt, setup, calibration selection and fitting metadata where available. No bounded smoke runs are included. `summary.json` holds the study summary. [scripts/check_paper_numbers.py](../../scripts/check_paper_numbers.py) recomputes the numbers of the case study, the known and unseen sensors, the recalibration control and the scale check from these records and from [results/evost/case_study/](../../results/evost/case_study/README.md). Some analysis scripts also accept the results of an earlier study to check that shared control arms agree. Their command-line options describe those inputs.

```bash
python scripts/render_evost_results.py
python figures/evost/plot_checkpoint_age.py --output outputs/checkpoint_age.pdf
python figures/evost/plot_graph_evo.py --out-dir outputs/graph_evolution
```

The renderer checks the primary twelve-setting ratios and bootstrap intervals against the saved summaries before writing [the results page](../../docs/evost_results.md). The graph plot uses integer node and undirected-edge counts, with average degree `2E/N` and density `2E/[N(N-1)]`.

## Environment

The experiment modules need Python, NumPy and PyTorch. The result renderer needs NumPy, and the figure scripts also need Matplotlib. The experiments do not need PyTorch Geometric or the baseline environment in [environment.yaml](../../environment.yaml).
