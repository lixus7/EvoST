# EvoST experiments

This folder holds the source of the EvoST experiments in the paper. [Running EvoST](../../docs/run_evost.md) explains the inputs and gives the command for each experiment. This page explains how the folder is organized.

| Path | Content |
|---|---|
| `run.py` | Entry point of the maintenance studies: `python experiments/evost/run.py STUDY ...` |
| `selfcheck.py` | CPU self-check of the RLS calibrator and of the delayed label release |
| `parameter_groups/`, `sensor_arrival/`, ... | One folder per maintenance study (listed below), with the source as it ran |
| `export_bundles.py` | Writes the checkpoint bundles the experiments read ([Running EvoST](../../docs/run_evost.md#inputs-checkpoint-bundles)) |
| [annual/](annual/README.md) | The annual-benchmark runner, with its own manifest |

## Studies

| Study | Paper | `run.py` name | Folder | Main module |
|---|---|---|---|---|
| Parameter groups: identities, shared weights, output head, all parameters | Section 6.3, Figure 3 | `maintenance` | `parameter_groups` | `e15_maintain.py` |
| EAC-style prompt and recent-data scratch controls | Section 6.3, Figure 3 | `prompt` | `prompt_and_scratch` | `e23_eac.py` |
| Further districts and checkpoint ages | Section 6.4, Table 2, Figure 6(d) | `age` | `checkpoint_ages` | `e11_ages.py` |
| STID | Section 6.4, Table 2 | `stid` | `stid` | `e13_stid.py` |
| Graph WaveNet | Section 6.4, Table 2 | `gwn` | `graph_wavenet` | `e14_gwn.py` |
| STAEFormer with time inputs | Section 6.4 | | `time_inputs` | `e19_timeinput.py` |
| Scheduled refitting of new identities, including borrow and keep | Section 6.5 | | `new_sensor_refit` | `e18_handover.py` |
| Joint refitting and calibrator restart, the final arrival design | Section 6.5, Figure 13 | `arrival` | `sensor_arrival` | `e21_handover2.py` |
| Recalibration control: the new-only refit with every calibrator rebuilt | Section 6.5 | | `sensor_arrival` | `e23_recalall.py` |
| Calibrator scale restricted to the targets the arrival masks allow | Appendix A.3 | | `sensor_arrival` | `e25_scale.py`, `scale_check.py` |
| History budgets and new-sensor initialization | Section 6.8, Figure 6(c) | `budget` | `history_budget` | `e16_budget.py`, `e17_coldstart.py` |
| Fixed-rate identity refresh and donor comparisons | supporting analyses | | `refresh_and_donors` | `e10_refresh.py` |
| Annual benchmark | Section 6.6 | | `annual` | `e24_annual.py` |

Figure and table numbers are those of the submission. [The overview in the main README](../../README.md#where-each-result-comes-from) also gives the extended version's.

## Study numbers in the source

Each study folder is a snapshot of the code that produced the records in `results/evost/<folder>/`. The runs were made under the study numbers g33 to g42 of the development record, and the module names and source comments still use these numbers and the rule identifiers of that record:

| Number | Folder |
|---|---|
| g33 | `refresh_and_donors` |
| g34 | `checkpoint_ages` |
| g35 | `stid` |
| g36 | `graph_wavenet` |
| g37 | `parameter_groups` |
| g38 | `history_budget` |
| g39 | `time_inputs` |
| g40 | `new_sensor_refit` |
| g41 | `sensor_arrival` |
| g42 | `prompt_and_scratch` |

Lower numbers (g28 to g32) are earlier stages. Their modules, such as `g28_common.py` and `analyze_g29.py`, are copied into the study folders that use them.

For the release, the folders were renamed after their content and each `results_V/` folder of run records became `runs/`. Renaming did not change the files of the snapshot. [The release manifest](../../results/evost/manifest.json) gives, for each file, its path in the development record (`source`), its path in this repository (`file`) and its SHA-256. A few files were edited for the release, in docstrings, configuration source paths or whitespace, and the manifest's `changes` field describes each edit. The manifest covers the original source and result snapshot. It does not cover the files added later: the bundle exporter, the recalibration and scale-check runs and the extra result files.

Within `sensor_arrival`, the final EvoST arrival configuration is the branch `a_prof0_r`. The other branches are controls and diagnostic variants. `e23_recalall.py` and `e25_scale.py` run `e21_handover2.py` unchanged with one added arm or one changed scale, after the main arrival runs. Their module docstrings were shortened for the release. Their records are in `results/evost/sensor_arrival/recalibration_control/` and `results/evost/sensor_arrival/scale_check_runs/`. `scale_check.py` computes `results/evost/sensor_arrival/scale_check.json` from the twelve main-panel bundles.

## Saved records

`results/evost/<folder>/runs/` keeps each completed run's metrics, completion receipt, setup, calibration selection and fitting metadata where available. No bounded smoke runs are included. `summary.json` holds the study summary. [scripts/check_paper_numbers.py](../../scripts/check_paper_numbers.py) recomputes the numbers of the case study, the known and unseen sensors, the recalibration control and the scale check from these records and from [results/evost/case_study/](../../results/evost/case_study/README.md). Some analysis scripts also accept the results of an earlier study to check that shared control arms agree. Their command-line options describe those inputs.

```bash
python scripts/render_evost_results.py
python figures/evost/plot_checkpoint_age.py --output outputs/checkpoint_age.pdf
python figures/evost/plot_graph_evo.py --out-dir outputs/graph_evolution
```

The renderer checks the primary twelve-setting ratios and bootstrap intervals against the saved summaries before writing [the results page](../../docs/evost_results.md). The graph plot uses integer node and undirected-edge counts, with average degree `2E/N` and density `2E/[N(N-1)]`.

## Environment

The experiment modules need Python, NumPy and PyTorch. The result renderer needs NumPy, and the figure scripts also need Matplotlib. The experiments do not need PyTorch Geometric or the baseline environment in [environment.yaml](../../environment.yaml).
