# EvoST experiments

These are the source snapshots used for identity maintenance and delayed-feedback experiments. The scientific Python code is preserved; a few docstrings replace a machine name with a generic reference. File hashes are recorded in [the release manifest](../../results/evost/manifest.json). The annual-benchmark runner is in [annual/](annual/README.md) with its own manifest; the benchmark's baseline runners and configurations are described in the [baseline guide](../../docs/baselines.md).

| Directory | Study | Main entry |
|---|---|---|
| `g33` | Fixed-rate identity refresh and donor comparisons | `e10_refresh.py` |
| `g34` | More districts and checkpoint ages | `e11_ages.py` |
| `g35` | STID identity refresh | `e13_stid.py` |
| `g36` | Graph WaveNet and adaptive adjacency | `e14_gwn.py` |
| `g37` | Identity, shared-weight, head, and full-parameter maintenance | `e15_maintain.py` |
| `g38` | Observation budgets and sensor initialization | `e16_budget.py`, `e17_coldstart.py` |
| `g39` | Explicit time-input sensitivity | `e19_timeinput.py` |
| `g40` | Scheduled refitting of new identities | `e18_handover.py` |
| `g41` | Joint identity refitting and calibration restart | `e21_handover2.py` |
| `g42` | EAC-style prompt and recent-data scratch controls | `e23_eac.py` |
| `annual` | Annual benchmark: EvoST and matched controls on each year's checkpoint, ten networks | `e24_annual.py` |

The final EvoST arrival branch is `g41/a_prof0_r`. Other branches are controls and diagnostic variants. Source comments retain the original study and rule identifiers.

## Environment

The experiment modules need Python, NumPy, and PyTorch. The saved-result renderer needs NumPy; figure rendering additionally needs Matplotlib. The repository's [environment.yaml](../../environment.yaml) contains the broader baseline stack. These standalone maintenance experiments do not require PyG.

```bash
python experiments/evost/run.py --help
python experiments/evost/run.py maintenance --help
python experiments/evost/selfcheck.py
python scripts/render_evost_results.py --check
```

The self-check uses small synthetic arrays on CPU. It checks RLS against batch ridge regression and checks that changing unavailable future labels leaves past forecasts unchanged. It does not train a traffic model or reproduce the paper's numerical results.

## Run from an exported checkpoint bundle

```bash
python experiments/evost/run.py maintenance \
  --bundle artifacts/bundles/stae/PEMS03_2020.npz \
  --seed 51 --out outputs/maintenance/PEMS03_2020/seed51 --device cuda:0

python experiments/evost/run.py prompt \
  --bundle artifacts/bundles/stae/PEMS03_2020.npz \
  --seed 51 --out outputs/prompt/PEMS03_2020/seed51 --device cuda:0

python experiments/evost/run.py arrival \
  --bundle artifacts/bundles/stae/PEMS03_2020.npz \
  --seed 51 --out outputs/arrival/PEMS03_2020/seed51 --device cuda:0
```

Use a new output directory for each run; the drivers refuse to overwrite an existing one. `--max-windows` is for bounded debugging and marks outputs as a smoke test. It must be omitted for full evaluation. STID and Graph WaveNet also require `--ckpt PATH` pointing to their trained checkpoint directory.

Bundles contain raw chronological splits and model arrays; large bundles and checkpoints are not included in this Git repository. The data downloads in [data/README.md](../../data/README.md) provide the benchmark data, but are not a release of these newer checkpoint bundles. [Bundle format](../../docs/evost_bundle_format.md) describes the required inputs. The committed metric records and analysis scripts can be used immediately without those bundles.

## Saved results

`results/evost/gXX/results_V/` preserves each completed run's metrics, completion receipt, setup, calibration selection, and fitting metadata when available. No bounded smoke runs are included. `summary.json` preserves the corresponding study summary. Some original analysis scripts also accept earlier study results to check agreement between shared control arms; their command-line options describe those inputs.

```bash
python scripts/render_evost_results.py
python figures/evost/plot_checkpoint_age.py --output outputs/checkpoint_age.pdf
python figures/evost/plot_graph_evo.py --out-dir outputs/graph_evolution
```

The renderer checks the primary twelve-setting ratios and bootstrap intervals against the saved summaries before writing [the results page](../../docs/evost_results.md). The corrected graph plot uses integer node and undirected-edge counts, with average degree `2E/N` and density `2E/[N(N-1)]`.
