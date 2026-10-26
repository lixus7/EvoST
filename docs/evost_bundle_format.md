# EvoST bundle inputs

The maintenance drivers consume NumPy `.npz` bundles loaded with `allow_pickle=False`. Bundles keep all sensor axes in the target year's stable-ID order. [export_bundles.py](../experiments/evost/export_bundles.py) writes them from the data and the per-year checkpoints ([Running EvoST](run_evost.md#inputs-checkpoint-bundles)). Large traffic arrays and checkpoint bundles are stored outside Git, and the committed result records can be analyzed without them.

## STAEFormer bundle

| Key | Content |
|---|---|
| `meta_json` | A scalar JSON string with `layout: "g30-stae"`, `dataset`, `year`, `nodes`, `seeds`, `ckpt_years`, `dropout`, `norm`, and split metadata |
| `raw_train` | The final 2,048 training windows plus the 24 rows needed to construct their inputs and targets; shape `[2072, N]` |
| `raw_val`, `raw_test` | Chronological raw splits `[T, N]` |
| `sensor_ids` | Stable target-year sensor IDs in array order |
| `known/yYEAR`, `new_mask/yYEAR` | Boolean masks for membership in each earlier checkpoint |
| `emb/SEED/yYEAR` | A checkpoint's identity table `[12, N, 80]`, mapped into target-year order |
| `ckpt/SEED/yYEAR/STATE_KEY` | Shared STAEFormer state arrays, excluding the adaptive identity table |
| `ref_index` and reference arrays | Saved forecasts used to check the standalone backbone against the original implementation |

`ckpt_years` lists the target year first, followed by available older years. `norm` maps a year string to the mean and standard deviation computed from that year's training inputs. An unknown sensor receives the old checkpoint's mean identity as the starting table row. The evaluation seed is the key used for checkpoint arrays; the recorded STAEFormer export maps evaluation seeds 51–53 to checkpoint seeds 42–44.

Donor diagnostics also read the earlier year's bundle from the same `stae/` directory. They compute historical profiles from its training tail and align its sensor IDs with the target year. Geographic controls and Graph WaveNet read the matching `gcn/DATASET_YEAR.npz` bundle next to `stae/`, using its `adj_norm` (the row-normalized adjacency) and sensor IDs. `export_bundles.py graph` writes these two arrays. [RefreshBundle](../experiments/evost/g37/e10_refresh.py) and [StaeBundle](../experiments/evost/g37/e5_stae.py) are the loaders that read these arrays. The sensor-arrival variants apply masking and refitting to these same raw arrays.

`g28_common.get_windows` preserves the original experiment's indexing: the number of windows is `T - 24`, window `i` takes rows `i:i+12` as input and `i+12:i+24` as targets. This convention intentionally matches the saved results, including the unused final possible window.

## Annual benchmark bundles

The [annual runner](../experiments/evost/annual/README.md) reads the same layout (`"g30-stae"` in `meta_json`) with three differences:

- `ckpt_years` is `[year]`: only the same-year checkpoint. When exported with `--cases`, the checkpoint is identified by relative path and SHA-256 in the cases file, which the exporter checks before loading it, and both values are stored in `meta_json` (`rel`, `manifest_sha256`) for the runner's checkpoint gate.
- `seeds` are the evaluation seeds 51–55, stored with the checkpoints of seeds 42–46.
- `norm[year]` is computed by [`utils/window_stats.window_input_stats`](../utils/window_stats.py) over the year's training split, the same statistics with which validation and test inputs are standardized.

`sensor_ids` may be empty for a year without a sensor-metadata snapshot; the same-year streams do not use sensor IDs. Reference forecasts are stored for graphs with at most 3,000 sensors and are compared with the standalone backbone at run time.

## STID and Graph WaveNet

These drivers also use the raw splits and stable IDs in the STAEFormer bundle. Their `--ckpt` directory supplies architecture-specific per-year weights and metadata produced by [train_stid.py](../experiments/evost/g35/train_stid.py) or [train_gwn.py](../experiments/evost/g36/train_gwn.py). Those preparation scripts additionally take the training rows preceding the bundle's tail, along with a checksum that verifies the two portions belong together.

The public benchmark download links are listed in [data/README.md](../data/README.md). They provide the data, not the bundles or checkpoints, which are large and are exported locally as described above.
