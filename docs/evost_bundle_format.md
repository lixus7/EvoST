# EvoST bundle inputs

The maintenance drivers consume NumPy `.npz` bundles loaded with `allow_pickle=False`. Bundles keep all sensor axes in the target year's stable-ID order. Large traffic arrays and checkpoint bundles are stored outside Git; the committed result records can be analyzed without them.

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

Donor diagnostics also read the earlier year's bundle from the same `stae/` directory. They compute historical profiles from its training tail and align its sensor IDs with the target year. Geographic controls read the matching `gcn/DATASET_YEAR.npz` bundle next to `stae/`, using its `adj_norm` and sensor IDs. [RefreshBundle](../experiments/evost/g37/e10_refresh.py) and [StaeBundle](../experiments/evost/g37/e5_stae.py) are the authoritative loaders; inspect their reads when exporting an existing dataset. The sensor-arrival variants apply masking and refitting to these same raw arrays.

`g28_common.get_windows` preserves the original experiment's indexing: the number of windows is `T - 24`, window `i` takes rows `i:i+12` as input and `i+12:i+24` as targets. This convention intentionally matches the saved results, including the unused final possible window.

## STID and Graph WaveNet

These drivers also use the raw splits and stable IDs in the STAEFormer bundle. Their `--ckpt` directory supplies architecture-specific per-year weights and metadata produced by [train_stid.py](../experiments/evost/g35/train_stid.py) or [train_gwn.py](../experiments/evost/g36/train_gwn.py). Those preparation scripts additionally take the training rows preceding the bundle's tail, along with a checksum that verifies the two portions belong together.

The public benchmark download links are listed in [data/README.md](../data/README.md). They should not be interpreted as a download of the new maintenance checkpoint bundles. The source release includes the experiment drivers and saved measurements; publishing those large bundles is a separate artifact release.
