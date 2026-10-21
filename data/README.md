# Datasets

The raw tensors are **too large to ship on GitHub** (each dataset is several GB after
preprocessing), so only the directory skeleton and the build notebooks live here.
The processed `.npz` files and sensor metadata are distributed separately:

- [Processed benchmark data](https://pan.baidu.com/s/1llz16kYY33TrWlKENNHC5A?pwd=xxtf), access code `xxtf`.
- [Raw data](https://pan.baidu.com/s/1BPuxL96npWlfRXDv38duww?pwd=xxtf), access code `xxtf`.

These downloads are the benchmark release. The newer EvoST maintenance checkpoint
bundles are separate inputs described in [the bundle guide](../docs/evost_bundle_format.md).

After downloading, drop the files in so the layout matches the skeleton below. The
config files in [`../conf/`](../conf) already point at these paths
(`data/<dataset>/{RawData,FastData,graph}/`), so no path editing is needed.

## Expected layout

```
data/
├── PEMS/                       # PEMS-Stream (TrafficStream, 2011–2017)
│   ├── RawData/<year>.npz      # raw yearly flow tensor,  key "x" → [T, N]
│   ├── FastData/<year>.npz     # preprocessed splits (see keys below)
│   └── graph/<year>_adj.npz    # yearly adjacency,       key "x" → [N, N]
├── pems03/ ... pems12/         # the 9 XXL expanding-sensor datasets
│   ├── RawData/<year>.npz
│   ├── FastData/<year>.npz
│   └── graph/<year>_adj.npz
├── tfnsw/                      # TFNSW expanding-sensor dataset
│   ├── RawData/<year>.npz
│   ├── FastData/<year>.npz
│   └── graph/<year>_adj.npz
├── sensor_metadata/
│   └── <dataset>/<year>/sensor_metadata.csv
└── processing/                 # notebooks that regenerate everything from raw PEMS dumps
```

Each year has its own sensor set and proximity graph. Additions and removals can
make counts non-monotonic; PEMS05 grows from 6 to 572 sensors across its full span,
about 95 times its initial size. Yearly sensor sets use the first available day in
the initial year and January 1 in subsequent years, with persistent IDs in sorted order.

## File formats

| File | Key(s) | Shape | Notes |
|------|--------|-------|-------|
| `RawData/<year>.npz`   | `x` | `[T, N]` | raw sensor flow; 5-min PeMS or hourly TFNSW |
| `graph/<year>_adj.npz` | `x` | `[N, N]` | weighted adjacency for that year |
| `FastData/<year>.npz`  | `train_x`, `train_y`, `val_x`, `val_y`, `test_x`, `test_y`, `edge_index`, `input_mean`, `input_std` | `x:[*,12,N]`, `y:[*,12,N]` | training-stat z-scored inputs, raw-scale targets, 12→12 windows |
| `sensor_metadata/<dataset>/<year>/sensor_metadata.csv` | `sensor_index`, `node_id` (or `station_id`) | `N` rows | stable sensor IDs in graph-row order |

`FastData/` is **derived** from `RawData/` + `graph/` by
[`utils/data_convert.py::generate_samples`](../utils/data_convert.py) (60/20/20
train/val/test split, training-split statistics reused for every input split,
and non-finite targets replaced by zero and masked in evaluation). You only need
`RawData/` + `graph/`; `FastData/` is rebuilt automatically the first time you run
with `"data_process": 1` in the config (set it back to `0` afterwards to reuse the
cache).

The annual PeMS experiments use the first 31 days of each year with chronological
60/20/20 splits. Full-archive observation counts in the paper's dataset overview
describe a larger collection. After training-stat normalization, non-finite inputs
are set to zero; invalid targets are masked. This evaluation convention does not
imply that every raw array was forward-filled or backward-filled.

Chronos also uses this layout by default. To keep large tensors elsewhere, set
`A2TTA_DATA_ROOT` to the directory containing `pems03/` ... `pems12/`, and set
`A2TTA_TFNSW_ROOT` to the TFNSW dataset directory.

## Regenerating from scratch

See [`processing/`](./processing) for the per-dataset notebooks
(`pemsXX_yearly_nodes.ipynb` → expanding-node bookkeeping, `pemsXX_build_eac_data.ipynb`
→ yearly `RawData`/`graph` tensors, `pems03_build_adj.ipynb` → adjacency construction,
`pems03_data_quality_check.py` → NaN / missing-detector audit).
