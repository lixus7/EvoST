# Sensor-coverage maps

`render_final.py` (with `_cohort_lib.py`) draws the 3 × 3 grid of district maps that colors every PeMS station by the first yearly snapshot in which it appears. The committed `pems_grid_final_{carto,vector}.{png,pdf}` are an earlier rendering with snapshots up to 2024; Figure 2 of the paper uses the snapshots up to 2025.

Inputs: PeMS station metadata snapshots `meta/dXX_text_meta_YYYY_MM_DD.txt` at the repository root; they are not included in this repository. The script needs `cartopy`; the `carto` mode fetches basemap tiles online, while the `vector` mode uses Natural Earth features.

```bash
python figures/data/render_final.py
```

The outputs are written next to the script.
