# Editable EvoST figures

| Figure | Editable source | Current PDF |
|---|---|---|
| Introduction, green palette | [H7_www_green.pptx](H7_www_green.pptx) | [H_fig1.pdf](../../docs/assets/H_fig1.pdf) |
| Method, blue/yellow/apricot palette | [fig2_01_blue_yellow_apricot.pptx](fig2_01_blue_yellow_apricot.pptx) | [method PDF](../../docs/assets/fig2_01_blue_yellow_apricot.pdf) |
| Checkpoint age | [plot script](plot_checkpoint_age.py), [values](checkpoint_age.json) | [age PDF](../../docs/assets/evost_checkpoint_age.pdf) |
| Sensor graph evolution | [plot script](plot_graph_evo.py), [integer counts](graph/graph_statistics.csv) | [graph PDF](../../docs/assets/fig_graph_evo.pdf) |

The presentation files contain editable native shapes and text. Their file names retain the original drawing identifiers. In the paper, Figure 2 is the methodology illustration; the dataset overview is Figure 8 of the submission (Figure 10 of the extended version).

Run the two numerical plot scripts from the repository root:

```bash
python figures/evost/plot_checkpoint_age.py --output outputs/checkpoint_age.pdf
python figures/evost/plot_graph_evo.py --out-dir outputs/graph_evolution
```

The graph counts use each undirected edge once. Average degree is `2E/N` and density is `2E/[N(N-1)]`. The 213-point series contains 212 points reconstructed from cached adjacency counts and one D04/2001 point recovered from the archived vector figure, with `N=1266` and `E=385104`.

The annual-benchmark per-step figures are in [docs/assets](../../docs/assets) (`per_horizon_evost_controls_*`); their values are in [results/evost/annual/per_step.csv](../../results/evost/annual/README.md). The sensor-coverage maps are drawn by [figures/data/render_final.py](../data/README.md).
