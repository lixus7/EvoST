# Editable EvoST figures

| Figure | Editable source | Current PDF |
|---|---|---|
| Introduction, green palette | [H7_www_green.pptx](H7_www_green.pptx) | [H_fig1.pdf](../../docs/assets/H_fig1.pdf) |
| Method, blue/yellow/apricot palette | [fig2_01_blue_yellow_apricot.pptx](fig2_01_blue_yellow_apricot.pptx) | [method PDF](../../docs/assets/fig2_01_blue_yellow_apricot.pdf) |
| Checkpoint age | [plot script](plot_checkpoint_age.py), [values](checkpoint_age.json) | [age PDF](../../docs/assets/evost_checkpoint_age.pdf) |
| Sensor graph evolution | [plot script](plot_graph_evo.py), [integer counts](graph/graph_statistics.csv) | [graph PDF](../../docs/assets/fig_graph_evo.pdf) |

The presentation files contain editable native shapes and text. Their file names retain the original drawing identifiers. In the paper, Figure 2 is the dataset overview; the methodology illustration follows it.

Run the two numerical plot scripts from the repository root:

```bash
python figures/evost/plot_checkpoint_age.py --output outputs/checkpoint_age.pdf
python figures/evost/plot_graph_evo.py --out-dir outputs/graph_evolution
```

The graph counts use each undirected edge once. Average degree is `2E/N` and density is `2E/[N(N-1)]`. The 213-point series contains 212 points reconstructed from cached adjacency counts and one D04/2001 point recovered from the archived vector figure, with `N=1266` and `E=385104`.

Original A²TTA result-figure notebooks and renderers remain under [notebook/](../../notebook/README.md). The [extended experiments](../../docs/extended_experiments.md) link the full rendered figures.
