# Paper tables

LaTeX sources of the tables of the paper, as in [paper/EvoST_extended.pdf](../paper/EvoST_extended.pdf). The submitted PDF (twelve-page limit) keeps a subset; the extended version keeps the rest. Several of these tables are now shown as figures in the paper (noted in brackets). The dataset-scale table (Table 3 of the submission, Table 4 of the extended version) and the split of the maintenance comparison by sensor group (Table 10 of the extended version) are part of the manuscript source.

| File | Submission | Extended version | Content |
|---|---|---|---|
| `main01.tex` | Table 1 | Table 1 | Avg results on PEMS03, PEMS04 and TFNSW, all methods |
| `evost_maintenance_summary.tex` | — (Figure 3) | — (Figure 3) | Parameter maintenance on five-year-old STAEFormer checkpoints |
| `evost_backbones.tex` | Table 2 | Table 2 | Five-year reuse of STAEFormer, STID and Graph WaveNet on sixteen settings |
| `evost_budget.tex` | — | Table 3 | History budgets |
| `evost_arrival.tex` | — (Figure 13) | — (Figure 4) | Sensor arrivals and joint refitting |
| `main02_app.tex` | — | Table 5 | Avg results on PEMS05–PEMS08 |
| `main03_app.tex` | — | Table 6 | Avg results on PEMS10–PEMS12 |
| `main3612_1.tex` | — | Table 7 | Horizons 3, 6, 12 and Avg, PEMS03–PEMS07 |
| `main3612_2.tex` | — | Table 8 | Horizons 3, 6, 12 and Avg, PEMS08, PEMS10–PEMS12 and TFNSW |
| `evost_annual_controls.tex` | — (Figure 10) | Table 9 (and Figure 13) | Matched controls of the annual benchmark on ten networks |
| `evost_cost.tex` | — (Figure 12) | — (Figure 14) | Median fitting costs |

The benchmark tables (`main*.tex`) define the `\best` and `\second` highlight macros with `\providecommand` and need `booktabs`, `multirow`, `graphicx` and `xcolor` with the `table` option. The EvoST rows and columns come from [results/evost/annual/](../results/evost/annual/README.md); the maintenance tables from the saved summaries in [results/evost/](../results/evost) (see [docs/evost_results.md](../docs/evost_results.md)).
