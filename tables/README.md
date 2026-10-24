# Paper tables

LaTeX sources of the tables of the paper, as in [paper/EvoST_extended.pdf](../paper/EvoST_extended.pdf). The submitted PDF (twelve-page limit) keeps a subset; the extended version keeps all of them. Table 1 (dataset scale) is part of the manuscript source.

| File | Submission | Extended version | Content |
|---|---|---|---|
| `main01.tex` | Table 2 | Table 2 | Avg results on PEMS03, PEMS04 and TFNSW, all methods |
| `evost_maintenance_summary.tex` | Table 3 | Table 3 | Parameter maintenance on five-year-old STAEFormer checkpoints |
| `evost_backbones.tex` | — | Table 4 | Five-year reuse of STAEFormer, STID and Graph WaveNet on sixteen settings |
| `evost_budget.tex` | — | Table 5 | History budgets |
| `evost_arrival.tex` | Table 4 | Table 6 | Sensor arrivals and joint refitting |
| `main02_app.tex` | — | Table 7 | Avg results on PEMS05–PEMS08 |
| `main03_app.tex` | — | Table 8 | Avg results on PEMS10–PEMS12 |
| `main3612_1.tex` | — | Table 9 | Horizons 3, 6, 12 and Avg, PEMS03–PEMS07 |
| `main3612_2.tex` | — | Table 10 | Horizons 3, 6, 12 and Avg, PEMS08, PEMS10–PEMS12 and TFNSW |
| `evost_annual_controls.tex` | Table 5 | Table 11 | Matched controls of the annual benchmark on ten networks |
| `evost_cost.tex` | Table 6 | Table 12 | Median fitting costs |

The benchmark tables (`main*.tex`) define the `\best` and `\second` highlight macros with `\providecommand` and need `booktabs`, `multirow`, `graphicx` and `xcolor` with the `table` option. The EvoST rows and columns come from [results/evost/annual/](../results/evost/annual/README.md); the maintenance tables from the saved summaries in [results/evost/](../results/evost) (see [docs/evost_results.md](../docs/evost_results.md)).
