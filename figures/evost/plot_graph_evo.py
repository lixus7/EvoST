"""Plot the corrected graph summary from editable integer counts."""
import argparse
import csv
from pathlib import Path

from graph_plot import DISTRICTS, plot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--data", type=Path, default=Path(__file__).parent / "graph/graph_statistics.csv"
)
parser.add_argument(
    "--out-dir", type=Path, default=Path(__file__).parent / "preview"
)
args = parser.parse_args()
points = []
seen = set()
with args.data.open() as handle:
    for row in csv.DictReader(handle):
        district = row["district"].zfill(2)
        year = int(row["year"])
        n = int(row["n_nodes"])
        e = int(row["undirected_edges"])
        assert district in DISTRICTS and (district, year) not in seen
        assert n > 1 and 0 <= e <= n * (n - 1) // 2
        seen.add((district, year))
        points.append(
            dict(
                district=district, year=year, n_nodes=n, undirected_edges=e,
                average_degree=2 * e / n, density=2 * e / (n * (n - 1)),
            )
        )
points.sort(key=lambda row: (row["district"], row["year"]))
assert set(row["district"] for row in points) == set(DISTRICTS)
args.out_dir.mkdir(parents=True, exist_ok=True)
plot(points, args.out_dir)
print(args.out_dir / "fig_graph_evo.pdf")
