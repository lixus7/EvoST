import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DISTRICTS = ('03','04','05','06','07','08','10','11','12')

def plot(points, output):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 17,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(DISTRICTS)))
    for district, color in zip(DISTRICTS, colors):
        rows = [point for point in points if point["district"] == district]
        years = [point["year"] for point in rows]
        common = dict(color=color, lw=2.4, ms=5, label="D" + district)
        left.plot(years, [p["average_degree"] for p in rows], "o-", **common)
        right.plot(years, [p["density"] for p in rows], "o-", **common)
    left.set_title("(a) Average node degree", fontsize=19)
    right.set_title("(b) Graph density", fontsize=19)
    left.set_ylabel(r"Average degree, $2E/N$", fontsize=18)
    right.set_ylabel(r"Density, $2E/[N(N-1)]$", fontsize=18)
    for axis in (left, right):
        axis.set_xlabel("Year", fontsize=18)
        axis.tick_params(labelsize=17)
        axis.grid(alpha=0.3)
        axis.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    left.set_ylim(bottom=0)
    right.set_ylim(0, 1)
    assert len(left.lines) == len(right.lines) == 9
    assert all(right.get_ylim()[0] <= p["density"] <= right.get_ylim()[1] for p in points)
    assert all(left.get_ylim()[0] <= p["average_degree"] <= left.get_ylim()[1] for p in points)
    fig.suptitle(
        "Structural evolution of the sensor graphs",
        fontweight="bold", fontsize=19, y=0.99,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.20, wspace=0.29)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=9, fontsize=16,
        frameon=False, bbox_to_anchor=(0.5, 0),
    )
    fig.savefig(
        output / "fig_graph_evo.pdf", bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(output / "fig_graph_evo.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
