#!/usr/bin/env python3
"""Validate the Markdown main table and regenerate all LaTeX variants.

``tables/main_table.md`` is the single source of truth. The script recomputes
best/second-best highlighting over non-foundation methods, then writes:

* ``main_table_full.tex``: all ten datasets;
* ``main_table_part1.tex``: PEMS03--PEMS07;
* ``main_table_part2.tex``: PEMS08--PEMS12;
* ``main_table_part3.tex``: TFNSW.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "tables"
MARKDOWN_PATH = TABLES_DIR / "main_table.md"
DATASETS = (
    "PEMS03", "PEMS04", "PEMS05", "PEMS06", "PEMS07",
    "PEMS08", "PEMS10", "PEMS11", "PEMS12", "TFNSW",
)
METRICS = ("MAE", "RMSE", "MAPE")
HORIZONS = ("3", "6", "12", "Avg")
FOUNDATION_METHODS = {
    "Chronos2-U", "Chronos2-M", "Chronos2-FT-M",
    "TimesFM", "TimesFM-FT", "MoE-ZS", "MoE-FT",
    "Moirai2-ZS", "Moirai2-FT",
}

# Group label, member column labels.
GROUPS = (
    ("Static STGNN Backbones", ("DCRNN", "ASTGNN", "TGCN", "GWN")),
    (r'Na\"ive Schemes', ("Pretrain", "Retrain", "OL-NN", "OL-AN")),
    ("Evolving-Graph Continual", ("TrafStm", "PECPM", "STKEC", "EAC")),
    ("Retrieval / TTC", ("STRAP", "ST-TTC")),
    ("Other Static Backbones", ("STID", "STNorm", "iTrans", "DLinear")),
    ("Chronos-2", ("Chronos2-U", "Chronos2-M", "Chronos2-FT-M")),
    ("TimesFM", ("TimesFM", "TimesFM-FT")),
    ("Moirai-MoE", ("MoE-ZS", "MoE-FT")),
    ("Moirai-2.0", ("Moirai2-ZS", "Moirai2-FT")),
    ("STAEFormer", ("STAEFormer",)),
    ("Ours", ("STAE-Ours", "A2TTA")),
)
SUBHEADERS = {
    "Chronos2-U": "U",
    "Chronos2-M": "M",
    "Chronos2-FT-M": "FT-M",
    "TimesFM": "ZS",
    "TimesFM-FT": "FT",
    "MoE-ZS": "ZS",
    "MoE-FT": "FT",
    "Moirai2-ZS": "ZS",
    "Moirai2-FT": "FT",
    "STAEFormer": "STAE",
}


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _clean(cell: str) -> str:
    text = cell.strip()
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2]
    elif text.startswith("_") and text.endswith("_"):
        text = text[1:-1]
    return text.strip()


def _mean_value(cell: str) -> float | None:
    text = _clean(cell)
    if text in {"", "--", "—"}:
        return None
    match = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    return float(match.group()) if match else None


def _rank(cells: list[str], methods: tuple[str, ...]) -> tuple[set[int], set[int]]:
    candidates = sorted(
        (
            (index, value)
            for index, cell in enumerate(cells)
            if methods[index] not in FOUNDATION_METHODS
            and (value := _mean_value(cell)) is not None
        ),
        key=lambda item: item[1],
    )
    if not candidates:
        return set(), set()
    best_value = candidates[0][1]
    best = {
        index for index, value in candidates
        if abs(value - best_value) < 1e-9
    }
    remaining = [(index, value) for index, value in candidates if index not in best]
    if not remaining:
        return best, set()
    second_value = remaining[0][1]
    second = {
        index for index, value in remaining
        if abs(value - second_value) < 1e-9
    }
    return best, second


def _parse_and_rehighlight():
    lines = MARKDOWN_PATH.read_text(encoding="utf-8").splitlines()
    methods: tuple[str, ...] | None = None
    dataset = None
    metric = None
    rows = {}

    for index, line in enumerate(lines):
        dataset_match = re.fullmatch(r"## (PEMS\d+|TFNSW)", line)
        if dataset_match:
            dataset = dataset_match.group(1)
            metric = None
            continue
        if not dataset or not line.startswith("|"):
            continue

        cells = _split_markdown_row(line)
        if len(cells) >= 3 and cells[0] == "Metric" and cells[1] == "Horizon":
            observed = tuple(cells[2:])
            if methods is None:
                methods = observed
            elif methods != observed:
                raise ValueError(f"method columns differ in {dataset}")
            continue
        if methods is None or len(cells) != len(methods) + 2:
            continue

        metric_cell = _clean(cells[0]).replace(" (%)", "")
        if metric_cell in METRICS:
            metric = metric_cell
        horizon = cells[1]
        if metric not in METRICS or horizon not in HORIZONS:
            continue

        method_cells = [_clean(cell) for cell in cells[2:]]
        best, second = _rank(method_cells, methods)
        highlighted = []
        for method_index, cell in enumerate(method_cells):
            if methods[method_index] in FOUNDATION_METHODS or cell in {"--", "—"}:
                highlighted.append(cell)
            elif method_index in best:
                highlighted.append(f"**{cell}**")
            elif method_index in second:
                highlighted.append(f"_{cell}_")
            else:
                highlighted.append(cell)
        display_metric = "MAPE (%)" if metric == "MAPE" else metric
        metric_output = f"**{display_metric}**" if horizon == "3" else ""
        lines[index] = (
            f"| {metric_output} | {horizon} | "
            + " | ".join(highlighted)
            + " |"
        )
        rows[(dataset, metric, horizon)] = (method_cells, best, second)

    if methods is None:
        raise ValueError("no main-table header found")
    expected_methods = tuple(
        method for _group, members in GROUPS for method in members
    )
    if methods != expected_methods:
        raise ValueError(
            "unexpected method columns:\n"
            f"observed={methods}\nexpected={expected_methods}"
        )
    expected_rows = {
        (dataset, metric, horizon)
        for dataset in DATASETS
        for metric in METRICS
        for horizon in HORIZONS
    }
    missing = sorted(expected_rows - set(rows))
    extra = sorted(set(rows) - expected_rows)
    if missing or extra:
        raise ValueError(f"main-table row mismatch: missing={missing}, extra={extra}")

    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return methods, rows


def _tex_value(cell: str) -> str:
    text = _clean(cell)
    if text in {"--", "—"}:
        return "--"
    if "±" not in text:
        return text
    value, deviation = (part.strip() for part in text.split("±", 1))
    return rf"{value}$_{{\pm{deviation}}}$"


def _build_tex(
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    rows,
    *,
    caption: str,
    label: str,
) -> str:
    output = [
        r"% Generated from tables/main_table.md by scripts/finalize_stae_ours.py.",
        r"\providecommand{\best}[1]{\cellcolor{yellow!35}\textbf{\textcolor{red}{#1}}}",
        r"\providecommand{\second}[1]{\cellcolor{black!10}\underline{#1}}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{'c' * (2 + len(methods))}}}",
        r"\toprule",
    ]
    group_header = ["", ""]
    for group, members in GROUPS:
        group_header.append(
            rf"\multicolumn{{{len(members)}}}{{c}}{{\textbf{{{group}}}}}"
        )
    output.append(" & ".join(group_header) + r" \\")

    start = 3
    rules = []
    for _group, members in GROUPS:
        rules.append(rf"\cmidrule(lr){{{start}-{start + len(members) - 1}}}")
        start += len(members)
    output.append(" ".join(rules))
    output.append(
        " & ".join(
            [r"\textbf{Metric}", r"\textbf{Len}"]
            + [
                rf"\textbf{{{SUBHEADERS.get(method, method)}}}"
                for method in methods
            ]
        )
        + r" \\"
    )
    output.append(r"\midrule")

    for dataset_index, dataset in enumerate(datasets):
        output.append(
            rf"\multicolumn{{{2 + len(methods)}}}{{c}}{{\textbf{{{dataset}}}}} \\"
        )
        output.append(r"\midrule")
        for metric_index, metric in enumerate(METRICS):
            for horizon_index, horizon in enumerate(HORIZONS):
                cells, best, second = rows[(dataset, metric, horizon)]
                rendered = []
                for method_index, cell in enumerate(cells):
                    value = _tex_value(cell)
                    if methods[method_index] in FOUNDATION_METHODS or value == "--":
                        rendered.append(value)
                    elif method_index in best:
                        rendered.append(rf"\best{{{value}}}")
                    elif method_index in second:
                        rendered.append(rf"\second{{{value}}}")
                    else:
                        rendered.append(value)
                metric_cell = (
                    rf"\multirow{{{len(HORIZONS)}}}{{*}}{{{metric}}}"
                    if horizon_index == 0 else ""
                )
                output.append(
                    " & ".join([metric_cell, horizon] + rendered) + r" \\"
                )
            if metric_index < len(METRICS) - 1:
                output.append(rf"\cmidrule(lr){{1-{2 + len(methods)}}}")
        if dataset_index < len(datasets) - 1:
            output.append(r"\midrule")
    output.extend((
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ))
    return "\n".join(output)


def main() -> None:
    methods, rows = _parse_and_rehighlight()
    foundation_note = (
        r" Mean$\pm$std; MAPE in percent. Foundation-model references are "
        r"excluded from ranking."
    )
    outputs = {
        TABLES_DIR / "main_table_full.tex": _build_tex(
            DATASETS, methods, rows,
            caption="Main experimental results." + foundation_note,
            label="tab:main_full",
        ),
        TABLES_DIR / "main_table_part1.tex": _build_tex(
            DATASETS[:5], methods, rows,
            caption="Main experimental results (part 1/3)." + foundation_note,
            label="tab:main_part1",
        ),
        TABLES_DIR / "main_table_part2.tex": _build_tex(
            DATASETS[5:9], methods, rows,
            caption="Main experimental results (part 2/3)." + foundation_note,
            label="tab:main_part2",
        ),
        TABLES_DIR / "main_table_part3.tex": _build_tex(
            DATASETS[9:], methods, rows,
            caption="Main experimental results (part 3/3)." + foundation_note,
            label="tab:main_part3",
        ),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(f"[write] {path}")
    print(
        f"[validate] {len(DATASETS)} datasets x {len(METRICS)} metrics x "
        f"{len(HORIZONS)} horizons = {len(rows)} rows; {len(methods)} methods"
    )


if __name__ == "__main__":
    main()
