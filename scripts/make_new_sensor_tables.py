#!/usr/bin/env python3
"""Build the corrected newly-added-sensor result tables.

The table combines three independently generated result layers:

* non-STAEFormer baselines, stored as per-dataset ``summary.csv`` files;
* frozen STAEFormer, stored in the same summary format;
* A2TTA on Online-AN and STAEFormer, stored as per-run detail CSVs.

Only ``scope=new`` is used. A2TTA detail files must record the strict
chronological protocol (batch size 1, 12-window label delay, 64-window update
cadence, training-only normalization, and metadata sensor identities).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "PEMS03", "PEMS04", "PEMS05", "PEMS06", "PEMS07",
    "PEMS08", "PEMS10", "PEMS11", "PEMS12", "TFNSW",
)
DATASET_PART1 = DATASETS[:5]
DATASET_PART2 = DATASETS[5:]
HORIZONS = ("3", "6", "12", "Avg")
METRICS = ("MAE", "RMSE", "MAPE")

# (group, display label, result slug)
COLUMNS = (
    ("Static STGNN Backbones", "DCRNN", "retrain_dcrnn"),
    ("Static STGNN Backbones", "ASTGNN", "retrain_astgnn"),
    ("Static STGNN Backbones", "TGCN", "retrain_tgcn"),
    ("Static STGNN Backbones", "GWN", "retrain_gwn"),
    (r'Na\"ive Schemes', "Pretrain", "pretrain_st"),
    (r'Na\"ive Schemes', "Retrain", "retrain_st"),
    (r'Na\"ive Schemes', "OL-NN", "oneline_st_nn"),
    (r'Na\"ive Schemes', "OL-AN", "oneline_st_an"),
    ("Evolving-Graph Continual", "TrafStm", "trafficstream"),
    ("Evolving-Graph Continual", "PECPM", "pecpm"),
    ("Evolving-Graph Continual", "STKEC", "stkec"),
    ("Evolving-Graph Continual", "EAC", "eac"),
    ("Retrieval / TTC", "STRAP", "strap"),
    ("Retrieval / TTC", "ST-TTC", "sttc"),
    ("Other Static Backbones", "STID", "retrain_stid"),
    ("Other Static Backbones", "STNorm", "retrain_stnorm"),
    ("Other Static Backbones", "iTrans", "retrain_itransformer"),
    ("Other Static Backbones", "DLinear", "retrain_dlinear"),
    ("Static Transformer", "STAEFormer", "retrain_staeformer"),
    ("Ours", "A2TTA (OL-AN)", "a2tta_lite"),
    ("Ours", "A2TTA (STAE)", "a2tta_staef"),
)
BASELINE_METHODS = {slug for _group, _label, slug in COLUMNS[:18]}
PLAIN_STAE_METHOD = "retrain_staeformer"
A2TTA_METHODS = {"a2tta_lite", "a2tta_staef"}
EXPECTED_METHODS = tuple(slug for _group, _label, slug in COLUMNS)
BACKBONE_TO_METHOD = {
    "trafficstream": "a2tta_lite",
    "staeformer": "a2tta_staef",
}

# (dataset, method, horizon, seed) -> metric dictionary
SeedKey = tuple[str, str, str, str]


def _summary_files(run_root: Path) -> list[Path]:
    files = sorted((run_root / "datasets").glob("*/summary.csv"))
    observed = {path.parent.name for path in files}
    missing = sorted(set(DATASETS) - observed)
    if missing:
        raise FileNotFoundError(
            f"{run_root} is missing dataset summaries: {', '.join(missing)}"
        )
    return files


def _read_summary_layer(
    run_root: Path,
    allowed_methods: set[str],
) -> dict[SeedKey, dict[str, float]]:
    values: dict[SeedKey, dict[str, float]] = {}
    for path in _summary_files(run_root):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["scope"] != "new" or row["method"] not in allowed_methods:
                    continue
                sensor_mode = row.get("sensor_id_mode", "")
                if sensor_mode and sensor_mode != "metadata_id_difference":
                    raise ValueError(
                        f"unexpected sensor_id_mode={sensor_mode!r} in {path}"
                    )
                key = (
                    row["dataset"], row["method"], row["horizon"], str(row["seed"])
                )
                current = {metric: float(row[metric]) for metric in METRICS}
                if key in values and values[key] != current:
                    raise ValueError(f"conflicting duplicate summary row {key} in {path}")
                values[key] = current
    return values


def _select_a2tta_detail_files(run_roots: list[Path]) -> list[Path]:
    """Select the newest file for each dataset/backbone/seed combination."""
    selected: dict[tuple[str, str, str], tuple[int, int, Path]] = {}
    for run_order, run_root in enumerate(run_roots):
        files = sorted((run_root / "detail").glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no detail CSVs under {run_root / 'detail'}")
        for path in files:
            with path.open(newline="", encoding="utf-8") as handle:
                first = next(csv.DictReader(handle), None)
            if first is None:
                raise ValueError(f"empty detail CSV: {path}")
            method = BACKBONE_TO_METHOD.get(first["backbone_method"].lower())
            if method is None:
                raise ValueError(
                    f"unknown backbone_method={first['backbone_method']!r} in {path}"
                )
            key = (first["dataset"], method, str(first["seed"]))
            candidate = (path.stat().st_mtime_ns, run_order, path)
            if key not in selected or candidate[:2] > selected[key][:2]:
                selected[key] = candidate
    return [candidate[2] for candidate in selected.values()]


def _read_a2tta_layer(run_roots: list[Path]) -> dict[SeedKey, dict[str, float]]:
    values: dict[SeedKey, dict[str, float]] = {}
    for path in _select_a2tta_detail_files(run_roots):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        first = rows[0]
        dataset = first["dataset"]
        seed = str(first["seed"])
        method = BACKBONE_TO_METHOD[first["backbone_method"].lower()]

        identity = {
            (
                row["dataset"],
                row["method"],
                row["backbone_method"].lower(),
                str(row["seed"]),
            )
            for row in rows
        }
        expected_identity = {
            (dataset, "tta_ctx_local", first["backbone_method"].lower(), seed)
        }
        if identity != expected_identity:
            raise ValueError(f"mixed run identity in {path}: {sorted(identity)}")

        protocol = {
            (
                row["label_delay_steps"],
                row["eval_batch_size"],
                row["adapt_every_windows"],
                row["input_normalization"],
                row["sensor_id_mode"],
            )
            for row in rows
        }
        if protocol != {("12", "1", "64", "train", "metadata")}:
            raise ValueError(f"non-strict or mixed A2TTA protocol in {path}: {protocol}")

        by_horizon: dict[str, list[dict[str, float]]] = defaultdict(list)
        seen_years: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row["scope"] != "new":
                continue
            horizon = row["horizon"]
            if row["year"] in seen_years[horizon]:
                raise ValueError(
                    f"duplicate new-sensor year/horizon in {path}: "
                    f"{row['year']}/{horizon}"
                )
            seen_years[horizon].add(row["year"])
            by_horizon[horizon].append(
                {metric: float(row[metric]) for metric in METRICS}
            )

        for horizon, yearly in by_horizon.items():
            values[(dataset, method, horizon, seed)] = {
                metric: mean(row[metric] for row in yearly)
                for metric in METRICS
            }
    return values


def _validate_coverage(
    values: dict[SeedKey, dict[str, float]], expected_seeds: int
) -> None:
    coverage: dict[tuple[str, str], set[str]] = defaultdict(set)
    horizons: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for dataset, method, horizon, seed in values:
        coverage[(dataset, method)].add(seed)
        horizons[(dataset, method, seed)].add(horizon)

    bad_seeds = {
        (dataset, method): sorted(coverage.get((dataset, method), set()))
        for dataset in DATASETS
        for method in EXPECTED_METHODS
        if len(coverage.get((dataset, method), set())) != expected_seeds
    }
    if bad_seeds:
        raise ValueError(f"incomplete seed coverage: {bad_seeds}")

    missing_horizons = {
        key: sorted(set(HORIZONS) - present)
        for key, present in horizons.items()
        if set(HORIZONS) - present
    }
    if missing_horizons:
        raise ValueError(f"missing published horizons: {missing_horizons}")

    expected_groups = len(DATASETS) * len(EXPECTED_METHODS) * expected_seeds
    observed_groups = len(
        {
            (dataset, method, seed)
            for dataset, method, _horizon, seed in values
        }
    )
    if observed_groups != expected_groups:
        raise ValueError(
            f"coverage mismatch: observed {observed_groups}, expected {expected_groups}"
        )
    print(
        f"[coverage] {len(DATASETS)} datasets x {len(EXPECTED_METHODS)} methods "
        f"x {expected_seeds} seeds = {observed_groups} groups"
    )


def _aggregate(values):
    buckets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (dataset, method, horizon, _seed), metrics in values.items():
        for metric, value in metrics.items():
            buckets[(dataset, method, horizon, metric)].append(value)
    return {
        key: (mean(items), stdev(items) if len(items) > 1 else 0.0)
        for key, items in buckets.items()
    }


def _rank(items):
    ordered = sorted(
        ((index, value[0]) for index, value in enumerate(items)),
        key=lambda item: item[1],
    )
    best_value = ordered[0][1]
    best = {index for index, value in ordered if abs(value - best_value) < 1e-9}
    remaining = [(index, value) for index, value in ordered if index not in best]
    if not remaining:
        return best, set()
    second_value = remaining[0][1]
    second = {
        index for index, value in remaining
        if abs(value - second_value) < 1e-9
    }
    return best, second


def _fmt_md(value, best: bool, second: bool) -> str:
    text = f"{value[0]:.2f} ± {value[1]:.2f}"
    if best:
        return f"**{text}**"
    if second:
        return f"_{text}_"
    return text


def _fmt_tex(value, best: bool, second: bool) -> str:
    text = f"{value[0]:.2f}$_{{\\pm{value[1]:.2f}}}$"
    if best:
        return rf"\textbf{{{text}}}"
    if second:
        return rf"\underline{{{text}}}"
    return text


def _build_markdown(aggregated) -> str:
    output = [
        "# Newly-Added-Sensor Forecast Errors",
        "",
        "Mean ± std over five seeds on sensors whose stable metadata IDs were "
        "absent from the preceding year. Years without a newly added ID are "
        "excluded. **Bold** = best; _underline_ = second best.",
        "",
    ]
    labels = [label for _group, label, _slug in COLUMNS]
    header = "| Metric | Horizon | " + " | ".join(labels) + " |"
    align = "|:------:|:-------:|" + "|".join([":-:"] * len(labels)) + "|"
    for dataset in DATASETS:
        output.extend((f"## {dataset}", "", header, align))
        for metric in METRICS:
            for index, horizon in enumerate(HORIZONS):
                row = [
                    aggregated[(dataset, slug, horizon, metric)]
                    for _group, _label, slug in COLUMNS
                ]
                best, second = _rank(row)
                cells = [
                    _fmt_md(value, col in best, col in second)
                    for col, value in enumerate(row)
                ]
                metric_label = "MAPE (%)" if metric == "MAPE" else metric
                metric_cell = f"**{metric_label}**" if index == 0 else ""
                output.append(
                    f"| {metric_cell} | {horizon} | " + " | ".join(cells) + " |"
                )
        output.append("")
    return "\n".join(output)


def _group_spans():
    spans = []
    for group, _label, _slug in COLUMNS:
        if spans and spans[-1][0] == group:
            spans[-1] = (group, spans[-1][1] + 1)
        else:
            spans.append((group, 1))
    return spans


def _build_latex(datasets, aggregated, part: int) -> str:
    spans = _group_spans()
    column_spec = "ccc|" + "|".join("c" * width for _group, width in spans)
    output = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Newly-added-sensor errors (part {part}/2).}}",
        rf"\label{{tab:tsas_new_sensors_part{part}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
    ]
    group_header = [
        r"\multirow{2}{*}{\textbf{Dataset}}",
        r"\multirow{2}{*}{\textbf{Metric}}",
        r"\multirow{2}{*}{\textbf{Horizon}}",
    ]
    for group, width in spans:
        group_header.append(
            rf"\multicolumn{{{width}}}{{c}}{{\textbf{{{group}}}}}"
        )
    output.append(" & ".join(group_header) + r" \\")

    start = 4
    rules = []
    for _group, width in spans:
        rules.append(rf"\cmidrule(lr){{{start}-{start + width - 1}}}")
        start += width
    output.append(" ".join(rules))
    output.append(
        " & ".join(["", "", ""] + [
            rf"\textbf{{{label}}}" for _group, label, _slug in COLUMNS
        ]) + r" \\"
    )
    output.append(r"\midrule")

    rows_per_dataset = len(METRICS) * len(HORIZONS)
    for dataset_index, dataset in enumerate(datasets):
        for metric_index, metric in enumerate(METRICS):
            for horizon_index, horizon in enumerate(HORIZONS):
                row = [
                    aggregated[(dataset, slug, horizon, metric)]
                    for _group, _label, slug in COLUMNS
                ]
                best, second = _rank(row)
                cells = [
                    _fmt_tex(value, col in best, col in second)
                    for col, value in enumerate(row)
                ]
                dataset_cell = (
                    rf"\multirow{{{rows_per_dataset}}}{{*}}{{\textbf{{{dataset}}}}}"
                    if metric_index == 0 and horizon_index == 0 else ""
                )
                metric_label = r"MAPE (\%)" if metric == "MAPE" else metric
                metric_cell = (
                    rf"\multirow{{{len(HORIZONS)}}}{{*}}{{{metric_label}}}"
                    if horizon_index == 0 else ""
                )
                output.append(
                    " & ".join(
                        [dataset_cell, metric_cell, horizon] + cells
                    ) + r" \\"
                )
            if metric_index < len(METRICS) - 1:
                output.append(rf"\cmidrule(lr){{2-{3 + len(COLUMNS)}}}")
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


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-run", type=Path, required=True,
        help="run root containing datasets/<dataset>/summary.csv",
    )
    parser.add_argument(
        "--stae-run", type=Path, required=True,
        help="frozen-STAEFormer run root in the same summary format",
    )
    parser.add_argument(
        "--a2tta-run", type=Path, action="append", required=True,
        help="A2TTA run root containing detail/*.csv; repeat for backfill roots",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "tables",
    )
    parser.add_argument("--expected-seeds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    values = {}
    values.update(_read_summary_layer(args.baseline_run.resolve(), BASELINE_METHODS))
    values.update(_read_summary_layer(args.stae_run.resolve(), {PLAIN_STAE_METHOD}))
    values.update(_read_a2tta_layer([path.resolve() for path in args.a2tta_run]))
    _validate_coverage(values, args.expected_seeds)
    aggregated = _aggregate(values)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        output_dir / "tsas_new_sensors.md": _build_markdown(aggregated),
        output_dir / "tsas_new_sensors_part1.tex": _build_latex(
            DATASET_PART1, aggregated, 1
        ),
        output_dir / "tsas_new_sensors_part2.tex": _build_latex(
            DATASET_PART2, aggregated, 2
        ),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(f"[write] {path}")


if __name__ == "__main__":
    main()
