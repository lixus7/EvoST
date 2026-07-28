#!/usr/bin/env python3
"""
evaluate_new_sensors.py

Re-run inference on existing checkpoints and report per-year MAE / RMSE / MAPE
restricted to the **newly added sensors**. A sensor is new in year t when its
stable metadata ID is absent from year t-1; yearly matrix row positions are not
assumed to be append-only. Years without new IDs are skipped.

Two CLI entry points:
  (1) Single job  : --conf <path> --seed <N>
  (2) Batched mode: --batch-spec <jobs.json>
      jobs.json = list of {"conf": <path>, "seed": <N>}. All jobs MUST share
      the same dataset (raw_data_path / save_data_path / graph_path /
      begin_year / end_year). Inside the process we load each year's adj +
      test data exactly once and iterate over jobs, so for the typical
      "1 dataset × ~13 methods × 5 seeds = 65 jobs" the .npz file is read
      once instead of 65 times.

Per-job outputs: log/<DS>/<logname>-<seed>/eval_new_sensors.{log,csv}
"""
import argparse
import csv
import json
import os
import os.path as osp
import sys

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "src"))
DEFAULT_SENSOR_METADATA_ROOT = osp.join(ROOT, "data", "sensor_metadata")

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from utils.common_tools import load_json_file, mkdirs
from utils.metric import masked_mae_np, masked_mse_np, masked_mape_np
from src.model.model import (
    TrafficStream_Model, STKEC_Model, EAC_Model, Universal_Model,
    STGNN_Model, DCRNN_Model, ASTGNN_Model, TGCN_Model,
    PECPM_Model, RAP_Model, STTTC_Model,
    GWN_Model, STID_Model, ITRANSFORMER_Model, DLINEAR_Model,
    STNORM_Model, STAEFORMER_Model,
)
from dataer.SpatioTemporalDataset import SpatioTemporalDataset


# Mirrors the `methods` registry assembled in main.py so re-evaluation
# instantiates exactly the architecture each checkpoint was trained with.
METHODS = {
    "TrafficStream": TrafficStream_Model,
    "STKEC": STKEC_Model,
    "EAC": EAC_Model,
    "Universal": Universal_Model,
    "STGNN": STGNN_Model,
    "DCRNN": DCRNN_Model,
    "ASTGNN": ASTGNN_Model,
    "TGCN": TGCN_Model,
    "PECPM": PECPM_Model,
    "RAP": RAP_Model,
    "STTTC": STTTC_Model,
    # Conventional ST baselines (retrain wrappers; method set in conf JSON).
    "GWN": GWN_Model,
    "STID": STID_Model,
    "ITRANSFORMER": ITRANSFORMER_Model,
    "DLINEAR": DLINEAR_Model,
    "STNORM": STNORM_Model,
    "STAEFORMER": STAEFORMER_Model,
}


class _Args:
    """Mutable bag-of-attributes; mirrors argparse.Namespace + JSON merge."""
    pass


def _to_args(conf, seed, gpuid):
    a = _Args()
    for k, v in conf.items():
        setattr(a, k, v)
    a.seed = seed
    a.gpuid = gpuid
    a.device = (
        torch.device("cuda:{}".format(gpuid))
        if torch.cuda.is_available() and gpuid != -1
        else torch.device("cpu")
    )
    if not hasattr(a, "method"):
        # stkec.json / trafficstream.json don't carry a "method" field.
        if getattr(a, "logname", "") == "stkec":
            a.method = "STKEC"
        else:
            a.method = "TrafficStream"
    if not hasattr(a, "backbone_type"):
        a.backbone_type = "stgnn"
    return a


def _instantiate(args):
    cls = METHODS.get(args.method, TrafficStream_Model)
    return cls(args)


def _slice_metrics(truth, pred, sl):
    g = truth[:, sl, :]
    p = pred[:, sl, :]
    if g.size == 0 or g.shape[1] == 0:
        return None
    out = {}
    mae_list, rmse_list, mape_list = [], [], []
    for i in range(1, 13):
        # Cumulative error over the first i horizon steps. Keys "1".."12";
        # "3"/"6"/"12" reproduce the published-table numbers exactly, and
        # plotting "1".."12" gives the cumulative-vs-horizon curve.
        cmae = masked_mae_np(g[:, :, :i], p[:, :, :i], 0)
        crmse = masked_mse_np(g[:, :, :i], p[:, :, :i], 0) ** 0.5
        cmape = masked_mape_np(g[:, :, :i], p[:, :, :i], 0)
        mae_list.append(cmae)
        rmse_list.append(crmse)
        mape_list.append(cmape)
        out[str(i)] = (float(cmae), float(crmse), float(cmape))
        # Per-step error at exactly horizon i (one time slice), for the
        # per-horizon figure ("each time slice", à la STWave Fig. 8).
        smae = masked_mae_np(g[:, :, i - 1:i], p[:, :, i - 1:i], 0)
        srmse = masked_mse_np(g[:, :, i - 1:i], p[:, :, i - 1:i], 0) ** 0.5
        smape = masked_mape_np(g[:, :, i - 1:i], p[:, :, i - 1:i], 0)
        out["step{}".format(i)] = (float(smae), float(srmse), float(smape))
    out["Avg"] = (float(np.mean(mae_list)), float(np.mean(rmse_list)), float(np.mean(mape_list)))
    return out


def _metadata_sensor_ids(metadata_root, dataset, year, expected_nodes):
    """Return stable sensor IDs in the exact local graph-row order."""
    path = osp.join(
        osp.abspath(metadata_root), dataset.lower(), str(year),
        "sensor_metadata.csv",
    )
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"missing sensor metadata for {dataset}-{year}: {path}"
        )
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "sensor_index" not in rows[0]:
        raise RuntimeError(f"invalid sensor metadata: {path}")
    rows.sort(key=lambda row: int(row["sensor_index"]))
    indices = [int(row["sensor_index"]) for row in rows]
    if indices != list(range(expected_nodes)):
        raise RuntimeError(
            f"{path} does not match graph rows 0..{expected_nodes - 1}"
        )
    if "node_id" in rows[0]:
        id_field = "node_id"
    elif "station_id" in rows[0]:
        id_field = "station_id"
    else:
        raise RuntimeError(
            f"{path} must contain either a node_id or station_id column"
        )
    sensor_ids = [str(row[id_field]).strip() for row in rows]
    if (
        len(sensor_ids) != expected_nodes
        or any(not sensor_id for sensor_id in sensor_ids)
        or len(set(sensor_ids)) != expected_nodes
    ):
        raise RuntimeError(
            f"{path} has {len(sensor_ids)} rows/{len(set(sensor_ids))} "
            f"unique IDs; expected {expected_nodes}"
        )
    return sensor_ids


def _build_model(args, year, graph_size_list):
    seed_dir = osp.join(args.model_path, "{}-{}".format(args.logname, args.seed))
    year_dir = osp.join(seed_dir, str(year))
    if not osp.isdir(year_dir):
        return None, None
    # Checkpoint filenames are validation losses. Use the numeric minimum;
    # lexicographic order is wrong whenever digit counts differ (e.g. 9 vs 10).
    pkls = sorted([f for f in os.listdir(year_dir) if f.endswith(".pkl")],
                  key=lambda f: float(f[:-4]))
    if not pkls:
        return None, None
    load_path = osp.join(year_dir, pkls[0])

    state = torch.load(load_path, map_location=args.device)["model_state_dict"]

    if args.method == "STKEC" and "memory" in state:
        args.cluster = state["memory"].shape[0]

    model = _instantiate(args)

    if args.method == "EAC":
        if year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx in range(year - args.begin_year):
                model.expand_adaptive_params(graph_size_list[idx + 1])
    if args.method == "Universal" and getattr(args, "use_eac", False):
        if year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx in range(year - args.begin_year):
                model.expand_adaptive_params(graph_size_list[idx])

    model.load_state_dict(state)
    model = model.to(args.device).eval()
    return model, load_path


# ---------------------------------------------------------------------------
# Batched driver: one process iterates years once, all jobs share that walk.
# ---------------------------------------------------------------------------
def evaluate_batch(
    jobs,
    gpuid,
    num_workers=4,
    batch_size_override=None,
    sensor_metadata_root=DEFAULT_SENSOR_METADATA_ROOT,
    output_stem="eval_new_sensors",
):
    """jobs = list of dicts {"conf": <path>, "seed": <int>} sharing one dataset."""
    if not jobs:
        print("[batch] empty job list", file=sys.stderr)
        return

    # Load and validate per-job configs share dataset-level fields.
    confs = [load_json_file(j["conf"]) for j in jobs]
    common_keys = ("raw_data_path", "save_data_path", "graph_path", "begin_year", "end_year")
    base = {k: confs[0].get(k) for k in common_keys}
    for j, conf in zip(jobs, confs):
        for k in common_keys:
            if conf.get(k) != base[k]:
                raise ValueError(
                    f"job {j['conf']} differs from batch leader in {k}: "
                    f"{conf.get(k)!r} vs {base[k]!r}"
                )

    datasets = {
        str(j.get("_ds") or osp.basename(osp.dirname(j["conf"]))).upper()
        for j in jobs
    }
    if len(datasets) != 1:
        raise ValueError(f"batch must contain one dataset, got {sorted(datasets)}")
    dataset = next(iter(datasets))
    if osp.basename(output_stem) != output_stem or not output_stem:
        raise ValueError(f"output_stem must be a plain filename stem: {output_stem!r}")

    device = (torch.device(f"cuda:{gpuid}")
              if torch.cuda.is_available() and gpuid != -1
              else torch.device("cpu"))

    # Set up per-job state (open log, init args, allocate result accumulators)
    states = []
    for j, conf in zip(jobs, confs):
        args = _to_args(conf, j["seed"], gpuid)
        seed_dir = osp.join(args.model_path, f"{args.logname}-{args.seed}")
        if not osp.isdir(seed_dir):
            print(f"[batch] skip {args.logname} seed={args.seed}: missing {seed_dir}",
                  file=sys.stderr)
            states.append(None)
            continue
        mkdirs(seed_dir)
        out_log = osp.join(seed_dir, f"{output_stem}.log")
        out_csv = osp.join(seed_dir, f"{output_stem}.csv")
        log_f = open(out_log, "w")
        st = {
            "args": args,
            "seed_dir": seed_dir,
            "out_csv": out_csv,
            "log_f": log_f,
            "rows": [],
            "new_metrics": {},
            "all_metrics": {},
        }
        log_f.write(f"[*] conf={j['conf']} seed={j['seed']} gpu={gpuid} dest={seed_dir}\n")
        log_f.write(f"[*] method={args.method} logname={args.logname} "
                    f"years={args.begin_year}-{args.end_year}\n")
        log_f.write(
            f"[*] new_sensor_definition=metadata_id_difference "
            f"metadata_root={osp.abspath(sensor_metadata_root)}\n"
        )
        log_f.flush()
        states.append(st)

    if not any(states):
        print("[batch] no live jobs", file=sys.stderr)
        return

    # Use any live job's args for the common dataset walk
    live_args = next(s["args"] for s in states if s is not None)
    bs = batch_size_override or getattr(live_args, "batch_size", 64)

    graph_size_list = []
    prev_ids = None

    def _broadcast(msg):
        for st in states:
            if st is not None:
                st["log_f"].write(msg + "\n")

    for year in range(live_args.begin_year, live_args.end_year + 1):
        adj_path = osp.join(live_args.graph_path, f"{year}_adj.npz")
        if not osp.exists(adj_path):
            _broadcast(f"  [year {year}] missing {adj_path} — skip")
            continue
        adj_raw = np.load(adj_path)["x"]
        n = adj_raw.shape[0]
        graph_size_list.append(n)
        sensor_ids = _metadata_sensor_ids(
            sensor_metadata_root, dataset, year, n
        )
        previous_ids = prev_ids
        prev_ids = set(sensor_ids)

        if previous_ids is None:
            for st in states:
                if st is not None:
                    st["args"].base_node_size = n
                    st["args"].init_graph_size = n
            _broadcast(f"  [year {year}] begin-year, N={n} (no prior year — skipping)")
            continue

        new_indices = np.asarray(
            [
                idx for idx, sensor_id in enumerate(sensor_ids)
                if sensor_id not in previous_ids
            ],
            dtype=np.int64,
        )
        new_count = int(new_indices.size)
        if new_count == 0:
            _broadcast(f"  [year {year}] N={n} (no newly added IDs — skipping)")
            continue

        data_path = osp.join(live_args.save_data_path, f"{year}.npz")
        if not osp.exists(data_path):
            _broadcast(f"  [year {year}] missing test data {data_path} — skip")
            continue

        # Year-shared state (loaded ONCE for all jobs)
        norm_adj = adj_raw / (np.sum(adj_raw, 1, keepdims=True) + 1e-6)
        adj_tensor = torch.from_numpy(norm_adj).to(torch.float).to(device)
        inputs = np.load(data_path, allow_pickle=True)
        # Materialize test_x / test_y so SpatioTemporalDataset doesn't trigger
        # a fresh npz decompression each time we wrap it in a new DataLoader.
        test_inputs = {"test_x": inputs["test_x"], "test_y": inputs["test_y"]}

        # Iterate jobs — each builds its own model for this year, then infers.
        for st in states:
            if st is None:
                continue
            args = st["args"]
            args.adj = adj_tensor
            args.graph_size = n
            args.year = year

            try:
                model, ckpt = _build_model(args, year, graph_size_list)
            except Exception as e:
                st["log_f"].write(f"  [year {year}] model build/load FAILED: {e}\n")
                continue
            if model is None:
                st["log_f"].write(f"  [year {year}] no checkpoint, skipping\n")
                continue

            loader = DataLoader(
                SpatioTemporalDataset(test_inputs, "test"),
                batch_size=bs, shuffle=False, pin_memory=True, num_workers=num_workers,
            )

            pred_chunks, truth_chunks = [], []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device, non_blocking=True)
                    out = model(data, args.adj)
                    if isinstance(out, tuple):  # STKEC returns (pred, scores)
                        out = out[0]
                    pred_dense, _ = to_dense_batch(out, batch=data.batch)
                    y_dense, _ = to_dense_batch(data.y, batch=data.batch)
                    pred_chunks.append(pred_dense.cpu().numpy())
                    truth_chunks.append(y_dense.cpu().numpy())

            truth = np.concatenate(truth_chunks, 0)
            pred = np.concatenate(pred_chunks, 0)

            if truth.shape[1] != n:
                raise RuntimeError(
                    f"{dataset}-{year}: dense N={truth.shape[1]} != graph/"
                    f"metadata N={n}; refusing to apply an ambiguous sensor mask"
                )
            new_slice = new_indices
            local_new = new_count
            all_slice = slice(0, n)

            nm = _slice_metrics(truth, pred, new_slice)
            am = _slice_metrics(truth, pred, all_slice)
            st["new_metrics"][year] = nm
            st["all_metrics"][year] = am

            st["log_f"].write(
                f"  [year {year}] N={n} new_ids={local_new} "
                f"ckpt={osp.basename(ckpt)}\n"
            )
            for k in ("3", "6", "12", "Avg"):
                st["log_f"].write(
                    f"    h={k:>3} new   MAE={nm[k][0]:>7.4f}  RMSE={nm[k][1]:>7.4f}  MAPE={nm[k][2]:>7.4f}\n"
                )
                st["log_f"].write(
                    f"    h={k:>3} all   MAE={am[k][0]:>7.4f}  RMSE={am[k][1]:>7.4f}  MAPE={am[k][2]:>7.4f}\n"
                )
            # Persist *every* horizon key (cumulative "1".."12", per-step
            # "step1".."step12", "Avg"); am/nm share the same key set.
            for k in am:
                st["rows"].append([year, n, local_new, "new", k, *nm[k]])
                st["rows"].append([year, n, local_new, "all", k, *am[k]])

            del model

        del inputs, test_inputs, adj_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Finalize each job's log + csv
    for st in states:
        if st is None:
            continue
        log_f = st["log_f"]
        log_f.write("\n========== Newly added sensors — aggregate ==========\n")
        log_f.write(f"years evaluated: {sorted(st['new_metrics'].keys())}\n")
        if st["new_metrics"]:
            log_f.write("h     MAE       RMSE      MAPE\n")
            for k in ("3", "6", "12", "Avg"):
                vs = list(st["new_metrics"].values())
                ms = [v[k][0] for v in vs]
                rs = [v[k][1] for v in vs]
                ps = [v[k][2] for v in vs]
                log_f.write(
                    f"{k:>3}   {np.mean(ms):>7.4f}  {np.mean(rs):>7.4f}  {np.mean(ps):>7.4f}\n"
                )

        log_f.write("\n========== All sensors (recomputed) ==========\n")
        if st["all_metrics"]:
            log_f.write("h     MAE       RMSE      MAPE\n")
            for k in ("3", "6", "12", "Avg"):
                vs = list(st["all_metrics"].values())
                ms = [v[k][0] for v in vs]
                rs = [v[k][1] for v in vs]
                ps = [v[k][2] for v in vs]
                log_f.write(
                    f"{k:>3}   {np.mean(ms):>7.4f}  {np.mean(rs):>7.4f}  {np.mean(ps):>7.4f}\n"
                )

        with open(st["out_csv"], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "n_total", "n_new", "scope", "horizon", "MAE", "RMSE", "MAPE"])
            w.writerows(st["rows"])
        log_f.write(f"[*] wrote {st['out_csv']}\n")
        log_f.close()


def evaluate(
    conf_path,
    seed,
    gpuid,
    num_workers=4,
    batch_size_override=None,
    sensor_metadata_root=DEFAULT_SENSOR_METADATA_ROOT,
    output_stem="eval_new_sensors",
):
    """Back-compat single-job entry point — thin wrapper around evaluate_batch."""
    evaluate_batch(
        [{
            "conf": conf_path,
            "seed": seed,
            "_ds": osp.basename(osp.dirname(conf_path)),
        }],
        gpuid,
        num_workers=num_workers,
        batch_size_override=batch_size_override,
        sensor_metadata_root=sensor_metadata_root,
        output_stem=output_stem,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default=None, help="single-job conf path")
    parser.add_argument("--seed", type=int, default=None, help="single-job seed")
    parser.add_argument("--batch-spec", default=None,
                        help="JSON file: list of {conf, seed} jobs sharing one dataset")
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--sensor_metadata_root",
        default=DEFAULT_SENSOR_METADATA_ROOT,
        help="root containing <dataset>/<year>/sensor_metadata.csv",
    )
    parser.add_argument(
        "--output_stem",
        default="eval_new_sensors",
        help="per-checkpoint output filename stem",
    )
    a = parser.parse_args()

    if a.batch_spec:
        with open(a.batch_spec) as f:
            jobs = json.load(f)
        evaluate_batch(jobs, a.gpuid,
                       num_workers=a.num_workers,
                       batch_size_override=a.batch_size,
                       sensor_metadata_root=a.sensor_metadata_root,
                       output_stem=a.output_stem)
    elif a.conf is not None and a.seed is not None:
        evaluate(a.conf, a.seed, a.gpuid,
                 num_workers=a.num_workers,
                 batch_size_override=a.batch_size,
                 sensor_metadata_root=a.sensor_metadata_root,
                 output_stem=a.output_stem)
    else:
        parser.error("provide either --batch-spec <json>, or --conf X --seed Y")
