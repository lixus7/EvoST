"""
A2TTA: Anchored-and-Agile Test-Time Adaptation for Traffic Forecasting.

Entry point that orchestrates per-year evaluation:
  * loads a frozen backbone checkpoint (default: Online-AN per-year .pkls)
  * builds a residual calibrator (zero-init) shared across years
  * optionally warms up the calibrator on the year's train+val split
  * runs chronological online TTA on the year's test split with delayed labels
  * logs metrics via the same `cal_metric` used by main.py and writes a CSV

Usage:
  python a2tta_main.py --conf conf/PEMS05/a2tta_olan_pems05.json \
      --method tta_ctx_local --seed 51 --gpuid 0 \
      --backbone_ckpt_logname oneline_st_an_pems05

See `scripts/a2tta_lite_pems05_run.sh` for the full experiment matrix.
"""
from __future__ import annotations

import argparse
import csv
import os
import os.path as osp
import sys
import time

sys.path.append("src/")

import numpy as np
import networkx as nx
import torch

from utils.data_convert import generate_samples
from utils.initialize import init, seed_anything, init_log
from utils.common_tools import mkdirs

from src.model.model import (
    TrafficStream_Model, STKEC_Model, EAC_Model, Universal_Model,
    STGNN_Model, DCRNN_Model, ASTGNN_Model, TGCN_Model,
    PECPM_Model, RAP_Model, STTTC_Model, STAEFORMER_Model,
)
from src.model.a2tta import ResidualCalibrator
from src.trainer.a2tta_trainer import (
    warmup_calibrator, online_a2tta_eval, append_results_csv,
)
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from torch_geometric.loader import DataLoader


METHOD_REGISTRY = {
    'TrafficStream': TrafficStream_Model,
    'STKEC': STKEC_Model,
    'EAC': EAC_Model,
    'Universal': Universal_Model,
    'STGNN': STGNN_Model,
    'DCRNN': DCRNN_Model,
    'ASTGNN': ASTGNN_Model,
    'TGCN': TGCN_Model,
    'PECPM': PECPM_Model,
    'RAP': RAP_Model,
    'STTTC': STTTC_Model,
    'STAEFORMER': STAEFORMER_Model,
}


# ---------------------------------------------------------------------------
# Leakage-free inputs and stable sensor identities
# ---------------------------------------------------------------------------

def _window_input_stats(raw_split: np.ndarray, x_len: int, y_len: int):
    """Mean/std over the exact input windows without materialising them.

    ``generate_dataset`` emits ``len(raw_split) - x_len - y_len`` overlapping
    windows in reverse order.  Window order does not affect the statistics, so
    temporal occurrence counts are used as weights here.
    """
    n_windows = len(raw_split) - x_len - y_len
    if n_windows <= 0:
        raise ValueError(
            f"split length {len(raw_split)} is too short for x={x_len}, y={y_len}"
        )
    weights = np.convolve(
        np.ones(n_windows, dtype=np.float64),
        np.ones(x_len, dtype=np.float64),
    )
    used = np.asarray(raw_split[: len(weights)], dtype=np.float64)
    finite = np.isfinite(used)
    weighted = weights[:, None]
    count = float((weighted * finite).sum())
    if count <= 0:
        return 0.0, 1.0
    total = float((np.where(finite, used, 0.0) * weighted).sum())
    total_sq = float((np.where(finite, used * used, 0.0) * weighted).sum())
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    std = var ** 0.5
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return mean, std


def _raw_input_windows(raw_split: np.ndarray, x_len: int, y_len: int):
    """Recreate ``generate_dataset(...)[0]`` in its latest-first order."""
    n_windows = len(raw_split) - x_len - y_len
    if n_windows <= 0:
        raise ValueError(
            f"split length {len(raw_split)} is too short for x={x_len}, y={y_len}"
        )
    starts = np.arange(n_windows - 1, -1, -1, dtype=np.int64)
    indices = starts[:, None] + np.arange(x_len, dtype=np.int64)[None, :]
    return raw_split[indices]


def _check_cached_target_anchors(cached_y, raw_split, x_len: int, y_len: int, split: str):
    """Catch split/order mismatches before replacing cached input features."""
    expected_first = np.nan_to_num(
        raw_split[len(raw_split) - 1 - y_len : len(raw_split) - 1],
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    expected_last = np.nan_to_num(
        raw_split[x_len : x_len + y_len],
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    if (
        cached_y.shape[0] == 0
        or not np.allclose(cached_y[0], expected_first, rtol=0.0, atol=1e-5)
        or not np.allclose(cached_y[-1], expected_last, rtol=0.0, atol=1e-5)
    ):
        raise RuntimeError(
            f"{split} RawData/FastData window order mismatch; refusing to "
            "apply training-stat normalization"
        )


def _load_inputs(args, year: int, graph):
    """Load cached labels/graph and optionally rebuild x with train statistics.

    Existing FastData files normalised train/val/test independently.  In
    ``train`` mode, the exact raw val/test windows are reconstructed and scaled
    with statistics fitted only on the training windows.  Original FastData is
    read-only and never overwritten.
    """
    fast_path = osp.join(args.save_data_path, f"{year}.npz")
    mode = str(args.a2tta.get("input_normalization", "train"))
    if args.data_process:
        generated = generate_samples(
            31,
            osp.join(args.save_data_path, str(year)),
            np.load(osp.join(args.raw_data_path, f"{year}.npz"))["x"],
            graph,
            val_test_mix=False,
        )
        args._input_normalization_stats = {
            "mode": "train",
            "mean": float(generated["input_mean"]),
            "std": float(generated["input_std"]),
        }
        return generated
    if mode == "cached_split":
        args._input_normalization_stats = {
            "mode": mode, "mean": float("nan"), "std": float("nan"),
        }
        return np.load(fast_path, allow_pickle=True)
    if mode != "train":
        raise ValueError(f"unknown input_normalization={mode!r}")

    with np.load(fast_path, allow_pickle=True) as cached_npz:
        cached = {key: cached_npz[key] for key in cached_npz.files}
    with np.load(osp.join(args.raw_data_path, f"{year}.npz")) as raw_npz:
        raw = np.asarray(raw_npz["x"])

    span = int(args.x_len + args.y_len)
    split_lengths = {
        split: int(cached[f"{split}_x"].shape[0]) + span
        for split in ("train", "val", "test")
    }
    used_steps = sum(split_lengths.values())
    if raw.shape[0] < used_steps:
        raise RuntimeError(
            f"RawData/{year}.npz has {raw.shape[0]} rows but cached splits "
            f"require {used_steps}"
        )
    raw = raw[:used_steps]
    if raw.shape[1] != graph.number_of_nodes():
        raise RuntimeError(
            f"RawData nodes={raw.shape[1]} != graph nodes={graph.number_of_nodes()} "
            f"for year={year}"
        )

    raw_splits = {}
    offset = 0
    for split in ("train", "val", "test"):
        length = split_lengths[split]
        raw_splits[split] = raw[offset : offset + length]
        offset += length
        _check_cached_target_anchors(
            cached[f"{split}_y"], raw_splits[split],
            int(args.x_len), int(args.y_len), split,
        )

    train_mean, train_std = _window_input_stats(
        raw_splits["train"], int(args.x_len), int(args.y_len)
    )
    # Rebuild every input split from RawData.  Some legacy FastData archives
    # were standardised with float32 reductions, whose small rounding error can
    # make an exact cached-value check fail on large graphs (for example,
    # PEMS04-2002).  Target anchors above already verify split boundaries and
    # window order; rebuilding train/val/test here gives all three splits the
    # same training-only statistics without trusting cached input scaling.
    for split in ("train", "val", "test"):
        raw_x = _raw_input_windows(
            raw_splits[split], int(args.x_len), int(args.y_len)
        )
        scaled = (raw_x - train_mean) / train_std
        cached[f"{split}_x"] = np.nan_to_num(
            scaled, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
    args._input_normalization_stats = {
        "mode": mode, "mean": float(train_mean), "std": float(train_std),
    }
    return cached


def _year_sensor_ids(args, year: int, expected_nodes: int):
    """Read current local-node order from the exported metadata sidecar."""
    root = str(args.a2tta.get("sensor_metadata_root", "data/sensor_metadata"))
    path = osp.join(root, str(args.dataset).lower(), str(year), "sensor_metadata.csv")
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"sensor metadata missing for stable-ID mapping: {path}"
        )
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "sensor_index" not in rows[0]:
        raise RuntimeError(f"invalid sensor metadata: {path}")
    rows.sort(key=lambda row: int(row["sensor_index"]))
    indices = [int(row["sensor_index"]) for row in rows]
    if indices != list(range(expected_nodes)):
        raise RuntimeError(
            f"{path} does not match graph order: expected indices 0..{expected_nodes - 1}"
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
            f"{path} has {len(sensor_ids)} rows/{len(set(sensor_ids))} unique IDs; "
            f"expected {expected_nodes}"
        )
    return sensor_ids


def _set_year_sensor_mapping(args, year: int, num_nodes: int):
    """Map local graph rows to persistent embedding rows keyed by sensor ID."""
    mode = str(args.a2tta.get("sensor_id_mode", "metadata"))
    if mode == "local_index":
        global_indices = list(range(num_nodes))
        for idx in range(len(args._sensor_id_registry), num_nodes):
            args._sensor_id_registry[f"local::{idx}"] = idx
        previous_size = getattr(args, "_previous_active_sensor_count", None)
        new_local_indices = (
            [] if previous_size is None
            else list(range(min(previous_size, num_nodes), num_nodes))
        )
        vars(args)["_previous_active_sensor_count"] = int(num_nodes)
    elif mode == "metadata":
        sensor_ids = _year_sensor_ids(args, year, num_nodes)
        previous_ids = getattr(args, "_previous_active_sensor_ids", None)
        new_local_indices = (
            [] if previous_ids is None
            else [
                idx for idx, sensor_id in enumerate(sensor_ids)
                if sensor_id not in previous_ids
            ]
        )
        global_indices = []
        for sensor_id in sensor_ids:
            key = f"{str(args.dataset).lower()}::{sensor_id}"
            if key not in args._sensor_id_registry:
                args._sensor_id_registry[key] = len(args._sensor_id_registry)
            global_indices.append(args._sensor_id_registry[key])
        vars(args)["_previous_active_sensor_ids"] = set(sensor_ids)
    else:
        raise ValueError(f"unknown sensor_id_mode={mode!r}")

    vars(args)["node_global_idx"] = torch.tensor(
        global_indices, dtype=torch.long, device=args.device
    )
    vars(args)["sensor_new_local_indices"] = new_local_indices
    vars(args)["sensor_new_count"] = int(len(new_local_indices))
    vars(args)["sensor_registry_size"] = int(len(args._sensor_id_registry))
    return len(args._sensor_id_registry)


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def _load_backbone_for_year(args, year: int):
    """Load the per-year backbone checkpoint produced by `oneline_st_an_pems05`
    (or the configured logname). Falls back to retrain ckpt if Online-AN is
    missing. Returns a torch.nn.Module on `args.device`.
    """
    backbone_method = getattr(args, "backbone_method", "TrafficStream")
    logname_primary = args.backbone_ckpt_logname
    logname_fallback = getattr(args, "backbone_ckpt_logname_fallback", "retrain_st_pems05")
    # Backbone ckpts may live under a different seed than the TTA seed (e.g.
    # STAEFormer retrain used seeds 42-46 while A2TTA runs 51-55). Default -1
    # keeps the legacy behaviour (same seed as the TTA run).
    backbone_seed = int(getattr(args, "backbone_seed", -1))
    if backbone_seed < 0:
        backbone_seed = args.seed

    # Set args.method for the backbone factory; restore after.
    saved_method = getattr(args, "method", None)
    saved_year = getattr(args, "year", None)
    vars(args)["method"] = backbone_method
    vars(args)["year"] = year

    def _try(logname):
        ckpt_dir = osp.join(args.model_path, f"{logname}-{backbone_seed}", str(year))
        if not osp.isdir(ckpt_dir):
            return None
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pkl")]
        if not files:
            return None
        # Min loss in filename (matches load_best_model logic).
        files = sorted(files, key=lambda f: float(f[:-4]))
        return osp.join(ckpt_dir, files[0])

    ckpt = _try(logname_primary) or _try(logname_fallback)
    if ckpt is None:
        raise FileNotFoundError(
            f"No backbone .pkl for year={year} under "
            f"{logname_primary}-{backbone_seed}/ or {logname_fallback}-{backbone_seed}/"
        )

    args.logger.info(f"  [backbone] year={year} loading {ckpt}")
    state = torch.load(ckpt, map_location=args.device)["model_state_dict"]
    model = METHOD_REGISTRY[backbone_method](args).to(args.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_matched = len(state) - len(unexpected)
    if n_matched == 0:
        # strict=False would otherwise leave a randomly-initialized model and
        # produce silent garbage (e.g. a TrafficStream ckpt fed to STAEFORMER).
        raise RuntimeError(
            f"Backbone ckpt {ckpt} shares NO keys with {backbone_method} "
            f"(missing={len(missing)}, unexpected={len(unexpected)}) — wrong "
            f"--backbone_ckpt_logname / --backbone_method pairing?"
        )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Restore.
    vars(args)["method"] = saved_method
    vars(args)["year"] = saved_year
    return model


# ---------------------------------------------------------------------------
# Per-year orchestration
# ---------------------------------------------------------------------------

def _run_year(args, year: int, calibrator: ResidualCalibrator):
    cfg = args.a2tta
    profile_path = str(cfg.get("profile_path", "") or "")
    profiling = bool(profile_path)

    def _sync():
        if profiling and args.device.type == "cuda":
            torch.cuda.synchronize(args.device)

    def _start():
        _sync()
        return time.perf_counter()

    def _elapsed(t0):
        _sync()
        return time.perf_counter() - t0

    if profiling and args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    t_year = _start()

    # --- Graph + adjacency ---
    t0 = _start()
    graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, f"{year}_adj.npz"))["x"])
    if year == args.begin_year:
        vars(args)["base_node_size"] = graph.number_of_nodes()
        vars(args)["init_graph_size"] = graph.number_of_nodes()
    vars(args)["graph_size"] = graph.number_of_nodes()
    vars(args)["year"] = year
    registry_size = _set_year_sensor_mapping(
        args, year, graph.number_of_nodes()
    )

    adj = np.load(osp.join(args.graph_path, f"{year}_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

    # --- Inputs (preprocessed FastData / RawData) ---
    inputs = _load_inputs(args, year, graph)
    norm_info = getattr(args, "_input_normalization_stats", {
        "mode": cfg.get("input_normalization", "cached_split"),
        "mean": float("nan"), "std": float("nan"),
    })
    args.logger.info(
        f"[*] Year {year} loaded input_norm={norm_info['mode']} "
        f"train_mean={norm_info['mean']:.6g} train_std={norm_info['std']:.6g} "
        f"sensor_ids={cfg.get('sensor_id_mode', 'metadata')} "
        f"new={args.sensor_new_count} registry={args.sensor_registry_size}"
    )
    input_load_seconds = _elapsed(t0)

    # --- Backbone (frozen) ---
    t0 = _start()
    backbone = _load_backbone_for_year(args, year)
    backbone_load_seconds = _elapsed(t0)

    # --- Calibrator (grow stable sensor-ID embedding registry if needed) ---
    calibrator.expand_nodes(registry_size)
    calibrator = calibrator.to(args.device)

    # --- Optional warm-up ---
    t0 = _start()
    if cfg.get("warmup_epochs", 0) > 0 and cfg["method"] != "backbone":
        train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"),
                                  batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=0)
        val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"),
                                batch_size=args.batch_size, shuffle=False,
                                pin_memory=True, num_workers=0)
        warmup_calibrator(
            backbone=backbone, calibrator=calibrator,
            train_loader=train_loader, val_loader=val_loader,
            args=args, epochs=cfg["warmup_epochs"], lr=cfg.get("warmup_lr", 1e-3),
        )
    warmup_seconds = _elapsed(t0)

    # --- Online evaluation ---
    t0 = _start()
    online_a2tta_eval(backbone, calibrator, inputs, args)
    eval_seconds = _elapsed(t0)

    if profiling:
        profile = dict(getattr(args, "_a2tta_last_profile", {}) or {})
        trainable_total, head_params, _ = calibrator.param_report()
        host_current_rss_mb, host_peak_rss_mb = _process_memory_mb()
        profile.update({
            "profile_tag": cfg.get("profile_tag", ""),
            "profile_repeat": int(cfg.get("profile_repeat", 0)),
            "dataset": cfg.get("dataset", args.dataset),
            "backbone_method": getattr(args, "backbone_method", ""),
            "method": cfg.get("method", ""),
            "seed": int(args.seed),
            "year": int(year),
            "num_nodes": int(graph.number_of_nodes()),
            "input_load_seconds": input_load_seconds,
            "backbone_load_seconds": backbone_load_seconds,
            "warmup_seconds": warmup_seconds,
            "eval_seconds": eval_seconds,
            "year_total_seconds": _elapsed(t_year),
            "calibrator_trainable_params": int(trainable_total),
            "calibrator_head_params": int(head_params),
            "backbone_params": int(sum(p.numel() for p in backbone.parameters())),
            "adapt_lr": float(cfg.get("adapt_lr", 0.0)),
            "adapt_steps": int(cfg.get("adapt_steps", 0)),
            "adapt_every_windows": (
                int(cfg.get("adapt_every_windows", -1))
                if int(cfg.get("adapt_every_windows", -1)) > 0
                else int(cfg.get("adapt_every_batches", 1))
                * int(cfg.get("eval_batch_size", 1))
            ),
            "local_steps": int(cfg.get("local_steps", 0)),
            "eval_batch_size": int(cfg.get("eval_batch_size", 0)),
            "input_normalization": cfg.get("input_normalization", ""),
            "sensor_id_mode": cfg.get("sensor_id_mode", ""),
            "sensor_registry_size": int(getattr(args, "sensor_registry_size", 0)),
            "sensor_new_count": int(getattr(args, "sensor_new_count", 0)),
            "samples_per_day": int(cfg.get("samples_per_day", 0)),
            "gpu_name": (torch.cuda.get_device_name(args.device)
                         if args.device.type == "cuda" else "cpu"),
            "peak_gpu_allocated_mb": (torch.cuda.max_memory_allocated(args.device) / 2**20
                                      if args.device.type == "cuda" else 0.0),
            "peak_gpu_reserved_mb": (torch.cuda.max_memory_reserved(args.device) / 2**20
                                     if args.device.type == "cuda" else 0.0),
            "host_current_rss_mb": host_current_rss_mb,
            "host_peak_rss_mb": host_peak_rss_mb,
        })
        _append_profile_csv(profile, profile_path)

    # Free per-year backbone.
    del backbone
    torch.cuda.empty_cache()


def _process_memory_mb():
    """Return Linux process (current RSS, lifetime high-water RSS) in MiB."""
    values = {"VmRSS": 0.0, "VmHWM": 0.0}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key = line.split(":", 1)[0]
                if key in values:
                    values[key] = float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return values["VmRSS"], values["VmHWM"]


PROFILE_COLUMNS = [
    "profile_tag", "profile_repeat", "dataset", "backbone_method", "method",
    "seed", "year", "num_nodes", "num_windows", "num_batches",
    "label_delay_steps", "label_delay_is_causal", "eval_batch_size",
    "input_normalization", "sensor_id_mode", "sensor_registry_size",
    "sensor_new_count", "samples_per_day",
    "label_keep_prob", "label_noise_std", "label_spike_prob",
    "label_spike_scale", "feedback_target_scale", "feedback_seen",
    "feedback_kept", "feedback_keep_rate_actual",
    "adapt_lr", "adapt_steps", "adapt_every_windows", "local_steps",
    "adapt_steps_total",
    "calibrator_trainable_params", "calibrator_head_params", "backbone_params",
    "input_load_seconds", "backbone_load_seconds", "warmup_seconds",
    "eval_seconds", "year_total_seconds", "data_transfer_seconds",
    "queue_seconds", "global_adapt_seconds", "backbone_infer_seconds",
    "local_clone_seconds", "calibrator_predict_seconds", "output_seconds",
    "metric_seconds", "peak_gpu_allocated_mb", "peak_gpu_reserved_mb", "gpu_name",
    "host_current_rss_mb", "host_peak_rss_mb",
]


def _append_profile_csv(row, path: str):
    """Append one per-year efficiency record to a job-local CSV."""
    path = path if osp.isabs(path) else osp.join(osp.dirname(__file__), path)
    os.makedirs(osp.dirname(path), exist_ok=True)
    write_header = not osp.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROFILE_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in PROFILE_COLUMNS})


def main(args):
    args.logger.info("params : %s", vars(args))
    args.result = {
        "3":   {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "6":   {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "12":  {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "Avg": {" MAE": {}, "MAPE": {}, "RMSE": {}},
    }
    mkdirs(args.save_data_path)

    # Build a calibrator once and let it persist (and grow) across years.
    cfg = args.a2tta
    # Stable identities are registered only when they first appear, so no
    # future-year sensor information is consumed by the current year.
    vars(args)["_sensor_id_registry"] = {}
    # Initialize with the year-0 graph size; will expand if later years grow.
    first_graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, f"{args.begin_year}_adj.npz"))["x"])
    calibrator = ResidualCalibrator(
        num_nodes_max=first_graph.number_of_nodes(),
        x_len=args.x_len, y_len=args.y_len,
        node_emb_dim=cfg.get("node_emb_dim", 16),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("calibrator_dropout", 0.1),
        horizon_emb_dim=cfg.get("horizon_emb_dim", 0),
        arch=cfg.get("calibrator_arch", "film"),
        rank=cfg.get("adapter_rank", 8),
        gate_type=cfg.get("gate_type", "channel"),
        temporal_kernel=cfg.get("temporal_kernel", 3),
    ).to(args.device)

    # Trainable-parameter report (ablation fairness): total, arch-head (excl.
    # node embedding), and the full list of trainable parameter names.
    _tot, _head, _named = calibrator.param_report()
    args.logger.info(
        f"[calibrator] arch={cfg.get('calibrator_arch','film')} "
        f"rank={cfg.get('adapter_rank',8)} trainable_total={_tot} "
        f"head_params(excl node_emb)={_head}")
    args.logger.info(f"[calibrator] trainable params: {_named}")

    # Track graph sizes per year (mirrors main.py's args.graph_size_list).
    vars(args)["graph_size_list"] = []

    for year in range(args.begin_year, args.end_year + 1):
        try:
            _run_year(args, year, calibrator)
        except FileNotFoundError as e:
            args.logger.warning(f"[skip] year {year}: {e}")
            continue
        args.graph_size_list.append(args.graph_size)
        if cfg.get("fast_dev_run", False):
            args.logger.info("[fast_dev_run] stopping after first year")
            break

    # Pretty-print final table (mirrors main.py).
    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info, vals = "", []
            for year in range(args.begin_year, args.end_year + 1):
                if year in args.result[i][j]:
                    info += "{:>10.2f}\t".format(args.result[i][j][year])
                    vals.append(args.result[i][j][year])
            if vals:
                args.logger.info("{:<4}\t{}\t".format(i, j) + info + "\t{:>8.2f}".format(np.mean(vals)))

    # Persist per-year per-horizon rows to CSV.
    csv_path = cfg.get("csv_path", "run_logs/a2tta_lite_results.csv")
    csv_path = osp.join(osp.dirname(__file__), csv_path) if not osp.isabs(csv_path) else csv_path
    append_results_csv(args, csv_path)
    args.logger.info(f"[a2tta] CSV → {csv_path}")


def _build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--conf", type=str, default="conf/PEMS05/a2tta_olan_pems05.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpuid", type=int, default=1)
    p.add_argument("--logname", type=str, default="a2tta_lite_pems05")
    p.add_argument("--method", type=str, default="tta_ctx_local",
                   choices=["backbone", "calibrator",
                            "tta_random", "tta_recent", "tta_error",
                            "tta_all", "tta_ctx_local",
                            "a2tta_lite", "a2tta_emb"],
                   help=("Which variant to run. OURS DEFAULT = tta_ctx_local. "
                         "backbone=frozen no-cal baseline; calibrator=warmed-up cal, no "
                         "online TTA; tta_random/recent/error=online-TTA selection "
                         "baselines; tta_all=adapt on all delayed labels (no selection); "
                         "tta_ctx_local=Ours, global tta_all calibrator + discardable "
                         "per-window local context clone; a2tta_lite/a2tta_emb=legacy. "
                         "(Other explored selection ideas are archived in a2tta_back.py.)"))
    p.add_argument("--dataset", type=str, default="PEMS05")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional explicit per-year checkpoint root, ignored when "
                        "--backbone_ckpt_logname is sufficient.")
    p.add_argument("--backbone_ckpt_logname", type=str, default="oneline_st_an_pems05",
                   help="Logname under args.model_path/<logname>-<seed>/<year>/*.pkl")
    p.add_argument("--backbone_ckpt_logname_fallback", type=str, default="retrain_st_pems05")
    p.add_argument("--backbone_method", type=str, default="TrafficStream",
                   choices=list(METHOD_REGISTRY.keys()))
    p.add_argument("--backbone_seed", type=int, default=-1,
                   help="Seed suffix of the backbone ckpt dir (e.g. 42 for "
                        "retrain_staeformer_*-42 while the TTA run uses 51). "
                        "-1 = use --seed (legacy behaviour).")
    p.add_argument("--freeze_backbone", type=int, default=1)

    # Calibrator hyperparams
    p.add_argument("--adapter_hidden_dim", type=int, default=64)
    p.add_argument("--node_emb_dim", type=int, default=16)
    p.add_argument("--calibrator_dropout", type=float, default=0.1)
    p.add_argument("--calibrator_arch", type=str, default="film",
                   choices=["residual", "norm", "film", "affine", "adapter",
                            "lowrank", "gated_adapter", "temporal_conv"],
                   help=("Calibrator architecture (structural ablation). OURS = film. "
                         "residual=raw-input MLP residual (legacy); norm=EMA-standardised "
                         "MLP residual; film=conditional per-sample affine gamma*y+beta "
                         "(gamma/beta from an MLP, init identity); affine=STATIC "
                         "channel-wise (1+gamma)*y+beta (no conditioning net); "
                         "adapter=bottleneck residual y+Wup(GELU(Wdown)); lowrank=linear "
                         "low-rank residual y+U(V); gated_adapter=adapter with tanh gate; "
                         "temporal_conv=depthwise 1-D conv over the prediction horizon. "
                         "All are identity at init."))
    p.add_argument("--adapter_rank", type=int, default=8,
                   help="Bottleneck/low-rank dim for adapter/lowrank/gated_adapter.")
    p.add_argument("--gate_type", type=str, default="channel", choices=["scalar", "channel"],
                   help="gated_adapter gate granularity (default channel-wise over horizon).")
    p.add_argument("--temporal_kernel", type=int, default=3,
                   help="Kernel size for temporal_conv (depthwise over horizon).")
    # Ours selection (tta_ctx_local): per-window local context clone.
    p.add_argument("--local_steps", type=int, default=3,
                   help="tta_ctx_local: gradient steps for the per-window local clone.")
    p.add_argument("--horizon_emb_dim", type=int, default=0,
                   help="If >0, add a per-horizon embedding to the calibrator "
                        "(STAEFormer-inspired). 0 = legacy calibrator.")

    # a2tta_emb knobs (only used when --method a2tta_emb)
    p.add_argument("--emb_lr", type=float, default=-1.0,
                   help="LR for adaptive-embedding rows; -1 = use --adapt_lr.")
    p.add_argument("--node_budget_frac", type=float, default=0.25,
                   help="Fraction of nodes whose embedding rows may update.")
    p.add_argument("--emb_batch_cap", type=int, default=16,
                   help="Max pool samples per fresh backbone forward (memory cap).")
    p.add_argument("--lambda_reg_emb", type=float, default=-1.0,
                   help="Proximal reg for embedding rows; -1 = use --lambda_reg.")

    # Online TTA hyperparams
    p.add_argument("--adapt_lr", type=float, default=1e-3)    # Ours tuned default
    p.add_argument("--adapt_steps", type=int, default=3)      # Ours tuned default
    p.add_argument("--adapt_every_batches", type=int, default=1)
    p.add_argument(
        "--adapt_every_windows",
        type=int,
        default=64,
        help=("Absolute number of chronological prediction windows between "
              "global adaptation events (default: 64). Set -1 only to recover "
              "the legacy adapt_every_batches * eval_batch_size cadence."),
    )
    p.add_argument("--label_delay_steps", type=int, default=-1,
                   help=("Absolute number of chronological prediction windows before a "
                         "sample's full target is released. -1 = forecast length H "
                         "(legacy/default). Values below H are non-causal diagnostics."))
    p.add_argument("--label_keep_prob", type=float, default=1.0,
                   help=("Probability that a matured feedback window is observed. "
                         "Missing feedback is never inserted into the TTA pool."))
    p.add_argument("--label_noise_std", type=float, default=0.0,
                   help=("Gaussian feedback-noise standard deviation as a fraction of "
                         "the current year's training-target standard deviation."))
    p.add_argument("--label_spike_prob", type=float, default=0.0,
                   help="Element-wise probability of an impulsive feedback-label error.")
    p.add_argument("--label_spike_scale", type=float, default=5.0,
                   help=("Absolute impulse magnitude in multiples of the current year's "
                         "training-target standard deviation; signs are random."))
    p.add_argument("--budget_frac", type=float, default=0.25)
    p.add_argument("--candidate_pool_size", type=int, default=512)
    p.add_argument("--lambda_cons", type=float, default=0.05)
    p.add_argument("--lambda_reg", type=float, default=1e-4)
    p.add_argument("--mc_K", type=int, default=4)

    # Active scoring weights
    p.add_argument("--w_err", type=float, default=1.0)
    p.add_argument("--w_unc", type=float, default=0.3)
    p.add_argument("--w_shift", type=float, default=0.3)
    p.add_argument("--w_recency", type=float, default=0.1)

    # Warm-up
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--warmup_lr", type=float, default=1e-3)

    # Eval-time loader / CSV / dev knobs
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help=("Logical evaluation batch size. The full tta_ctx_local method "
              "requires 1 to preserve strict window-by-window causality."),
    )
    p.add_argument(
        "--input_normalization",
        type=str,
        default="train",
        choices=["train", "cached_split"],
        help=("Input scaling for A2TTA evaluation. train=reconstruct val/test "
              "windows from RawData and reuse training-split mean/std "
              "(default, leakage-free); cached_split=legacy FastData scaling."),
    )
    p.add_argument(
        "--sensor_id_mode",
        type=str,
        default="metadata",
        choices=["metadata", "local_index"],
        help=("Embedding lookup identity. metadata=stable real sensor IDs "
              "across years (default); local_index=legacy yearly row indices."),
    )
    p.add_argument(
        "--sensor_metadata_root",
        type=str,
        default="data/sensor_metadata",
        help=("Root containing <dataset>/<year>/sensor_metadata.csv in the "
              "same local-node order as RawData/graph."),
    )
    p.add_argument(
        "--samples_per_day",
        type=int,
        default=-1,
        help="-1 auto-detects 24 for TFNSW and 288 for five-minute PEMS.",
    )
    p.add_argument("--infer_chunk", type=int, default=16,
                   help="Internal sub-batch for STAEFORMER backbone inference "
                        "(memory only; logical batch / TTA cadence unchanged). "
                        "Lower for dense graphs: PEMS07→2, PEMS04/12→8.")
    p.add_argument("--csv_path", type=str, default="run_logs/a2tta_lite_results.csv")
    p.add_argument(
        "--detail_csv_path",
        type=str,
        default="",
        help=("Optional per-year all/new-sensor metrics at cumulative horizons "
              "1--12, exact steps 1--12, and Avg. Empty disables export."),
    )
    p.add_argument("--profile_path", type=str, default="",
                   help="Optional per-year efficiency CSV. Empty disables phase profiling.")
    p.add_argument("--profile_tag", type=str, default="")
    p.add_argument("--profile_repeat", type=int, default=0)
    p.add_argument(
        "--visualization_root",
        type=str,
        default="",
        help=("If set, save chronological [window,node,horizon] predictions to "
              "<root>/<dataset>/<year>/predictions/a2tta_stae.npy and verify "
              "the emitted targets against ground_truth.npy."),
    )
    p.add_argument("--fast_dev_run", type=int, default=0)
    return p


def _stash_a2tta_cfg(args):
    """Pull all A2TTA knobs into a single ``args.a2tta`` trainer dictionary."""
    samples_per_day = int(args.samples_per_day)
    if samples_per_day <= 0:
        samples_per_day = 24 if str(args.dataset).upper() == "TFNSW" else 288
    detail_csv_path = str(args.detail_csv_path or "")
    args.a2tta = {
        "dataset": args.dataset,
        "method": args.method,
        "hidden_dim": args.adapter_hidden_dim,
        "node_emb_dim": args.node_emb_dim,
        "calibrator_dropout": args.calibrator_dropout,
        "calibrator_arch": args.calibrator_arch,
        "adapter_rank": args.adapter_rank,
        "gate_type": args.gate_type,
        "temporal_kernel": args.temporal_kernel,
        "local_steps": args.local_steps,
        "adapt_lr": args.adapt_lr,
        "adapt_steps": args.adapt_steps,
        "adapt_every_batches": args.adapt_every_batches,
        "adapt_every_windows": args.adapt_every_windows,
        "label_delay_steps": args.label_delay_steps,
        "label_keep_prob": args.label_keep_prob,
        "label_noise_std": args.label_noise_std,
        "label_spike_prob": args.label_spike_prob,
        "label_spike_scale": args.label_spike_scale,
        "horizon_emb_dim": args.horizon_emb_dim,
        "emb_lr": args.emb_lr if args.emb_lr > 0 else args.adapt_lr,
        "node_budget_frac": args.node_budget_frac,
        "emb_batch_cap": args.emb_batch_cap,
        "lambda_reg_emb": args.lambda_reg_emb if args.lambda_reg_emb >= 0 else args.lambda_reg,
        "budget_frac": args.budget_frac,
        "candidate_pool_size": args.candidate_pool_size,
        "lambda_cons": args.lambda_cons,
        "lambda_reg": args.lambda_reg,
        "mc_K": args.mc_K,
        "w_err": args.w_err,
        "w_unc": args.w_unc,
        "w_shift": args.w_shift,
        "w_recency": args.w_recency,
        "warmup_epochs": args.warmup_epochs,
        "warmup_lr": args.warmup_lr,
        "eval_batch_size": args.eval_batch_size,
        "input_normalization": args.input_normalization,
        "sensor_id_mode": args.sensor_id_mode,
        "sensor_metadata_root": args.sensor_metadata_root,
        "samples_per_day": samples_per_day,
        "infer_chunk": args.infer_chunk,
        "csv_path": args.csv_path,
        "detail_csv_path": detail_csv_path,
        "profile_path": args.profile_path,
        "profile_tag": args.profile_tag,
        "profile_repeat": args.profile_repeat,
        "visualization_root": args.visualization_root,
        "fast_dev_run": bool(args.fast_dev_run),
    }


def _init_preserving_explicit_cli(args, parser, argv):
    """Load the JSON config while keeping explicitly supplied CLI overrides.

    ``utils.initialize.init`` historically applies the JSON after argument
    parsing. That makes options such as ``--adapt_lr`` and ``--logname``
    silently ineffective whenever the same key exists in the config. Dataset
    settings still come from JSON, but options present in ``argv`` win here.
    """
    snapshot = vars(args).copy()
    explicit = set()
    for action in parser._actions:
        for option in action.option_strings:
            if any(token == option or token.startswith(option + "=") for token in argv):
                explicit.add(action.dest)
                break
    init(args)
    for dest in explicit:
        vars(args)[dest] = snapshot[dest]
    # ``init`` computed this before an explicit --logname was restored.
    vars(args)["path"] = osp.join(args.model_path, f"{args.logname}-{args.seed}")
    mkdirs(args.path)


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    vars(args)["device"] = (
        torch.device(f"cuda:{args.gpuid}")
        if torch.cuda.is_available() and args.gpuid != -1
        else torch.device("cpu")
    )

    # Methods registry consumed by `_load_backbone_for_year` via args.methods.
    vars(args)["methods"] = METHOD_REGISTRY

    # Pull JSON conf into args (same as main.py / utils.initialize.init).
    _init_preserving_explicit_cli(args, parser, sys.argv[1:])
    seed_anything(args.seed)
    init_log(args)
    _stash_a2tta_cfg(args)
    main(args)
