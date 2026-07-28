"""
A2TTA trainer and online evaluator.

Pipeline per year:
  1. Load frozen backbone checkpoint (Online-AN by default; configurable).
  2. (Optional) warm up the residual calibrator on the year's train split,
     keeping the backbone frozen. L1 loss on raw-scale targets.
  3. Run causal online evaluation on the test split:
        * test windows are reordered to true chronological order
          (`generate_dataset` produces them latest-first, see data_convert.py)
        * each window enters a `pending_queue`; once its configured label delay
          has elapsed (the target horizon by default), it is released into
          `candidate_pool`
        * every `adapt_every_windows` chronological windows we (a) score
          candidates with ActiveSelector, (b) take `adapt_steps` gradient
          steps on the calibrator only (backbone stays frozen). If the
          window-based cadence is omitted, the legacy batch cadence is
          converted to an equivalent number of windows.
  4. Compute per-horizon metrics with the same `cal_metric` used by every
     other baseline so numbers are directly comparable to Online-AN etc.

Anti-leakage invariants: a sample's target is never read until its release
time, and the local-context method processes exactly one chronological input
window per logical batch. The ``pending`` queue stores targets, but adaptation
only consumes post-release entries from ``candidate_pool``.
"""
from __future__ import annotations

import os
import csv
import copy
import math
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import (
    cal_metric,
    masked_mae_np,
    masked_mape_np,
    masked_mse_np,
)
from src.model.a2tta import ResidualCalibrator, ActiveSelector
# Minimal helpers for the local-context clone (tta_ctx_local, Ours selection).
from src.trainer.ctx_local import stack_pool, context_row_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense(pred_flat, y_flat, batch):
    pred, _ = to_dense_batch(pred_flat, batch=batch)
    y, _ = to_dense_batch(y_flat, batch=batch)
    return pred, y


def _build_node_idx(B: int, N: int, device, base=None) -> torch.Tensor:
    """Repeat the current year's stable global sensor IDs for a batch."""
    if base is None:
        base = torch.arange(N, device=device)
    else:
        base = torch.as_tensor(base, dtype=torch.long, device=device)
        if base.numel() != N:
            raise ValueError(
                f"node-id mapping has {base.numel()} entries for a {N}-node graph"
            )
    return base.unsqueeze(0).expand(B, N).reshape(-1)


class _XOnly:
    """Minimal stand-in for a PyG batch when a model's forward only reads .x."""
    __slots__ = ("x",)
    def __init__(self, x):
        self.x = x


def _backbone_infer(backbone, data, adj, N, chunk: int = 16):
    """Inference-only backbone call. For STAEFORMER the spatial-attention
    activations scale with batch×N, so chunk the batch internally on dense
    graphs. The logical batch and TTA cadence are unchanged. Legacy backbones
    pass through untouched. Call under torch.no_grad()."""
    if not hasattr(getattr(backbone, "backbone", None), "adaptive_embedding"):
        return backbone(data, adj)
    x = data.x
    B = x.shape[0] // N
    if B <= chunk:
        return backbone(data, adj)
    outs = []
    for i in range(0, B, chunk):
        outs.append(backbone(_XOnly(x[i * N:(i + chunk) * N]), adj))
    return torch.cat(outs, dim=0)


def _staeformer_forward_flat(model, x3: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Grad-enabled STAEFORMER_Model forward from a dense [B, N, T_in] tensor.

    Mirrors STAEFORMER_Model.forward but skips the PyG Data wrapper (and its
    eval-mode no-op dropout). Used by the a2tta_emb mode, which needs a FRESH
    backbone forward so gradients reach the adaptive embedding — the pool's
    cached y_base tensors are detached and cannot carry gradients.
    """
    T = x3.shape[-1]
    feat = model.backbone(x3, adj).reshape(-1, T)
    h = feat + x3.reshape(-1, T)
    return model.fc(model.activation(h))


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

def warmup_calibrator(
    backbone: nn.Module,
    calibrator: ResidualCalibrator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    epochs: int = 5,
    lr: float = 1e-3,
):
    """Train calibrator on train split, backbone frozen."""
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    opt = optim.AdamW(filter(lambda p: p.requires_grad, calibrator.parameters()), lr=lr)
    N = args.adj.shape[0]
    infer_chunk = int(getattr(args, "a2tta", {}).get("infer_chunk", 16))

    best_val = float("inf")
    for ep in range(epochs):
        calibrator.train()
        total, cnt = 0.0, 0
        for data in train_loader:
            data = data.to(args.device, non_blocking=True)
            with torch.no_grad():
                y_base = _backbone_infer(backbone, data, args.adj, N, infer_chunk)  # [B*N, T]
            B = y_base.shape[0] // N
            x_in = data.x.reshape(-1, N, args.gcn["in_channel"]).reshape(-1, args.gcn["in_channel"])
            node_idx = _build_node_idx(
                B, N, args.device, getattr(args, "node_global_idx", None)
            )
            y_pred = calibrator(y_base, x_in, node_idx)
            loss = F.l1_loss(y_pred, data.y, reduction="mean")
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss); cnt += 1

        # Validation
        calibrator.eval()
        v_total, v_cnt = 0.0, 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(args.device, non_blocking=True)
                y_base = _backbone_infer(backbone, data, args.adj, N, infer_chunk)
                B = y_base.shape[0] // N
                x_in = data.x.reshape(-1, N, args.gcn["in_channel"]).reshape(-1, args.gcn["in_channel"])
                node_idx = _build_node_idx(
                    B, N, args.device, getattr(args, "node_global_idx", None)
                )
                y_pred = calibrator(y_base, x_in, node_idx)
                loss = masked_mae_np(data.y.cpu().numpy(), y_pred.cpu().numpy(), 0)
                if not np.isfinite(loss):
                    continue
                v_total += float(loss); v_cnt += 1
        v_mae = v_total / max(v_cnt, 1)
        if hasattr(args, "logger"):
            tr = total / max(cnt, 1)
            args.logger.info(
                f"  [warmup] year={args.year} ep={ep} train_l1={tr:.4f} val_mae={v_mae:.4f}"
            )
        if v_mae < best_val:
            best_val = v_mae
    return best_val


# ---------------------------------------------------------------------------
# Online delayed-label evaluation with active selection
# ---------------------------------------------------------------------------

class _Sample:
    """One chronological window kept in queues for delayed-label adaptation."""
    __slots__ = ("idx", "x_flat", "y_flat", "node_idx", "y_base", "y_pred", "release_idx")

    def __init__(self, idx, x_flat, y_flat, node_idx, y_base, y_pred, release_idx):
        self.idx = idx
        self.x_flat = x_flat
        self.y_flat = y_flat
        self.node_idx = node_idx
        self.y_base = y_base
        self.y_pred = y_pred
        self.release_idx = release_idx


def _chronological_test_arrays(inputs):
    """`generate_dataset` emits test windows latest-first; reverse to chronological."""
    test_x = inputs["test_x"][::-1].copy()
    test_y = inputs["test_y"][::-1].copy()
    return test_x, test_y


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    """Write a NumPy array atomically so interrupted GPU jobs are resumable."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp, path)


def _export_visualization_prediction(pred_arr, truth_arr, args) -> None:
    """Validate and export A2TTA(STAE) output in the canonical visualization layout."""
    root = str(args.a2tta.get("visualization_root", "") or "")
    if not root:
        return
    if args.a2tta.get("fast_dev_run", False):
        raise RuntimeError("refusing to export a fast_dev_run as a complete prediction")

    dataset = str(args.a2tta.get("dataset", args.dataset)).lower()
    year_dir = os.path.join(os.path.abspath(root), dataset, str(int(args.year)))
    gt_path = os.path.join(year_dir, "ground_truth.npy")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"canonical ground truth missing: {gt_path}")

    pred = np.ascontiguousarray(pred_arr, dtype=np.float32)
    truth = np.ascontiguousarray(truth_arr, dtype=np.float32)
    expected = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    if truth.shape != expected.shape:
        raise ValueError(
            f"target shape mismatch for {dataset}-{args.year}: "
            f"A2TTA={truth.shape}, canonical={expected.shape}"
        )
    if not np.array_equal(truth, expected):
        max_err = float(np.max(np.abs(truth - expected)))
        raise ValueError(
            f"target values/order mismatch for {dataset}-{args.year}; max_abs={max_err}"
        )
    if pred.shape != expected.shape or not np.isfinite(pred).all():
        raise ValueError(
            f"invalid prediction for {dataset}-{args.year}: "
            f"shape={pred.shape}, finite={bool(np.isfinite(pred).all())}"
        )

    out = os.path.join(year_dir, "predictions", "a2tta_stae.npy")
    _atomic_save_npy(out, pred)
    if hasattr(args, "logger"):
        args.logger.info(f"[visualization-export] A2TTA(STAE) -> {out} {pred.shape}")


DETAIL_COLUMNS = [
    "dataset", "method", "backbone_method", "seed", "year", "scope",
    "n_total", "n_selected", "horizon", "MAE", "RMSE", "MAPE",
    "label_delay_steps", "eval_batch_size", "adapt_every_windows",
    "input_normalization", "sensor_id_mode",
]


def _append_detail_metrics(pred_arr, truth_arr, args) -> None:
    """Append all-sensor and true ID-based new-sensor horizon metrics."""
    path = str(args.a2tta.get("detail_csv_path", "") or "")
    if not path:
        return
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_total = int(truth_arr.shape[1])
    new_indices = np.asarray(
        getattr(args, "sensor_new_local_indices", []), dtype=np.int64
    )
    scopes = [("all", truth_arr, pred_arr, n_total)]
    if new_indices.size:
        scopes.append((
            "new",
            truth_arr[:, new_indices, :],
            pred_arr[:, new_indices, :],
            int(new_indices.size),
        ))

    cfg = args.a2tta
    configured_delay = int(cfg.get("label_delay_steps", -1))
    delay = int(args.y_len) if configured_delay < 0 else configured_delay
    cadence = int(cfg.get("adapt_every_windows", -1))
    if cadence <= 0:
        cadence = (
            int(cfg.get("adapt_every_batches", 1))
            * int(cfg.get("eval_batch_size", 1))
        )
    rows = []
    for scope, truth, pred, n_selected in scopes:
        cumulative = []
        for horizon in range(1, int(args.y_len) + 1):
            values = []
            for name, truth_slice, pred_slice in (
                (
                    str(horizon),
                    truth[:, :, :horizon],
                    pred[:, :, :horizon],
                ),
                (
                    f"step{horizon}",
                    truth[:, :, horizon - 1:horizon],
                    pred[:, :, horizon - 1:horizon],
                ),
            ):
                mae = float(masked_mae_np(truth_slice, pred_slice, 0))
                rmse = float(masked_mse_np(truth_slice, pred_slice, 0) ** 0.5)
                mape = float(masked_mape_np(truth_slice, pred_slice, 0))
                values.append((name, mae, rmse, mape))
                rows.append({
                    "dataset": cfg.get("dataset", args.dataset),
                    "method": cfg.get("method", ""),
                    "backbone_method": getattr(args, "backbone_method", ""),
                    "seed": int(args.seed),
                    "year": int(args.year),
                    "scope": scope,
                    "n_total": n_total,
                    "n_selected": n_selected,
                    "horizon": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "label_delay_steps": delay,
                    "eval_batch_size": int(cfg.get("eval_batch_size", 0)),
                    "adapt_every_windows": cadence,
                    "input_normalization": cfg.get("input_normalization", ""),
                    "sensor_id_mode": cfg.get("sensor_id_mode", ""),
                })
            cumulative.append(values[0][1:])
        avg = np.asarray(cumulative, dtype=np.float64).mean(axis=0)
        rows.append({
            "dataset": cfg.get("dataset", args.dataset),
            "method": cfg.get("method", ""),
            "backbone_method": getattr(args, "backbone_method", ""),
            "seed": int(args.seed),
            "year": int(args.year),
            "scope": scope,
            "n_total": n_total,
            "n_selected": n_selected,
            "horizon": "Avg",
            "MAE": float(avg[0]),
            "RMSE": float(avg[1]),
            "MAPE": float(avg[2]),
            "label_delay_steps": delay,
            "eval_batch_size": int(cfg.get("eval_batch_size", 0)),
            "adapt_every_windows": cadence,
            "input_normalization": cfg.get("input_normalization", ""),
            "sensor_id_mode": cfg.get("sensor_id_mode", ""),
        })

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    if hasattr(args, "logger"):
        args.logger.info(
            f"[detail-export] {path} year={args.year} rows={len(rows)} "
            f"new_sensors={int(new_indices.size)}"
        )


def _feedback_label_copy(
    y_clean: torch.Tensor,
    *,
    seed: int,
    year: int,
    window_idx: int,
    keep_prob: float,
    noise_std: float,
    spike_prob: float,
    spike_scale: float,
    target_scale: float,
):
    """Return the delayed-feedback copy, or ``None`` when it is missing.

    The clean tensor used by the metric is never modified.  A separate NumPy
    generator is deterministically keyed by (seed, year, chronological window),
    so corruption does not consume PyTorch/global RNG state and missing-rate
    conditions are nested across runs.
    """
    if keep_prob >= 1.0 and noise_std <= 0.0 and spike_prob <= 0.0:
        return y_clean

    ss = np.random.SeedSequence([
        int(seed) & 0xFFFFFFFF,
        int(year) & 0xFFFFFFFF,
        int(window_idx) & 0xFFFFFFFF,
        0xA2_77_A,
    ])
    rng = np.random.default_rng(ss)
    if rng.random() >= keep_prob:
        return None

    y_feedback = y_clean.clone()
    scale = max(float(target_scale), 1e-8)
    if noise_std > 0.0:
        noise = rng.normal(0.0, float(noise_std) * scale, size=tuple(y_clean.shape))
        y_feedback.add_(torch.as_tensor(noise, dtype=y_clean.dtype, device=y_clean.device))
    if spike_prob > 0.0:
        mask = rng.random(tuple(y_clean.shape)) < float(spike_prob)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=tuple(y_clean.shape))
        impulse = mask * signs * float(spike_scale) * scale
        y_feedback.add_(torch.as_tensor(impulse, dtype=y_clean.dtype, device=y_clean.device))
    return y_feedback


def _shift_score_against_source(
    x_in_flat: torch.Tensor,
    src_mean: torch.Tensor,
    src_std: torch.Tensor,
) -> torch.Tensor:
    """Per-sample distance between input window stats and source train stats.

    Both inputs already in z-score space; we use mean+slope of the window.
    Returns [B*N] -> averaged later to per-sample [P].
    """
    m = x_in_flat.mean(dim=-1)
    s = x_in_flat.std(dim=-1, unbiased=False)
    z_m = (m - src_mean[0]) / (src_std[0] + 1e-6)
    z_s = (s - src_mean[1]) / (src_std[1] + 1e-6)
    return (z_m.abs() + z_s.abs())  # [B*N]


def online_a2tta_eval(
    backbone: nn.Module,
    calibrator: ResidualCalibrator,
    inputs,
    args,
):
    """Run causal online TTA on the year's test split. Logs per-horizon MAE/RMSE/MAPE
    via cal_metric, identical to test_model so numbers are directly comparable.
    """
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    N = args.adj.shape[0]
    H = args.y_len
    device = args.device

    cfg = args.a2tta
    method_mode: str = cfg["method"]   # "backbone" | "calibrator" | "tta_random" | "tta_recent" | "tta_error" | "a2tta_lite" | "tta_all"
    adapt_lr: float = cfg["adapt_lr"]
    adapt_steps: int = cfg["adapt_steps"]
    cand_pool_size: int = cfg["candidate_pool_size"]
    budget_frac: float = cfg["budget_frac"]
    lambda_cons: float = cfg["lambda_cons"]
    lambda_reg: float = cfg["lambda_reg"]
    fast_dev_run: bool = cfg.get("fast_dev_run", False)
    bs: int = int(cfg.get("eval_batch_size", 1))
    if bs <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {bs}")
    adapt_every_batches = int(cfg.get("adapt_every_batches", 1))
    if adapt_every_batches <= 0:
        raise ValueError(
            f"adapt_every_batches must be positive, got {adapt_every_batches}"
        )
    configured_adapt_every_windows = int(cfg.get("adapt_every_windows", -1))
    adapt_every_windows = (
        configured_adapt_every_windows
        if configured_adapt_every_windows > 0
        else adapt_every_batches * bs
    )
    if adapt_every_windows % bs != 0:
        raise ValueError(
            "adapt_every_windows must be divisible by eval_batch_size so an "
            f"adaptation boundary is observable (got {adapt_every_windows} "
            f"windows and batch size {bs})"
        )
    samples_per_day: int = int(cfg.get("samples_per_day", 288))
    if samples_per_day <= 0:
        raise ValueError(f"samples_per_day must be positive, got {samples_per_day}")
    infer_chunk: int = int(cfg.get("infer_chunk", 16))
    mc_K: int = int(cfg.get("mc_K", 4))
    configured_delay = int(cfg.get("label_delay_steps", -1))
    label_delay_steps = H if configured_delay < 0 else configured_delay
    label_delay_is_causal = label_delay_steps >= H
    label_keep_prob = float(cfg.get("label_keep_prob", 1.0))
    label_noise_std = float(cfg.get("label_noise_std", 0.0))
    label_spike_prob = float(cfg.get("label_spike_prob", 0.0))
    label_spike_scale = float(cfg.get("label_spike_scale", 5.0))
    if not 0.0 <= label_keep_prob <= 1.0:
        raise ValueError(f"label_keep_prob must be in [0,1], got {label_keep_prob}")
    if label_noise_std < 0.0:
        raise ValueError(f"label_noise_std must be >=0, got {label_noise_std}")
    if not 0.0 <= label_spike_prob <= 1.0:
        raise ValueError(f"label_spike_prob must be in [0,1], got {label_spike_prob}")
    if label_spike_scale < 0.0:
        raise ValueError(f"label_spike_scale must be >=0, got {label_spike_scale}")
    profiling = bool(str(cfg.get("profile_path", "") or ""))

    def _sync():
        if profiling and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _start():
        _sync()
        return time.perf_counter()

    def _add(profile, key, t0):
        _sync()
        profile[key] += time.perf_counter() - t0

    profile = {
        "data_transfer_seconds": 0.0,
        "queue_seconds": 0.0,
        "global_adapt_seconds": 0.0,
        "backbone_infer_seconds": 0.0,
        "local_clone_seconds": 0.0,
        "calibrator_predict_seconds": 0.0,
        "output_seconds": 0.0,
        "metric_seconds": 0.0,
    }
    if not label_delay_is_causal and hasattr(args, "logger"):
        args.logger.warning(
            f"[NON-CAUSAL diagnostic] label_delay_steps={label_delay_steps} < H={H}"
        )

    # Adaptation enabled?
    do_adapt = method_mode in (
        "tta_random", "tta_recent", "tta_error", "a2tta_lite", "tta_all",
        "a2tta_emb", "tta_ctx_local",
    )
    use_calibrator = method_mode != "backbone"

    if method_mode == "tta_random":
        selector = ActiveSelector(mode="random")
    elif method_mode == "tta_recent":
        selector = ActiveSelector(mode="recent")
    elif method_mode == "tta_error":
        selector = ActiveSelector(mode="error_only")
    elif method_mode in ("tta_all", "tta_ctx_local"):
        # Full-pool adaptation; bypasses the active scoring pass. tta_ctx_local
        # updates the global calibrator exactly like tta_all, then specialises a
        # discardable local clone at prediction time (see the predict block).
        selector = ActiveSelector(mode="all")
    else:
        selector = ActiveSelector(
            w_err=cfg.get("w_err", 1.0), w_unc=cfg.get("w_unc", 0.3),
            w_shift=cfg.get("w_shift", 0.3), w_recency=cfg.get("w_recency", 0.1),
            mode="active",
        )

    # tta_ctx_local (Ours selection): keep the stable global all-label calibrator,
    # then at each window clone it and take a few steps on context-weighted delayed
    # labels before predicting (the clone is discarded). The exploration of other
    # selection / weighting ideas that did NOT beat tta_all is archived in
    # a2tta_back.py — see appendix on sample selection.
    ctx_local = method_mode == "tta_ctx_local"
    if ctx_local and bs != 1:
        raise ValueError(
            "tta_ctx_local requires eval_batch_size=1: its local context must "
            "contain only the current chronological window"
        )
    local_steps = int(cfg.get("local_steps", 3))

    # Snapshot the calibrator init for proximal regularizer.
    init_state = {k: v.detach().clone() for k, v in calibrator.state_dict().items()}

    # Optimizer for the (small) calibrator.
    opt = None
    if do_adapt:
        opt = optim.AdamW(calibrator.parameters(), lr=adapt_lr)

    # a2tta_emb: additionally adapt the backbone's spatio-temporal adaptive
    # embedding (STAEFormer's per-(timestep,node) table) — but ONLY rows of
    # actively-selected nodes, chosen by per-node delayed-label error. The
    # rest of the backbone stays frozen. All other method modes leave the
    # backbone untouched (emb_param stays None), so legacy paths are intact.
    emb_param = None
    emb_init = None
    emb_opt = None
    if method_mode == "a2tta_emb":
        emb_param = getattr(getattr(backbone, "backbone", None), "adaptive_embedding", None)
        if emb_param is None:
            raise ValueError(
                "a2tta_emb requires a STAEFORMER backbone "
                "(backbone.backbone.adaptive_embedding not found) — pass "
                "--backbone_method STAEFORMER with matching checkpoints."
            )
        emb_param.requires_grad_(True)
        emb_init = emb_param.detach().clone()
        # weight_decay=0: rows whose grad we zero must not drift via decay.
        emb_opt = optim.AdamW([emb_param], lr=float(cfg.get("emb_lr", adapt_lr)), weight_decay=0.0)
    node_budget_frac = float(cfg.get("node_budget_frac", 0.25))
    emb_batch_cap = int(cfg.get("emb_batch_cap", 16))
    lambda_reg_emb = float(cfg.get("lambda_reg_emb", lambda_reg))

    # Source-stats reference for shift_score (use train split mean/std of last+slope features).
    train_x = inputs["train_x"]  # [T, x_len, N] (z-score already)
    src_m = float(np.mean(train_x))
    src_s = float(np.std(train_x))
    # use mean of mean / mean of std as crude reference
    src_mean = torch.tensor([src_m, np.std(train_x.reshape(-1, train_x.shape[-1]).mean(axis=0))],
                            device=device, dtype=torch.float32)
    src_std = torch.tensor([np.std(train_x), 1.0], device=device, dtype=torch.float32)

    # Reverse test arrays into chronological order, then build a vanilla DataLoader.
    test_x, test_y = _chronological_test_arrays(inputs)
    # Use only the already-available training split to calibrate synthetic
    # corruption magnitude; evaluation targets do not set the noise scale.
    # Skip the scan entirely for clean/missing-only runs, preserving the exact
    # timing path used by the completed efficiency profiles.
    feedback_target_scale = 1.0
    if label_noise_std > 0.0 or label_spike_prob > 0.0:
        feedback_target_scale = float(np.nanstd(inputs.get("train_y", test_y)))
        if not np.isfinite(feedback_target_scale) or feedback_target_scale < 1e-8:
            feedback_target_scale = 1.0
    chrono_inputs = {"test_x": test_x, "test_y": test_y}
    test_loader = DataLoader(
        SpatioTemporalDataset(chrono_inputs, "test"),
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
        num_workers=0,  # ordering matters; keep determinism
    )

    pending = deque()           # _Sample objects, ordered by chronological idx
    pool: deque[_Sample] = deque(maxlen=cand_pool_size)

    pred_chrono, truth_chrono = [], []
    next_idx = 0    # global chronological index of the next window to be predicted
    batch_idx = 0
    n_adapt_steps_total = 0
    feedback_seen = 0
    feedback_kept = 0

    for batch in test_loader:
        if fast_dev_run and batch_idx >= 4:
            break

        t_phase = _start()
        batch = batch.to(device, non_blocking=True)
        _add(profile, "data_transfer_seconds", t_phase)
        # `to_dense_batch` is correct for variable-N sparse Data, but our
        # SpatioTemporalDataset already yields fixed [N, T] per item, so each
        # sample contributes exactly N rows. Recover B from the data.
        B = batch.x.shape[0] // N
        x_flat = batch.x       # [B*N, T_in]
        y_flat = batch.y       # [B*N, T]
        node_idx = _build_node_idx(
            B, N, device, getattr(args, "node_global_idx", None)
        )

        # ---------- Release samples from pending → candidate pool ----------
        # A sample with global idx i is released once
        # next_idx >= i + label_delay_steps (H in the default protocol).
        t_phase = _start()
        while pending and pending[0].release_idx <= next_idx:
            pool.append(pending.popleft())
        _add(profile, "queue_seconds", t_phase)

        # ---------- Adapt calibrator on current pool ----------
        # Use the chronological window index rather than the loader-batch index.
        # With eval_batch_size=1 and adapt_every_windows=64, updates occur at
        # windows 64, 128, ...: the same global cadence as the legacy
        # eval_batch_size=64/adapt_every_batches=1 protocol, without grouping
        # future input windows into the current prediction context.
        adapt_due = next_idx % adapt_every_windows == 0
        if do_adapt and len(pool) >= max(8, int(0.1 * cand_pool_size)) and adapt_due:
            t_phase = _start()
            n_steps = adapt_steps
            for _ in range(n_steps):
                # Build per-sample features for selection (cheap, on existing tensors).
                ps = list(pool)
                P = len(ps)

                # recent_error (per-sample MAE from cached predictions vs delayed y)
                with torch.no_grad():
                    rec_err = torch.stack([
                        (s.y_pred - s.y_flat).abs().mean() for s in ps
                    ])
                    # uncertainty via MC-dropout on calibrator only
                    if mc_K > 1 and selector.mode == "active":
                        unc_list = []
                        for s in ps:
                            _, std = calibrator.predict_with_uncertainty(
                                s.y_base, s.x_flat, s.node_idx, K=mc_K
                            )
                            unc_list.append(std.mean())
                        unc = torch.stack(unc_list)
                    else:
                        unc = torch.zeros(P, device=device)

                    if selector.mode == "active":
                        shift = torch.stack([
                            _shift_score_against_source(s.x_flat, src_mean, src_std).mean()
                            for s in ps
                        ])
                    else:
                        shift = torch.zeros(P, device=device)

                    # Recency: bigger idx = more recent. Normalize via _safe_norm later.
                    recency = torch.tensor([float(s.idx) for s in ps], device=device)

                budget = max(1, int(budget_frac * P))
                sel = selector.select(rec_err, unc, shift, recency, budget=budget)
                sel_list = [ps[int(i)] for i in sel.tolist()]

                # Stack selected samples into a single big batch for one update.
                xs = torch.cat([s.x_flat for s in sel_list], dim=0)
                ys = torch.cat([s.y_flat for s in sel_list], dim=0)
                ybs = torch.cat([s.y_base for s in sel_list], dim=0)
                ns = torch.cat([s.node_idx for s in sel_list], dim=0)

                calibrator.train()
                y_hat = calibrator(ybs, xs, ns)
                loss_sup = F.l1_loss(y_hat, ys, reduction="mean")

                # Optional weak consistency loss (cheap noise aug on x).
                if lambda_cons > 0:
                    noise1 = 0.01 * torch.randn_like(xs)
                    noise2 = 0.01 * torch.randn_like(xs)
                    y1 = calibrator(ybs, xs + noise1, ns)
                    y2 = calibrator(ybs, xs + noise2, ns)
                    loss_cons = F.l1_loss(y1, y2, reduction="mean")
                else:
                    loss_cons = torch.zeros((), device=device)

                # Proximal regularizer to anchor near init (avoid overfitting to small pool).
                if lambda_reg > 0:
                    loss_reg = torch.zeros((), device=device)
                    for k, p in calibrator.named_parameters():
                        if k in init_state:
                            loss_reg = loss_reg + ((p - init_state[k]) ** 2).sum()
                else:
                    loss_reg = torch.zeros((), device=device)

                loss = loss_sup + lambda_cons * loss_cons + lambda_reg * loss_reg
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                n_adapt_steps_total += 1

                # ---- a2tta_emb: update actively-selected adaptive-emb rows ----
                if emb_param is not None:
                    # Active NODE selection: rank nodes by mean delayed-label
                    # error over the selected samples (extends the sample-level
                    # active principle to the node axis).
                    with torch.no_grad():
                        err_node = torch.stack([
                            (s.y_pred - s.y_flat).abs().mean(dim=-1) for s in sel_list
                        ]).mean(dim=0)                                   # [N]
                        k = max(1, int(node_budget_frac * err_node.shape[0]))
                        node_mask = torch.zeros_like(err_node, dtype=torch.bool)
                        node_mask[torch.topk(err_node, k=k).indices] = True
                    # Fresh backbone forward on a capped sub-batch (gradients
                    # cannot flow through the pool's cached y_base).
                    sub = sel_list[:emb_batch_cap]
                    Bs = len(sub)
                    xs_e = torch.cat([s.x_flat for s in sub], dim=0)
                    ys_e = torch.cat([s.y_flat for s in sub], dim=0)
                    ns_e = torch.cat([s.node_idx for s in sub], dim=0)
                    y_base_fresh = _staeformer_forward_flat(
                        backbone, xs_e.reshape(Bs, N, -1), args.adj)
                    y_hat_e = calibrator(y_base_fresh, xs_e, ns_e)
                    loss_e = (F.l1_loss(y_hat_e, ys_e, reduction="mean")
                              + lambda_reg_emb * ((emb_param - emb_init) ** 2).sum())
                    if torch.isfinite(loss_e):
                        emb_opt.zero_grad(set_to_none=True)
                        # Calibrator grads from this backward are discarded by
                        # its own zero_grad at the next calibrator step.
                        loss_e.backward()
                        if emb_param.grad is not None:
                            emb_param.grad[:, ~node_mask, :] = 0
                        emb_opt.step()
            _add(profile, "global_adapt_seconds", t_phase)

        # ---------- Predict current chronological window ----------
        calibrator.eval()
        t_phase = _start()
        with torch.no_grad():
            y_base = _backbone_infer(backbone, batch, args.adj, N, infer_chunk)             # [B*N, T]
        _add(profile, "backbone_infer_seconds", t_phase)

        if ctx_local and len(pool) >= max(8, int(0.1 * cand_pool_size)):
            t_phase = _start()
            # Ours (selection): clone the stable global calibrator, specialise the
            # clone on context-weighted delayed labels for a few steps, predict
            # with it, then discard it — the global calibrator is never biased.
            vars(args)["_a2tta_current_window_idx"] = int(next_idx)
            local = copy.deepcopy(calibrator); local.train()
            lopt = optim.AdamW(local.parameters(), lr=adapt_lr)
            pool_d = stack_pool(list(pool), device)
            ctx_l = {"x_curr": x_flat.reshape(B, N, -1), "yb_curr": y_base.reshape(B, N, -1),
                     "target_idx": next_idx}
            xs_l = pool_d["x"].reshape(-1, pool_d["x"].shape[-1])
            ys_l = pool_d["y"].reshape(-1, pool_d["y"].shape[-1])
            ybs_l = pool_d["yb"].reshape(-1, pool_d["yb"].shape[-1])
            ns_l = pool_d["node_idx"].reshape(-1)
            for _ in range(local_steps):
                rw = context_row_weights(
                    "hybrid", pool_d, ctx_l, local, spd=samples_per_day
                )
                yh = local(ybs_l, xs_l, ns_l)
                pr = (yh - ys_l).abs().mean(dim=-1)
                loss = (pr * rw).mean() if rw is not None else pr.mean()
                if not torch.isfinite(loss):
                    break
                lopt.zero_grad(set_to_none=True); loss.backward(); lopt.step()
            local.eval()
            with torch.no_grad():
                y_pred_flat = local(y_base, x_flat, node_idx)
            del local, lopt
            _add(profile, "local_clone_seconds", t_phase)
        else:
            t_phase = _start()
            with torch.no_grad():
                if use_calibrator:
                    y_pred_flat = calibrator(y_base, x_flat, node_idx)
                else:
                    y_pred_flat = y_base
            _add(profile, "calibrator_predict_seconds", t_phase)

        t_phase = _start()
        pred_dense, y_dense = _to_dense(y_pred_flat, y_flat, batch.batch)
        pred_chrono.append(pred_dense.detach().cpu().numpy())
        truth_chrono.append(y_dense.detach().cpu().numpy())
        _add(profile, "output_seconds", t_phase)

        # ---------- Enqueue this batch's samples for future delayed adaptation ----------
        # Slice the flat tensors back into per-sample chunks of N rows.
        for j in range(B):
            i_global = next_idx + j
            x_j = x_flat[j * N : (j + 1) * N].detach()
            y_j = y_flat[j * N : (j + 1) * N].detach()
            feedback_seen += 1
            y_feedback_j = _feedback_label_copy(
                y_j, seed=args.seed, year=args.year, window_idx=i_global,
                keep_prob=label_keep_prob, noise_std=label_noise_std,
                spike_prob=label_spike_prob, spike_scale=label_spike_scale,
                target_scale=feedback_target_scale,
            )
            if y_feedback_j is None:
                continue
            feedback_kept += 1
            yb_j = y_base[j * N : (j + 1) * N].detach()
            yp_j = y_pred_flat[j * N : (j + 1) * N].detach()
            n_j = node_idx[j * N : (j + 1) * N].detach()
            pending.append(_Sample(
                idx=i_global, x_flat=x_j, y_flat=y_feedback_j, node_idx=n_j,
                y_base=yb_j, y_pred=yp_j,
                release_idx=i_global + label_delay_steps,
            ))
        next_idx += B
        batch_idx += 1

    # Stack predictions in chronological order, run cal_metric.
    pred_arr = np.concatenate(pred_chrono, axis=0)
    truth_arr = np.concatenate(truth_chrono, axis=0)
    _export_visualization_prediction(pred_arr, truth_arr, args)
    _append_detail_metrics(pred_arr, truth_arr, args)
    t_phase = _start()
    cal_metric(truth_arr, pred_arr, args)
    _add(profile, "metric_seconds", t_phase)

    profile.update({
        "num_windows": int(pred_arr.shape[0]),
        "num_batches": int(batch_idx),
        "label_delay_steps": int(label_delay_steps),
        "label_delay_is_causal": int(label_delay_is_causal),
        "adapt_every_windows": int(adapt_every_windows),
        "adapt_steps_total": int(n_adapt_steps_total),
        "label_keep_prob": float(label_keep_prob),
        "label_noise_std": float(label_noise_std),
        "label_spike_prob": float(label_spike_prob),
        "label_spike_scale": float(label_spike_scale),
        "feedback_target_scale": float(feedback_target_scale),
        "feedback_seen": int(feedback_seen),
        "feedback_kept": int(feedback_kept),
        "feedback_keep_rate_actual": (float(feedback_kept) / feedback_seen
                                      if feedback_seen else float("nan")),
    })
    args._a2tta_last_profile = profile
    if not hasattr(args, "_a2tta_feedback_stats"):
        args._a2tta_feedback_stats = {}
    args._a2tta_feedback_stats[int(args.year)] = {
        k: profile[k] for k in (
            "label_keep_prob", "label_noise_std", "label_spike_prob",
            "label_spike_scale", "feedback_target_scale", "feedback_seen",
            "feedback_kept", "feedback_keep_rate_actual",
        )
    }

    if hasattr(args, "logger"):
        args.logger.info(
            f"  [a2tta] year={args.year} method={method_mode} "
            f"label_delay={label_delay_steps} (H={H}, causal={label_delay_is_causal}) "
            f"samples_per_day={samples_per_day} "
            f"adapt_every_windows={adapt_every_windows} "
            f"adapt_steps={n_adapt_steps_total} pool_size_final={len(pool)} "
            f"pending_final={len(pending)} feedback={feedback_kept}/{feedback_seen} "
            f"noise_std={label_noise_std} spike_prob={label_spike_prob}"
        )
    return n_adapt_steps_total


# ---------------------------------------------------------------------------
# CSV result writer (one row per year × method × seed × horizon)
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "dataset", "method", "seed", "year", "horizon",
    "MAE", "RMSE", "MAPE",
    "adapt_lr", "adapt_steps", "budget_frac", "candidate_pool_size",
    "lambda_cons", "hidden_dim", "label_delay_steps", "eval_batch_size",
    "adapt_every_windows",
    "input_normalization", "sensor_id_mode", "samples_per_day",
    "label_keep_prob", "label_noise_std", "label_spike_prob",
    "label_spike_scale", "feedback_target_scale", "feedback_seen",
    "feedback_kept", "feedback_keep_rate_actual",
]


def append_results_csv(args, csv_path: str):
    """Pull args.result and append per-year per-horizon rows."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        cfg = args.a2tta
        for h in ["3", "6", "12", "Avg"]:
            res = args.result.get(h, {})
            mae_d = res.get(" MAE", {})
            rmse_d = res.get("RMSE", {})
            mape_d = res.get("MAPE", {})
            for year in sorted(set(mae_d.keys()) | set(rmse_d.keys()) | set(mape_d.keys())):
                feedback = getattr(args, "_a2tta_feedback_stats", {}).get(int(year), {})
                w.writerow({
                    "dataset": cfg.get("dataset", "PEMS05"),
                    "method": cfg.get("method", "?"),
                    "seed": args.seed,
                    "year": year,
                    "horizon": h,
                    "MAE": mae_d.get(year, ""),
                    "RMSE": rmse_d.get(year, ""),
                    "MAPE": mape_d.get(year, ""),
                    "adapt_lr": cfg.get("adapt_lr", ""),
                    "adapt_steps": cfg.get("adapt_steps", ""),
                    "budget_frac": cfg.get("budget_frac", ""),
                    "candidate_pool_size": cfg.get("candidate_pool_size", ""),
                    "lambda_cons": cfg.get("lambda_cons", ""),
                    "hidden_dim": cfg.get("hidden_dim", ""),
                    "label_delay_steps": (args.y_len if int(cfg.get("label_delay_steps", -1)) < 0
                                          else int(cfg.get("label_delay_steps", -1))),
                    "eval_batch_size": cfg.get("eval_batch_size", ""),
                    "adapt_every_windows": (
                        int(cfg.get("adapt_every_windows", -1))
                        if int(cfg.get("adapt_every_windows", -1)) > 0
                        else int(cfg.get("adapt_every_batches", 1))
                        * int(cfg.get("eval_batch_size", 1))
                    ),
                    "input_normalization": cfg.get("input_normalization", ""),
                    "sensor_id_mode": cfg.get("sensor_id_mode", ""),
                    "samples_per_day": cfg.get("samples_per_day", ""),
                    "label_keep_prob": cfg.get("label_keep_prob", 1.0),
                    "label_noise_std": cfg.get("label_noise_std", 0.0),
                    "label_spike_prob": cfg.get("label_spike_prob", 0.0),
                    "label_spike_scale": cfg.get("label_spike_scale", 5.0),
                    "feedback_target_scale": feedback.get("feedback_target_scale", ""),
                    "feedback_seen": feedback.get("feedback_seen", ""),
                    "feedback_kept": feedback.get("feedback_kept", ""),
                    "feedback_keep_rate_actual": feedback.get("feedback_keep_rate_actual", ""),
                })
