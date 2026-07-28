"""Focused regression tests for the strict chronological A2TTA protocol."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
from torch import nn

import a2tta_main as entry
import chronos_baseline as chronos
from src.model.a2tta import ResidualCalibrator
import src.trainer.a2tta_trainer as trainer
from utils.data_convert import _is_pems_path, generate_samples


class _Backbone(nn.Module):
    """Small deterministic backbone with the same flat output convention."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon

    def forward(self, data, adj):
        del adj
        return data.x[:, -self.horizon :]


def _protocol_cfg(**overrides):
    cfg = {
        "method": "tta_ctx_local",
        "adapt_lr": 1e-2,
        "adapt_steps": 1,
        "adapt_every_batches": 1,
        "adapt_every_windows": 1,
        "label_delay_steps": 2,
        "label_keep_prob": 1.0,
        "label_noise_std": 0.0,
        "label_spike_prob": 0.0,
        "label_spike_scale": 5.0,
        "candidate_pool_size": 8,
        "budget_frac": 1.0,
        "lambda_cons": 0.0,
        "lambda_reg": 0.0,
        "mc_K": 1,
        "samples_per_day": 24,
        "infer_chunk": 16,
        "eval_batch_size": 1,
        "local_steps": 1,
        "profile_path": "",
        "detail_csv_path": "",
        "visualization_root": "",
        "fast_dev_run": False,
    }
    cfg.update(overrides)
    return cfg


class StrictProtocolTests(unittest.TestCase):
    def test_parser_defaults_are_strict(self):
        args = entry._build_parser().parse_args([])
        self.assertEqual(args.eval_batch_size, 1)
        self.assertEqual(args.adapt_every_windows, 64)
        self.assertEqual(args.input_normalization, "train")
        self.assertEqual(args.sensor_id_mode, "metadata")

    def test_pems_path_detection_is_location_and_case_independent(self):
        self.assertTrue(_is_pems_path("data/pems05/FastData/2025"))
        self.assertTrue(_is_pems_path("/download/PEmS12/RawData/2025.npz"))
        self.assertFalse(_is_pems_path("data/tfnsw/FastData/2025"))

    def test_pems_rebuild_uses_31_day_rule_and_training_statistics(self):
        timeline = np.arange(400, dtype=np.float32)[:, None]
        raw = np.repeat(timeline, 2, axis=1)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            output = Path(tmp) / "pems05" / "FastData" / "2025"
            output.parent.mkdir(parents=True)
            generated = generate_samples(
                1, str(output), raw, nx.path_graph(2)
            )

        lengths = {
            split: generated[f"{split}_x"].shape[0] + 24
            for split in ("train", "val", "test")
        }
        self.assertEqual(sum(lengths.values()), 288)
        self.assertAlmostEqual(
            float(generated["train_x"].mean()), 0.0, places=5
        )
        self.assertAlmostEqual(
            float(generated["train_x"].std()), 1.0, places=5
        )
        self.assertGreater(float(generated["test_x"].mean()), 1.0)
        self.assertTrue(np.isfinite(float(generated["input_mean"])))
        self.assertGreater(float(generated["input_std"]), 0.0)

    def test_chronos_alignment_accepts_training_stat_cache(self):
        timeline = np.arange(400, dtype=np.float32)[:, None]
        raw = np.repeat(timeline, 2, axis=1)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp) / "pems05"
            raw_dir = root / "RawData"
            fast_dir = root / "FastData"
            raw_dir.mkdir(parents=True)
            fast_dir.mkdir(parents=True)
            np.savez(raw_dir / "2025.npz", x=raw)
            generated = generate_samples(
                31, str(fast_dir / "2025"), raw, nx.path_graph(2)
            )
            test_x, test_y = chronos.build_test_windows(
                str(raw_dir), 2025, str(fast_dir), verify=True
            )

        np.testing.assert_allclose(
            chronos.z_score(
                test_x, generated["input_mean"], generated["input_std"]
            ),
            generated["test_x"],
            rtol=0.0,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            test_y, generated["test_y"], rtol=0.0, atol=1e-3
        )

    def test_explicit_cli_values_survive_json_initialization(self):
        parser = entry._build_parser()
        argv = ["--adapt_lr", "0.123", "--logname", "explicit"]
        args = parser.parse_args(argv)
        old_init, old_mkdirs = entry.init, entry.mkdirs

        def fake_init(target):
            target.adapt_lr = 0.9
            target.logname = "from_config"
            target.dataset = "PEMS07"
            target.model_path = "/tmp/models"
            target.path = "/tmp/models/from_config"

        try:
            entry.init = fake_init
            entry.mkdirs = lambda _path: None
            entry._init_preserving_explicit_cli(args, parser, argv)
        finally:
            entry.init, entry.mkdirs = old_init, old_mkdirs

        self.assertEqual(args.adapt_lr, 0.123)
        self.assertEqual(args.logname, "explicit")
        self.assertEqual(args.dataset, "PEMS07")
        self.assertEqual(args.path, "/tmp/models/explicit-42")

    def test_training_window_statistics_match_materialized_windows(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(31, 4)).astype(np.float32)
        raw[3, 1] = np.nan
        windows = entry._raw_input_windows(raw, x_len=5, y_len=3)
        mean, std = entry._window_input_stats(raw, x_len=5, y_len=3)
        materialized = windows.astype(np.float64)
        self.assertAlmostEqual(mean, float(np.nanmean(materialized)), places=10)
        self.assertAlmostEqual(std, float(np.nanstd(materialized)), places=10)

    def test_sensor_embeddings_follow_stable_ids(self):
        sequences = {
            1: ["a", "b", "c"],
            2: ["b", "d", "a"],
            3: ["d", "c", "a"],
        }
        args = SimpleNamespace(
            a2tta={"sensor_id_mode": "metadata"},
            dataset="PEMSXX",
            device=torch.device("cpu"),
            _sensor_id_registry={},
        )
        old_reader = entry._year_sensor_ids
        entry._year_sensor_ids = (
            lambda _args, year, expected: sequences[year][:expected]
        )
        try:
            entry._set_year_sensor_mapping(args, 1, 3)
            self.assertEqual(args.node_global_idx.tolist(), [0, 1, 2])
            self.assertEqual(args.sensor_new_local_indices, [])

            entry._set_year_sensor_mapping(args, 2, 3)
            self.assertEqual(args.node_global_idx.tolist(), [1, 3, 0])
            self.assertEqual(args.sensor_new_local_indices, [1])

            entry._set_year_sensor_mapping(args, 3, 3)
            self.assertEqual(args.node_global_idx.tolist(), [3, 2, 0])
            self.assertEqual(args.sensor_new_local_indices, [1])
            self.assertEqual(args.sensor_registry_size, 4)
        finally:
            entry._year_sensor_ids = old_reader

    def test_local_context_rejects_multiwindow_batches(self):
        args = SimpleNamespace(
            adj=torch.eye(2),
            y_len=2,
            device=torch.device("cpu"),
            a2tta=_protocol_cfg(
                eval_batch_size=64,
                adapt_every_windows=64,
            ),
        )
        calibrator = ResidualCalibrator(
            num_nodes_max=2,
            x_len=3,
            y_len=2,
            node_emb_dim=2,
            hidden_dim=8,
            dropout=0.0,
            arch="film",
        )
        with self.assertRaisesRegex(ValueError, "eval_batch_size=1"):
            trainer.online_a2tta_eval(
                _Backbone(2), calibrator, inputs={}, args=args
            )

    def test_future_target_change_cannot_affect_prerelease_predictions(self):
        torch.manual_seed(13)
        rng = np.random.default_rng(13)
        windows, x_len, horizon, nodes = 18, 3, 2, 2
        x_chrono = rng.normal(size=(windows, x_len, nodes)).astype(np.float32)
        y_chrono = rng.normal(size=(windows, horizon, nodes)).astype(np.float32)
        train_x = rng.normal(size=(10, x_len, nodes)).astype(np.float32)
        train_y = rng.normal(size=(10, horizon, nodes)).astype(np.float32)

        base_inputs = {
            "train_x": train_x,
            "train_y": train_y,
            "test_x": x_chrono[::-1].copy(),
            "test_y": y_chrono[::-1].copy(),
        }
        changed_inputs = copy.deepcopy(base_inputs)
        changed = y_chrono.copy()
        changed_from = 10
        changed[changed_from:] += 25.0
        changed_inputs["test_y"] = changed[::-1].copy()

        initial = ResidualCalibrator(
            num_nodes_max=nodes,
            x_len=x_len,
            y_len=horizon,
            node_emb_dim=2,
            hidden_dim=8,
            dropout=0.0,
            arch="film",
        )

        def run(inputs):
            captured = {}
            args = SimpleNamespace(
                adj=torch.eye(nodes),
                y_len=horizon,
                device=torch.device("cpu"),
                a2tta=_protocol_cfg(),
                seed=51,
                year=2025,
                node_global_idx=torch.arange(nodes),
            )
            old_metric = trainer.cal_metric
            trainer.cal_metric = (
                lambda truth, pred, _args: captured.update(
                    truth=truth.copy(), pred=pred.copy()
                )
            )
            try:
                trainer.online_a2tta_eval(
                    _Backbone(horizon),
                    copy.deepcopy(initial),
                    inputs,
                    args,
                )
            finally:
                trainer.cal_metric = old_metric
            return captured["pred"]

        pred_original = run(base_inputs)
        pred_changed = run(changed_inputs)
        first_possible_effect = changed_from + horizon
        np.testing.assert_allclose(
            pred_original[:first_possible_effect],
            pred_changed[:first_possible_effect],
            rtol=0.0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
