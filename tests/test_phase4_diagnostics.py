"""Regression checks for non-mutating phase-4 diagnostic operations."""
import contextlib
import io
import unittest

import torch
from torch import nn

from diagnose_phase4 import gradient_metrics, bn_buffers, preserve_runtime_state, probe_gradients
from test_graph_phases import make_model


class DiagnosticTests(unittest.TestCase):
    def test_gradient_geometry_and_absent_derivatives(self):
        result = gradient_metrics([torch.tensor([3., 4.]), None], [torch.tensor([-6., -8.]), torch.tensor([2.])])
        self.assertAlmostEqual(result["task_norm"], 5.)
        self.assertAlmostEqual(result["cf_norm"], 104 ** .5)
        self.assertAlmostEqual(result["cosine"], -50 / (5 * 104 ** .5))
        self.assertEqual(result["task_unused"], 1)
        self.assertIsNone(gradient_metrics([None], [torch.ones(2)])["cosine"])

    def test_bn_and_modes_restore_even_when_probe_fails(self):
        model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
        model.eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with preserve_runtime_state(model):
                model.train()
                model(torch.randn(4, 3))
                raise RuntimeError("test failure")
        self.assertTrue(all(not module.training for module in model.modules()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(set(bn_buffers(model)), {"0.running_mean", "0.running_var", "0.num_batches_tracked"})

    def test_actual_cf_probe_preserves_all_parameters_buffers_and_grads(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(42)
            model = make_model(4)
            model.backbone = nn.Linear(32, 32)
            model.eval()
            before = {key: value.clone() for key, value in model.state_dict().items()}
            batches = [{"image": torch.randn(2, 4, 32), "au_labels": torch.randint(0, 2, (2, 8)).float()}]
            with contextlib.redirect_stdout(io.StringIO()):
                rows = probe_gradients(model, batches, "cpu", seed=0)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["task_norm"] > 0 and row["cf_norm"] > 0 for row in rows))
            self.assertTrue(all(-1.00001 <= row["cosine"] <= 1.00001 for row in rows))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(not module.training for module in model.modules()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
