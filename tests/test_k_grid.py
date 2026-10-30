"""Refinement snapshots must match independent runs and leave weights untouched."""
import contextlib
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from diagnose_phase4 import au_metrics
from test_graph_phases import make_model
from validate_k_grid import evaluate_k_grid, print_results


def original_refinement(model, au_logits, emotion_logits, steps):
    """Independent reference: the original hard-coded ten-step objective."""
    targets = model.au_to_expr_pseudo((torch.sigmoid(au_logits.detach()) > .5).float())
    with torch.enable_grad():
        au = au_logits.detach().clone().requires_grad_(True)
        expression = emotion_logits.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([au, expression], lr=.01)
        for _ in range(steps):
            optimizer.zero_grad()
            e_au = model.facs_au_violation_loss(torch.sigmoid(au))
            e_expression = model.expression_bce_loss(torch.sigmoid(expression), targets)
            e_prior = F.mse_loss(au, au_logits.detach()) + F.mse_loss(expression, emotion_logits.detach())
            energy = .05 * (e_au + e_expression) + 1.0 * e_prior
            energy.backward()
            optimizer.step()
    return torch.sigmoid(au).detach(), torch.sigmoid(expression).detach()


class KGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_snapshots_match_independent_original_loop_for_both_datasets(self):
        for dataset in ("disfa", "bp4d"):
            with self.subTest(dataset=dataset):
                torch.manual_seed(71)
                model = make_model(3, dataset=dataset).eval()
                au = torch.randn(3, model.num_aus, requires_grad=True)
                expression = torch.randn(3, model.num_emotions, requires_grad=True)
                with torch.no_grad():
                    actual = model.refine_graph_predictions(au, expression, [0, 1, 3, 10, 15])
                for k, pair in actual.items():
                    expected = original_refinement(model, au, expression, k)
                    for measured, reference in zip(pair, expected):
                        torch.testing.assert_close(measured, reference, rtol=0, atol=0)
                        self.assertFalse(measured.requires_grad)
                self.assertIsNone(au.grad)
                self.assertIsNone(expression.grad)
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_single_validation_pass_matches_independent_metrics_and_preserves_state(self):
        torch.manual_seed(19)
        model = make_model(3).eval()
        batches = [{"image": torch.randn(size, 4, 32),
                    "au_labels": torch.randint(0, 2, (size, 8)).float()} for size in (3, 1)]
        saved = {name: value.clone() for name, value in model.state_dict().items()}
        with patch.object(model, "forward", wraps=model.forward) as forward:
            measured = evaluate_k_grid(model, batches, "cpu", [3, 10, 15])
        self.assertEqual(forward.call_count, len(batches))
        self.assertTrue(all(call.kwargs == {} and len(call.args) == 1 for call in forward.call_args_list))
        self.assertEqual(set(measured), {0, 3, 10, 15})
        all_labels = torch.cat([batch["au_labels"] for batch in batches])
        for k in measured:
            expected_probs = []
            for batch in batches:
                with torch.no_grad():
                    output = model(batch["image"])
                probabilities, _ = original_refinement(model, output["graph_au_logits"], output["graph_emo_logits"], k)
                if k == 0:
                    torch.testing.assert_close(probabilities, output["graph_au_probs"], rtol=0, atol=0)
                expected_probs.append(probabilities)
            reference = au_metrics(torch.cat(expected_probs), all_labels)
            for key in ("f1", "precision", "recall"):
                torch.testing.assert_close(measured[k][key], reference[key], rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            print_results(measured, model.au_ids)
        for k in measured:
            self.assertIn(f"K = {k}", log.getvalue())
        self.assertIn("AU26:", log.getvalue())
        self.assertIn("Best validation K:", log.getvalue())

    def test_invalid_grid_and_double_refinement_are_rejected(self):
        model = make_model(3)
        with self.assertRaises(ValueError):
            model.refine_graph_predictions(torch.zeros(1, 8), torch.zeros(1, 8), [-1])
        with self.assertRaises(ValueError):
            evaluate_k_grid(model, [], "cpu", [-1])
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_k_grid(model, [], "cpu", [0])
        model.cfg.enable_test_time_refinement = True
        with self.assertRaisesRegex(ValueError, "automatic"):
            evaluate_k_grid(model, [], "cpu", [1])


if __name__ == "__main__":
    unittest.main()
