"""Refinement snapshots must match independent runs and leave weights untouched."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch.nn import functional as F

from config import DISFA_AUS
from diagnose_phase4 import au_metrics
from test_graph_phases import make_model
from validate_k_grid import (
    analyze_k_changes,
    evaluate_k_grid,
    print_change_report,
    print_results,
    save_visualizations,
)


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

    def test_change_analysis_and_visualizations_cover_flips_and_zero_flip_fallback(self):
        # Construct realistic DISFA-like predictions where K refinement fixes and breaks samples.
        # DISFA_AUS = [1, 2, 4, 6, 9, 12, 25, 26] -> indices 6 (AU25) and 7 (AU26).
        labels = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1],  # sample 0: AU25 FN->TP (Fixed)
            [0, 0, 0, 0, 0, 0, 0, 0],  # sample 1: AU26 FP->TN (Fixed)
            [0, 0, 0, 0, 0, 0, 0, 1],  # sample 2: AU25 TN->FP (Broken)
            [0, 0, 0, 0, 0, 1, 0, 1],  # sample 3: AU26 TP->FN (Broken)
            [0, 0, 0, 0, 0, 1, 1, 1],  # sample 4: unchanged TP
            [0, 0, 0, 0, 0, 0, 0, 0],  # sample 5: unchanged TN
        ], dtype=torch.float32)

        p0 = torch.tensor([
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.80, 0.44, 0.88],  # AU26=1, AU25=0 violation
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.20, 0.12, 0.56],  # AU26=1, AU25=0 violation
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.20, 0.45, 0.85],  # AU26=1, AU25=0 violation
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.80, 0.15, 0.55],  # AU26=1, AU25=0 violation
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.85, 0.90, 0.82],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.08, 0.05],
        ], dtype=torch.float32)
        p5 = p0.clone()
        p5[0, 6] = 0.52  # flips at K=5 (FN->TP)
        p5[1, 7] = 0.47  # flips at K=5 (FP->TN)
        p5[2, 6] = 0.49  # not yet flipped
        p5[3, 7] = 0.51  # not yet flipped

        p10 = p5.clone()
        p10[0, 6] = 0.60  # FN->TP (Fixed)
        p10[1, 7] = 0.40  # FP->TN (Fixed)
        p10[2, 6] = 0.57  # TN->FP (Broken)
        p10[3, 7] = 0.44  # TP->FN (Broken)

        predictions = {0: p0, 5: p5, 10: p10}
        emotions = {
            0: torch.tensor([[0.1, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1]] * 6),
            5: torch.tensor([[0.1, 0.1, 0.82, 0.1, 0.2, 0.1, 0.1, 0.1]] * 6),
            10: torch.tensor([[0.1, 0.1, 0.85, 0.1, 0.2, 0.1, 0.1, 0.1]] * 6),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_paths = []
            for idx in range(6):
                img_file = tmp_path / f"SN00{idx % 2 + 1}_{idx:04d}.png"
                Image.new("RGB", (260, 260), color=(40 * idx % 255, 120, 180)).save(img_file)
                img_paths.append(img_file)

            class DummyDataset:
                def __init__(self, paths):
                    self.samples = [(f"SN00{i % 2 + 1}", i * 20, str(p)) for i, p in enumerate(paths)]

            analysis = analyze_k_changes(
                predictions, labels, DISFA_AUS, compare_k=10,
                dataset=DummyDataset(img_paths), emotion_predictions=emotions,
            )
            stats10 = analysis["per_k"][10]
            self.assertEqual(stats10["fn_to_tp"], 1)
            self.assertEqual(stats10["fp_to_tn"], 1)
            self.assertEqual(stats10["tn_to_fp"], 1)
            self.assertEqual(stats10["tp_to_fn"], 1)
            self.assertEqual(stats10["fixed"], 2)
            self.assertEqual(stats10["broken"], 2)
            self.assertEqual(stats10["net_gain"], 0)
            self.assertEqual(analysis["per_k"][0]["hard_facs_violations"], 4)
            self.assertEqual(stats10["hard_facs_violations"], 0)

            # Verify exact confusion-matrix conservation across all K values.
            truth = labels.bool()
            tp0 = (truth & (p0 > 0.5)).sum(dim=0)
            fp0 = ((~truth) & (p0 > 0.5)).sum(dim=0)
            for k in (5, 10):
                tpk = (truth & (predictions[k] > 0.5)).sum(dim=0)
                fpk = ((~truth) & (predictions[k] > 0.5)).sum(dim=0)
                s = analysis["per_k"][k]
                self.assertEqual(
                    (tpk - tp0).tolist(),
                    [a - b for a, b in zip(s["per_au_fn_to_tp"], s["per_au_tp_to_fn"])],
                )
                self.assertEqual(
                    (fpk - fp0).tolist(),
                    [a - b for a, b in zip(s["per_au_tn_to_fp"], s["per_au_fp_to_tn"])],
                )

            by_sample = {ev["sample_idx"]: ev for ev in analysis["compare_events"]}
            self.assertEqual(by_sample[0]["first_flip_k"], 5)
            self.assertEqual(by_sample[2]["first_flip_k"], 10)
            self.assertIn(26, by_sample[0]["partner_aus"])

            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                print_change_report(analysis, DISFA_AUS)
            self.assertIn("Refinement Change Summary vs K=0", log.getvalue())
            self.assertIn("FN->TP", log.getvalue())
            self.assertIn("TP->FN", log.getvalue())

            viz_out = tmp_path / "viz_flips"
            saved = save_visualizations(analysis, predictions, labels, viz_out, num_samples=4)
            for key in ("changed_samples_csv", "summary_json", "curves_and_flips",
                        "per_au_breakdown", "sample_gallery", "trajectories"):
                self.assertIn(key, saved)
                self.assertTrue(saved[key].is_file())
                self.assertGreater(saved[key].stat().st_size, 0)

            with open(saved["changed_samples_csv"], "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            with open(saved["summary_json"], "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["compare_k"], 10)
            self.assertEqual(summary["num_flipped_events"], 4)

            # Also verify zero-flip fallback path renders cleanly without crashing.
            zero_flip_preds = {0: p0, 10: p0.clone()}
            zero_analysis = analyze_k_changes(zero_flip_preds, labels, DISFA_AUS, compare_k=10)
            self.assertEqual(len(zero_analysis["compare_events"]), 0)
            saved_zero = save_visualizations(zero_analysis, zero_flip_preds, labels, tmp_path / "viz_zero")
            for path in saved_zero.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_evaluate_k_grid_return_details_matches_standard_results(self):
        torch.manual_seed(23)
        model = make_model(3).eval()
        batches = [{"image": torch.randn(4, 4, 32), "au_labels": torch.randint(0, 2, (4, 8)).float()}]
        plain = evaluate_k_grid(model, batches, "cpu", [0, 5])
        detailed_metrics, details = evaluate_k_grid(model, batches, "cpu", [0, 5], return_details=True)
        self.assertEqual(set(plain), set(detailed_metrics))
        for k in plain:
            for key in ("f1", "precision", "recall"):
                torch.testing.assert_close(plain[k][key], detailed_metrics[k][key], rtol=0, atol=0)
            self.assertEqual(details["predictions"][k].shape, (4, 8))
            self.assertEqual(details["emotion_predictions"][k].shape, (4, 8))


if __name__ == "__main__":
    unittest.main()
