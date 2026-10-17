"""Graph-stage regressions; pretrained boundaries use synthetic test features only."""
import contextlib
import copy
import io
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from config import ModelConfig, get_dataset_aus, get_dataset_au_expression_prior
from models.ctrlau import CtrlAUModel
from train import (
    TRAINING_SCHEMA, configure_training_phase, keep_frozen_modules_eval,
    load_training_checkpoint, main,
)


class OfflineEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed_dim = 512
        self.embedding = nn.Embedding(8, 512)


def make_model(phase=2, dataset="disfa"):
    cfg = ModelConfig()
    cfg.dataset_name = dataset
    cfg.au_ids = get_dataset_aus(dataset)
    cfg.au_expression_prior = get_dataset_au_expression_prior(dataset)
    cfg.backbone_feat_dim = 32
    cfg.gat_hidden_dim = 16
    cfg.phase = phase
    with patch("models.ctrlau.resnet50", return_value=nn.Identity()), patch(
        "models.ctrlau.TextEncoder", OfflineEncoder
    ):
        model = CtrlAUModel(cfg)
    args = types.SimpleNamespace(phase=phase, model="ctrlau", lr=1e-4)
    with contextlib.redirect_stdout(io.StringIO()):
        configure_training_phase(model, args)
    model._cached_text_emb = torch.randn(model.num_aus, 512)
    model._cached_emotion_text_emb = torch.randn(model.num_emotions, 512)
    return model


class GraphPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_phase2_and_phase3_backward_freezing_and_call_counts(self):
        for dataset in ("disfa", "bp4d"):
            for phase in (2, 3):
                with self.subTest(dataset=dataset, phase=phase):
                    model = make_model(phase, dataset)
                    model.train()
                    keep_frozen_modules_eval(model)
                    frozen = {k: p.clone() for k, p in model.named_parameters() if not p.requires_grad}
                    bn = {k: b.clone() for k, b in model.named_buffers() if "running_" in k}
                    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                    x = torch.randn(2, 4, 32)
                    y = torch.randint(0, 2, (2, model.num_aus)).float()
                    with patch.object(model.graph_module, "forward_au_au", wraps=model.graph_module.forward_au_au) as aa, patch.object(
                        model.graph_module, "forward_au_exp", wraps=model.graph_module.forward_au_exp
                    ) as ae, patch.object(model, "_get_text_embeddings", wraps=model._get_text_embeddings) as text:
                        out = model(x, au_labels=y)
                    self.assertEqual((aa.call_count, ae.call_count), (1, 1) if phase == 2 else (3, 3))
                    self.assertEqual(text.call_count, 0 if phase == 2 else 1)
                    self.assertEqual(out["graph_au_probs"].shape, y.shape)
                    self.assertFalse(any("cycle" in k or "support" in k for k in out["losses"]))
                    self.assertEqual("loss_cf_important" in out["losses"], phase == 3)
                    loss = out["losses"]["total_loss"]
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for name, p in model.named_parameters():
                        if name in frozen:
                            self.assertIsNone(p.grad, name)
                        if p.grad is not None:
                            self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(model.graph_module.au_au_adj.grad.abs().sum().item(), 0)
                    self.assertGreater(model.graph_module.au_to_expr_adj.grad.abs().sum().item(), 0)
                    if phase == 3:
                        self.assertGreater(model.cf_threshold.grad.abs().item(), 0)
                        self.assertGreater(model.cf_threshold_exp.grad.abs().item(), 0)
                    optim.step()
                    for name, p in model.named_parameters():
                        if name in frozen:
                            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
                    for name, b in model.named_buffers():
                        if name in bn:
                            torch.testing.assert_close(b, bn[name], rtol=0, atol=0)

    def test_phase_switch_and_phase1_exclusion(self):
        model = make_model(3)
        model.set_phase(1)
        self.assertEqual(model.cfg.phase, 1)
        with patch.object(model.graph_module, "forward_au_au", side_effect=AssertionError("graph in phase 1")):
            out = model(torch.randn(2, 4, 32), au_labels=torch.randint(0, 2, (2, 8)).float())
        self.assertNotIn("graph_au_probs", out)
        self.assertFalse(any("facs" in k or "cf_" in k for k in out["losses"]))
        with self.assertRaises(ValueError):
            model.set_phase(5)

    def test_no_expression_to_au_message_or_refinement_dependency(self):
        model = make_model(2)
        au = torch.randn(2, 8, 256, requires_grad=True)
        expr = torch.randn(2, 8, 256, requires_grad=True)
        nodes, _ = model.graph_module.forward_au_exp(au, expr)
        reverse = torch.autograd.grad(nodes[:, :8].sum(), expr, allow_unused=True)[0]
        self.assertTrue(reverse is None or torch.count_nonzero(reverse) == 0)
        model.eval()
        model.cfg.enable_test_time_refinement = True
        x = torch.randn(2, 4, 32)
        with torch.no_grad():
            first = model(x)["graph_au_probs"]
            for classifier in model.graph_emo_classifiers:
                classifier.bias.add_(10)
            second = model(x)["graph_au_probs"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_semantic_pullback_is_coordinate_consistent_and_bounded(self):
        model = make_model(3)
        # A known linear projection makes the raw-space gradient analytic.
        model.visual_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.visual_proj.weight.copy_(torch.tensor([[2., 1.], [0., 3.]]))
        h = torch.tensor([[[1., 2.], [3., 1.]]], requires_grad=True)
        text = torch.tensor([[1., 0.], [0., 1.]])
        components = model._semantic_components(h, text)
        raw_dirs = torch.nn.functional.normalize(text @ model.visual_proj.weight, dim=-1)
        expected = (h * raw_dirs).sum(-1, keepdim=True) * raw_dirs
        torch.testing.assert_close(components, expected)
        with torch.no_grad():
            rotation = torch.tensor([[0., -1.], [1., 0.]])
            model.visual_proj.weight.copy_(rotation @ model.visual_proj.weight)
        rotated = model._semantic_components(h, text @ rotation.t())
        torch.testing.assert_close(rotated, components)
        changed = model._intervene(h, components, torch.ones(1, 2), 1.)
        torch.testing.assert_close((changed[:, 0] * raw_dirs).sum(-1), torch.zeros(1, 2), atol=1e-6, rtol=0)
        changed.sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        for strength in (-1., 5., float("nan")):
            with self.assertRaises(ValueError):
                model._intervene(h, components, torch.ones(1, 2), strength)
        with torch.no_grad():
            model.visual_proj.weight.zero_()
        with self.assertRaisesRegex(RuntimeError, "zero"):
            model._semantic_components(h, text)

    def test_continuous_masks_and_disabled_phase3_skip(self):
        model = make_model(3)
        with torch.no_grad():
            model.graph_module.au_au_adj[0, 1] = .1
            model.graph_module.au_au_adj[1, 0] = .2
        x = torch.randn(2, 4, 32)
        y = torch.randint(0, 2, (2, 8)).float()
        with patch.object(model, "_intervene", wraps=model._intervene) as intervene:
            model(x, au_labels=y)
        mask = intervene.call_args_list[0].args[2]
        self.assertNotEqual(mask[0, 1].item(), mask[1, 0].item())
        for key in ("lambda_cf_au_imp", "lambda_cf_au_unimp", "lambda_cf_exp_imp", "lambda_cf_exp_unimp"):
            setattr(model.cfg, key, 0.)
        with patch.object(model, "_semantic_components", side_effect=AssertionError("disabled CF executed")):
            out = model(x, au_labels=y)
        self.assertNotIn("loss_cf_important", out["losses"])

    def test_checkpoint_promotion_and_rejection(self):
        model = make_model(2)
        payload = {
            "state_dict": copy.deepcopy(model.state_dict()), "phase": 1,
            "fold": 2, "dataset": "disfa", "au_ids": list(model.au_ids),
            "model": "ctrlau", "num_emotions": 8, "epoch": 5,
            "graph_topology": "au_to_expression",
        }
        payload["state_dict"]["cycle_loss.rule_matrix"] = model.au_expression_support.clone()
        payload["state_dict"]["au_expression_support_loss.support_matrix"] = model.au_expression_support.clone()

        def load(data, phase=2):
            stream = io.BytesIO()
            torch.save(data, stream)
            stream.seek(0)
            return load_training_checkpoint(model, stream, phase=phase, fold=2, dataset="disfa", model_name="ctrlau", device="cpu")

        load(payload)
        torch.testing.assert_close(model.graph_au_classifiers[0].weight, model.au_head.classifiers[0].weight)
        bad = []
        for key, value in (("fold", 1), ("dataset", "bp4d"), ("model", "symgraphau"),
                           ("num_emotions", 7), ("graph_topology", "bidirectional"),
                           ("au_ids", list(reversed(model.au_ids)))):
            item = copy.deepcopy(payload)
            item[key] = value
            bad.append(item)
        missing = copy.deepcopy(payload)
        del missing["state_dict"]["global_linear.fc.weight"]
        bad.append(missing)
        bad_weights = copy.deepcopy(payload)
        bad_weights["state_dict"]["wal_weights"].fill_(2.)
        bad.append(bad_weights)
        wrong_shape = copy.deepcopy(payload)
        wrong_shape["state_dict"]["global_linear.fc.weight"] = torch.ones(1)
        bad.append(wrong_shape)
        bad.append({"state_dict": payload["state_dict"]})
        for item in bad:
            with self.subTest(metadata=set(item)), self.assertRaises(ValueError):
                load(item)
        with self.assertRaisesRegex(ValueError, "Cannot enter"):
            load(payload, phase=3)
        stage2 = copy.deepcopy(payload)
        stage2["phase"] = 2
        for key in ("cycle_loss.rule_matrix", "au_expression_support_loss.support_matrix"):
            stage2["state_dict"].pop(key)
        with self.assertRaisesRegex(ValueError, "Old graph-stage"):
            load(stage2, phase=3)
        stage2["training_schema"] = TRAINING_SCHEMA
        load(stage2, phase=3)
        with self.assertRaisesRegex(ValueError, "best_f1"):
            load(stage2)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        stage2.update(best_f1=.6513, optimizer=optimizer.state_dict())
        restored = load(stage2)
        self.assertEqual(restored["best_f1"], .6513)

    def test_cli_rejects_missing_checkpoint_and_overshoot_before_dataset_loading(self):
        for argv in (["train.py", "--phase", "2"], ["train.py", "--cf-scale", "5"]):
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    main()
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
