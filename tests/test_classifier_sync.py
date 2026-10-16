"""Classifier synchronization must preserve the training state."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from config import ModelConfig
from models.ctrlau import CtrlAUModel


class OfflineTextEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed_dim = 512
        self.embedding = nn.Embedding(8, self.embed_dim)


class ClassifierSyncTests(unittest.TestCase):
    def setUp(self):
        cfg = ModelConfig()
        cfg.backbone_feat_dim = 32
        cfg.gat_hidden_dim = 16
        with patch("models.ctrlau.resnet50", return_value=nn.Identity()), patch(
            "models.ctrlau.TextEncoder", OfflineTextEncoder
        ):
            self.model = CtrlAUModel(cfg)

    def test_sync_preserves_fold_weights_and_loss_modules(self):
        model = self.model
        model.update_class_weights(torch.tensor([1.8, 2.0, .6, 1., 1.9, .5, .1, .1]))
        expected = model.wal_weights.clone()
        weight_buffer = model.wal_weights
        modules = {
            name: module for name, module in model.named_children()
            if "loss" in name
        }
        for _ in range(2):
            model.init_graph_classifiers_from_base()
            self.assertIs(model.wal_weights, weight_buffer)
            self.assertIs(model.wal_loss.weight, weight_buffer)
            torch.testing.assert_close(model.wal_weights, expected)
            for name, module in modules.items():
                self.assertIs(getattr(model, name), module)

    def test_sync_preserves_thresholds_and_optimizer_references(self):
        model = self.model
        model.set_phase(3)
        with torch.no_grad():
            model.cf_threshold.fill_(.27)
            model.cf_threshold_exp.fill_(.73)
        thresholds = [model.cf_threshold, model.cf_threshold_exp]
        optimizer = torch.optim.SGD(thresholds, lr=.1)
        model.init_graph_classifiers_from_base()
        self.assertIs(model.cf_threshold, thresholds[0])
        self.assertIs(model.cf_threshold_exp, thresholds[1])
        torch.testing.assert_close(model.cf_threshold, torch.tensor(.27))
        torch.testing.assert_close(model.cf_threshold_exp, torch.tensor(.73))
        optimizer.zero_grad()
        (model.cf_threshold.square() + model.cf_threshold_exp.square()).backward()
        optimizer.step()
        self.assertLess(model.cf_threshold.item(), .27)
        self.assertLess(model.cf_threshold_exp.item(), .73)

    def test_sync_only_copies_classifiers(self):
        model = self.model
        model.set_phase(2)
        model.eval()
        cached_au = torch.randn(8, 512)
        cached_expr = torch.randn(8, 512)
        model._cached_text_emb = cached_au
        model._cached_emotion_text_emb = cached_expr
        trainability = {name: p.requires_grad for name, p in model.named_parameters()}
        with torch.no_grad():
            for classifier in model.au_head.classifiers:
                classifier.weight.add_(1.)
            for classifier in model.emotion_head.classifiers:
                classifier.bias.add_(1.)
        model.init_graph_classifiers_from_base()
        self.assertIs(model._cached_text_emb, cached_au)
        self.assertIs(model._cached_emotion_text_emb, cached_expr)
        self.assertFalse(model.training)
        self.assertEqual(model.phase, 2)
        self.assertEqual(
            {name: p.requires_grad for name, p in model.named_parameters()}, trainability
        )
        for source, destinations in [
            (model.au_head.classifiers, [model.au_au_classifiers, model.graph_au_classifiers]),
            (model.emotion_head.classifiers, [model.graph_emo_classifiers]),
        ]:
            for destination in destinations:
                for original, copied in zip(source, destination):
                    torch.testing.assert_close(original.weight, copied.weight)
                    torch.testing.assert_close(original.bias, copied.bias)


if __name__ == "__main__":
    unittest.main()
