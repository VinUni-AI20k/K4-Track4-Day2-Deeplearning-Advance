"""test_codes.py - kiểm tra tự viết cho các phần dễ sai của codes/ (RUBRIC mục H).

[Implemented by Claude (AI assistant)]
Chạy (từ thư mục codes/):  python -m unittest test_codes -v
Không cần GPU, không tải trọng số (timm tạo model với pretrained=False).

Mỗi lớp test ứng với một file:
  TestLosses     -> losses.py    (focal gamma=0 == CE, label smoothing, class weights, Mixup/CutMix)
  TestModel      -> model.py     (3 nhóm tham số, không decay norm/bias, đóng băng + BN ở eval)
  TestTrain      -> train.py     (warmup + cosine, EMA, parse_overrides, đường dẫn file)
  TestInference  -> inference.py (gộp BN chính xác, TTA, gộp view, temperature scaling, ensemble)
  TestBenchmark  -> benchmark.py (percentile)
  TestContract   -> định dạng predictions đọc lại được bằng eval.py gốc
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
import inference  # noqa: E402
import losses  # noqa: E402
import model as model_mod  # noqa: E402
import train  # noqa: E402  (train.py tự thêm repo gốc vào sys.path để import eval.py)
import eval as ev  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(64, 9) * 3
        self.y = torch.randint(0, 9, (64,))

    def test_focal_gamma0_equals_ce(self):
        fl = losses.FocalLoss(gamma=0.0)(self.logits, self.y)
        ce = F.cross_entropy(self.logits, self.y)
        self.assertLess(abs(float(fl - ce)), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(float(losses.FocalLoss(gamma=2.0)(self.logits, self.y)),
                        float(F.cross_entropy(self.logits, self.y)))

    def test_label_smoothing(self):
        self.assertLess(abs(float(losses.LabelSmoothingCE(0.0)(self.logits, self.y)
                                  - F.cross_entropy(self.logits, self.y))), 1e-6)
        ours = losses.LabelSmoothingCE(0.1)(self.logits, self.y)
        ref = F.cross_entropy(self.logits, self.y, label_smoothing=0.1)
        self.assertLess(abs(float(ours - ref)), 1e-5)

    def test_class_weights(self):
        counts = [675, 638, 618, 613, 637, 605, 644, 609, 5463]
        w = losses.class_weights(counts).numpy()
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)
        self.assertAlmostEqual(float(w[0] * counts[0]), float(w[8] * counts[8]), places=3)  # ~ 1/n
        wb = losses.class_weights(counts, beta=0.999).numpy()
        self.assertAlmostEqual(float(wb.sum()), 9.0, places=4)
        self.assertGreater(wb[0], wb[8])

    def test_build_criterion_kinds(self):
        w = losses.class_weights([10] * 9)
        for kind in ("ce", "ls", "focal", "ce_weighted"):
            crit = losses.build_criterion(kind, smoothing=0.1, gamma=2.0, weight=w)
            self.assertTrue(torch.isfinite(crit(self.logits, self.y)))
        # trọng số đều => CE có trọng số == CE thường
        self.assertLess(abs(float(losses.build_criterion("ce_weighted", weight=w)(self.logits, self.y)
                                  - F.cross_entropy(self.logits, self.y))), 1e-6)

    def test_cutmix_lambda_matches_pixel_fraction(self):
        np.random.seed(1)
        torch.manual_seed(1)
        x = torch.zeros(8, 3, 32, 32)
        for i in range(8):
            x[i] = i  # mỗi ảnh một giá trị hằng để biết pixel đến từ ảnh nào
        y = torch.arange(8)
        for _ in range(20):
            xm, (ya, yb, lam) = losses.mix_batch(x, y, alpha=1.0, mode="cutmix")
            for i in range(8):
                frac_own = float((xm[i, 0] == x[i, 0, 0, 0]).float().mean())
                if yb[i] != ya[i]:
                    self.assertAlmostEqual(frac_own, lam, places=6)
            self.assertTrue(torch.equal(ya, y))

    def test_mixup_and_mixed_loss(self):
        np.random.seed(0)
        torch.manual_seed(0)
        x = torch.randn(8, 3, 8, 8)
        y = torch.randint(0, 9, (8,))
        xm, (ya, yb, lam) = losses.mix_batch(x, y, alpha=0.4, mode="mixup")
        self.assertTrue(0.0 <= lam <= 1.0)
        self.assertEqual(xm.shape, x.shape)
        # x_mix = lam * x + (1 - lam) * x[perm]: kiểm tra với hoán vị suy ra từ chính x_mix
        self.assertTrue(torch.allclose(xm, lam * x + (1 - lam) * x[[int(((lam * x[i] + (1 - lam) * x - xm[i])
                                                                          .abs().amax(dim=(1, 2, 3))).argmin())
                                                                     for i in range(8)]], atol=1e-5))
        ce = nn.CrossEntropyLoss()
        logits = torch.randn(8, 9)
        self.assertAlmostEqual(float(losses.mixed_loss(ce, logits, (y, yb, 1.0))), float(ce(logits, y)), places=6)


class TestModel(unittest.TestCase):
    def test_param_groups_no_decay_on_norm_bias(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=9)
        groups = {g["name"]: g for g in model_mod.param_groups(m, 1e-4, 1e-3, 0.05)}
        self.assertTrue(all(p.ndim <= 1 for p in groups["backbone_no_decay"]["params"]))
        self.assertTrue(all(p.ndim > 1 for p in groups["backbone_decay"]["params"]))
        self.assertEqual(groups["backbone_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(groups["head_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(groups["head"]["lr"], 1e-3)
        n_total = sum(p.numel() for p in m.parameters())
        n_groups = sum(p.numel() for g in groups.values() for p in g["params"])
        self.assertEqual(n_total, n_groups)

    def test_frozen_backbone_keeps_bn_in_eval(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=9, init="frozen")
        groups = model_mod.param_groups(m, 1e-4, 1e-3, 0.05)
        self.assertEqual({g["name"] for g in groups}, {"head", "head_no_decay"})
        train.set_train_mode(m)
        self.assertTrue(m.training)
        bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]
        self.assertTrue(bns and all(not b.training for b in bns))
        before = bns[0].running_mean.clone()
        m(torch.randn(4, 3, 64, 64))
        self.assertTrue(torch.equal(before, bns[0].running_mean))

    def test_zero_head_gives_ln9_initial_loss(self):
        for name in ("mobilenetv3_large_100", "resnet18"):
            m = model_mod.build_model(name, pretrained=False, num_classes=9).eval()
            loss = F.cross_entropy(m(torch.randn(4, 3, 64, 64)), torch.randint(0, 9, (4,)))
            self.assertAlmostEqual(float(loss), float(np.log(9)), places=5)

    def test_count(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=9)
        self.assertAlmostEqual(model_mod.count_params(m), 11.18, delta=0.05)
        self.assertAlmostEqual(model_mod.count_gmacs(m, 224), 1.82, delta=0.05)  # ResNet-18 ~1.8 GMAC


class TestTrain(unittest.TestCase):
    def test_scheduler_warmup_then_cosine(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([{"params": [p], "lr": 1e-3}])
        cfg = train.Config(epochs=4, warmup_epochs=1)
        sch = train.build_scheduler(opt, cfg, steps_per_epoch=10)
        lrs = []
        for _ in range(40):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step(); sch.step()
        self.assertTrue(all(a < b for a, b in zip(lrs[:9], lrs[1:10])))  # warmup tăng
        self.assertAlmostEqual(max(lrs), 1e-3, places=9)
        self.assertLess(lrs[-1], 1e-5)                                  # cosine về ~0
        self.assertTrue(all(a >= b for a, b in zip(lrs[10:], lrs[11:])))

    def test_ema_update(self):
        m = nn.Linear(2, 2)
        ema = train.EMA(m, decay=0.9)
        with torch.no_grad():
            m.weight.add_(1.0)
        old = ema.module.weight.clone()
        ema.update(m)
        new = ema.module.weight
        self.assertTrue(torch.all(new > old) and torch.all(new < m.weight))

    def test_parse_overrides(self):
        d = train.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false", "lr_head=0.002",
                                   "sampler=balanced", "class_weight_beta=0.99"])
        self.assertEqual(d, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 0.002,
                             "sampler": "balanced", "class_weight_beta": 0.99})
        with self.assertRaises(ValueError):
            train.parse_overrides(["not_a_field=1"])

    def test_paths(self):
        cfg = train.Config(exp_id="F01", seed=2, pred_dir="predictions")
        self.assertEqual(train.pred_path(cfg, "test"), Path("predictions/F01_seed2_test.csv"))
        self.assertEqual(ev.parse_seed(str(train.pred_path(cfg, "test"))), 2)
        self.assertFalse(train.Config().save_test_predictions)


class TestInference(unittest.TestCase):
    def _randomize_bn(self, m):
        torch.manual_seed(0)
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.5, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
                mod.weight.data.uniform_(0.5, 1.5)
                mod.bias.data.uniform_(-0.2, 0.2)

    def test_fuse_conv_bn_resnet_and_mobilenet(self):
        for name in ("resnet18", "mobilenetv3_large_100", "efficientnet_b0"):
            m = model_mod.build_model(name, pretrained=False, num_classes=9).eval()
            self._randomize_bn(m)
            fused = inference.fuse_conv_bn(m)
            self.assertGreater(fused.n_fused_bn, 10, name)
            remaining = sum(isinstance(x, nn.BatchNorm2d) for x in fused.modules())
            self.assertEqual(remaining, 0, f"{name}: còn {remaining} BN")
            err = inference.check_fusion(m, fused, img_size=96)
            self.assertLess(err, 1e-3, f"{name}: sai số {err}")

    def test_fuse_noop_on_layernorm_model(self):
        m = model_mod.build_model("convnext_atto", pretrained=False, num_classes=9).eval()
        self.assertEqual(inference.fuse_conv_bn(m).n_fused_bn, 0)

    def test_views(self):
        x = torch.arange(2 * 3 * 10 * 10, dtype=torch.float32).reshape(2, 3, 10, 10)
        self.assertTrue(torch.equal(inference.view_hflip(x)[..., 0], x[..., -1]))
        crops = inference.views_multicrop(x, 8, flip=True)
        self.assertEqual(len(crops), 10)
        self.assertTrue(all(c.shape == (2, 3, 8, 8) for c in crops))
        self.assertTrue(torch.equal(crops[4], x[..., 1:9, 1:9]))
        scales = inference.views_multiscale(x, [10, 6, 12])
        self.assertEqual([s.shape[-1] for s in scales], [10, 6, 12])

    def test_aggregate_views(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(3, 50, 9))
        for space in ("prob", "logit"):
            p = inference.aggregate_views(list(z), space)
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
        np.testing.assert_allclose(inference.aggregate_views([z[0]], "prob"),
                                   inference.aggregate_views([z[0]], "logit"), atol=1e-12)

    def test_temperature_recovers_known_T(self):
        rng = np.random.default_rng(0)
        n, k, true_T = 20000, 9, 2.5
        base = rng.normal(size=(n, k)) * 2.0
        p = np.exp(base - base.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        y = np.array([rng.choice(k, p=row) for row in p])
        T = inference.fit_temperature(base * true_T, y)  # logit quá tự tin gấp true_T lần
        self.assertAlmostEqual(T, true_T, delta=0.1)
        probs = inference.apply_temperature(base * true_T, T)
        np.testing.assert_array_equal(probs.argmax(1), base.argmax(1))  # accuracy không đổi

    def test_ensemble(self):
        a = np.array([[0.9, 0.1], [0.2, 0.8]])
        b = np.array([[0.5, 0.5], [0.6, 0.4]])
        np.testing.assert_allclose(inference.ensemble_probs([a, b]), [[0.7, 0.3], [0.4, 0.6]])


class TestBenchmark(unittest.TestCase):
    def test_bench_percentiles(self):
        r = benchmark.bench(lambda: sum(range(1000)), warmup=3, iters=60)
        self.assertEqual(r["n"], 60)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])

    def test_latency_report_cpu(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=9)
        r = benchmark.latency_report(m, batch_size=1, img_size=64, device="cpu", warmup=2, iters=5)
        self.assertEqual(r["device"], "cpu")
        self.assertGreater(r["images_per_s"], 0)


class TestContract(unittest.TestCase):
    def test_saved_predictions_accepted_by_eval(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(30, 9))
        probs = train.softmax_np(logits)
        with tempfile.TemporaryDirectory() as d:
            path = ev.save_predictions(Path(d) / "T00_seed1_test.csv", [f"{i}.jpg" for i in range(30)],
                                       rng.integers(0, 9, 30), probs)
            pred = ev.read_pred(str(path))
            self.assertEqual(pred.seed, 1)
            np.testing.assert_allclose(pred.probs, probs, atol=1e-6)
        m = train.metrics_from_logits(np.zeros(30, int), logits)
        self.assertIn("macro_f1", m)


if __name__ == "__main__":
    unittest.main()
