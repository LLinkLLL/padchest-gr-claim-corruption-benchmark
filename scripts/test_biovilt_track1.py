"""Deterministic toy tests for the frozen, inference-free heatmap protocol."""
import math
import unittest

import numpy as np

from benchmark_common import clustered_bootstrap_interval
from analyse_track1 import target_zone_hit
from biovilt_track1_metrics import (geometry, peak_info, native_metrics, classify, point_hits,
                                   point_in_zone, union_area, cluster_mean_ci)


class GeometryTests(unittest.TestCase):
    def test_overlap_union_and_partial_tile(self):
        boxes = [(0.10, 0.10, 0.20, 0.20), (0.15, 0.15, 0.25, 0.25)]
        self.assertAlmostEqual(union_area(boxes), 0.0175)
        g = geometry(512, 512, (2, 2), boxes, "laterality", "right", "left")
        self.assertAlmostEqual(float(g["foreground"].sum()), 0.0175)
        self.assertAlmostEqual(float((g["foreground"] + g["background"]).sum()), float(g["total"].sum()))
        tiny = geometry(512, 512, (2, 2), [(0.10, 0.10, 0.11, 0.11)], "laterality", "right", "left")
        self.assertGreater(tiny["foreground"].sum(), 0)
        self.assertFalse(point_hits((0.28125, 0.28125), [(0.10, 0.10, 0.11, 0.11)]))

    def test_crop_and_coordinates(self):
        g = geometry(1000, 1200, (14, 14), [(0.2, 0.2, 0.3, 0.3)], "vertical", "upper", "lower")
        self.assertEqual(g["crop_size"], 875)
        self.assertEqual((g["crop_left"], g["crop_top"]), (62, 162))
        h = np.zeros((14, 14)); h[3, 9] = 1
        p = peak_info(h, g)
        x, y = p["point"]
        self.assertAlmostEqual((x * 1000 - 62) / 875 * 14, 9.5)
        self.assertAlmostEqual((y * 1200 - 162) / 875 * 14, 3.5)
        self.assertEqual(g["crop_status"], "fully_visible")

    def test_fractional_zone_split(self):
        g = geometry(513, 513, (3, 3), [(0.1, 0.1, 0.2, 0.2)], "laterality", "right", "left")
        np.testing.assert_allclose(g["target"] + g["original"], g["total"], atol=1e-14)
        self.assertTrue(np.any((g["target"] > 0) & (g["original"] > 0)))

    def test_peak_ties_constant_and_reproducibility(self):
        g = geometry(512, 512, (2, 2), [(0.1, 0.1, 0.2, 0.2)], "laterality", "right", "left")
        h = np.array([[0.3, 0.8], [0.8, 0.1]])
        first, second = peak_info(h, g), peak_info(h.copy(), g)
        self.assertEqual(first["point"], second["point"])
        self.assertEqual((first["peak_row"], first["peak_col"]), (0, 1))
        self.assertEqual(first["peak_tie_count"], 2)
        self.assertEqual(math.dist(first["point"], second["point"]), 0)
        self.assertIsNone(peak_info(np.ones((2, 2)), g)["point"])

    def test_zone_compatibility_and_midlines(self):
        for kind, targets in [("laterality", ("left", "right")), ("vertical", ("upper", "lower", "apical", "basal"))]:
            for target in targets:
                for x in (0.1, 0.5, 0.9):
                    for y in (0.1, 0.5, 0.9):
                        self.assertEqual(point_in_zone((x, y), kind, target), target_zone_hit([x, y, x, y], kind, target))

    def test_taxonomy_and_gt_boundary(self):
        gt = [(0.1, 0.1, 0.3, 0.3)]
        self.assertTrue(point_hits((0.1, 0.3), gt))
        self.assertEqual(classify(True, (0.2, 0.2), gt, "laterality", "left")[0], "source_evidence_adherence")
        self.assertEqual(classify(True, (0.8, 0.2), gt, "laterality", "left")[0], "corrupted_text_following")
        self.assertEqual(classify(True, (0.4, 0.4), gt, "laterality", "left")[0], "other_mixed")
        self.assertEqual(classify(True, (0.6, 0.2), [(0.4, 0.1, 0.7, 0.3)], "laterality", "left")[0], "other_mixed")
        self.assertEqual(classify(True, None, gt, "laterality", "left")[0], "other_mixed")
        self.assertEqual(classify(False, None, gt, "laterality", "left")[0], "invalid")


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.g = geometry(512, 512, (14, 14), [(0.1, 0.1, 0.4, 0.4)], "laterality", "right", "left")
        self.h = np.linspace(0.15, 0.75, 196).reshape(14, 14)

    def test_identity_and_additive_shift(self):
        a = native_metrics(self.h, self.g)
        same = native_metrics(self.h.copy(), self.g)
        b = native_metrics(self.h - 0.3, self.g)
        for key in ("signed_cnr", "mu_gt", "zone_contrast", "top5_mean", "map_mean"):
            self.assertAlmostEqual(a[key], same[key], places=12)
        for key in ("signed_cnr", "zone_contrast"):
            self.assertAlmostEqual(a[key], b[key], places=12)
        for key in ("mu_gt", "mu_background", "top5_mean", "map_mean"):
            self.assertAlmostEqual(b[key] - a[key], -0.3, places=12)
        self.assertEqual(peak_info(self.h, self.g)["point"], peak_info(self.h - 0.3, self.g)["point"])
        self.assertAlmostEqual(np.corrcoef(self.h.ravel(), (self.h - 0.3).ravel())[0, 1], 1)

    def test_scale_and_target_change(self):
        a = native_metrics(self.h, self.g)
        b = native_metrics(self.h * 0.5, self.g)
        self.assertAlmostEqual(a["signed_cnr"], b["signed_cnr"], places=12)
        self.assertAlmostEqual(a["top5_mean"] * 0.5, b["top5_mean"], places=12)
        shifted = self.h.copy(); shifted[:, 7:] += 0.1
        c = native_metrics(shifted, self.g)
        self.assertGreater(c["zone_contrast"], a["zone_contrast"])

    def test_top_k_and_constant(self):
        a = native_metrics(self.h, self.g)
        self.assertEqual(a["top_k"], 10)
        self.assertAlmostEqual(a["top5_mean"], np.sort(self.h.ravel())[-10:].mean())
        b = native_metrics(np.full((14, 14), -0.2), self.g)
        self.assertTrue(math.isnan(b["signed_cnr"]))
        self.assertAlmostEqual(b["top5_mean"], -0.2)
        self.assertAlmostEqual(b["map_mean"], -0.2)
        self.assertAlmostEqual(b["zone_contrast"], 0)

    def test_invisible_gt_and_no_background(self):
        g = geometry(512, 512, (14, 14), [(0.0, 0.0, 0.01, 0.01)], "laterality", "right", "left")
        a = native_metrics(self.h, g)
        self.assertEqual(g["crop_status"], "fully_outside")
        self.assertTrue(math.isnan(a["mu_gt"]))
        self.assertTrue(math.isnan(a["signed_cnr"]))
        for key in ("zone_contrast", "top5_mean", "map_mean"):
            self.assertTrue(math.isfinite(a[key]))
        g = geometry(512, 512, (14, 14), [(0, 0, 1, 1)], "laterality", "right", "left")
        a = native_metrics(self.h, g)
        self.assertTrue(math.isnan(a["signed_cnr"]))
        self.assertTrue(math.isfinite(a["mu_gt"]))
        self.assertEqual(a["reasons"]["signed_cnr"], "no_visible_background")


class BootstrapTests(unittest.TestCase):
    def test_equivalent_to_shared_with_repeated_patients(self):
        rows = [{"patient_id": "a", "v": 1.0}, {"patient_id": "a", "v": 0.0},
                {"patient_id": "b", "v": 1.0}, {"patient_id": "c", "v": 0.0}]
        stat = cluster_mean_ci(rows, lambda r: r["v"], seed=2026, replicates=1000)
        expected = clustered_bootstrap_interval(rows, "patient_id", lambda rs: sum(r["v"] for r in rs) / len(rs), 1000, 2026)
        np.testing.assert_allclose([stat["ci_low"], stat["ci_high"]], expected, atol=1e-12)
        self.assertEqual(stat["n_valid_pairs"], 4)
        self.assertEqual(stat["n_valid_patients"], 3)

    def test_missing_not_zero_and_all_missing(self):
        rows = [{"patient_id": "a", "v": float("nan")}, {"patient_id": "b", "v": 0.4}]
        stat = cluster_mean_ci(rows, lambda r: r["v"], seed=2026)
        self.assertAlmostEqual(stat["mean"], 0.4)
        self.assertEqual(stat["n_valid_pairs"], 1)
        self.assertLess(stat["valid_replicates"], 1000)
        stat = cluster_mean_ci(rows[:1], lambda r: r["v"], seed=2026)
        self.assertTrue(math.isnan(stat["mean"]))
        self.assertEqual(stat["valid_replicates"], 0)


if __name__ == "__main__":
    unittest.main()
