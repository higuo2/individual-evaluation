import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import analyze, threshold_at_fpr


class AnalysisTests(unittest.TestCase):
    def test_threshold_uses_validation_and_never_exceeds_target(self):
        threshold, fpr = threshold_at_fpr([float(x) for x in range(60)])
        self.assertEqual(threshold, 56)
        self.assertEqual(fpr, .05)

    def test_area_and_paired_location(self):
        rows = []
        for i in range(20):
            rows.append(dict(detector="D", root_id=f"v{i}", sample_id=f"v{i}",
                             split="validation", kind="pristine", score=float(i) / 100,
                             area_band="", location=""))
        for i in range(2):
            root = f"t{i}"
            rows.append(dict(detector="D", root_id=root, sample_id=root,
                             split="test", kind="pristine", score=0.1,
                             area_band="", location=""))
            rows.append(dict(detector="D", root_id=root, sample_id=root + "f",
                             split="test", kind="edited", score=.9,
                             area_band="(0,0.02]", location="foreground"))
            rows.append(dict(detector="D", root_id=root, sample_id=root + "b",
                             split="test", kind="edited", score=.05,
                             area_band="(0,0.02]", location="background"))
        report = analyze(rows, bootstrap_repeats=10)["D"]
        self.assertEqual(report["area"]["(0,0.02]"]["edited_images"], 4)
        self.assertEqual(report["area"]["(0,0.02]"]["unique_roots"], 2)
        self.assertEqual(report["paired_location"]["(0,0.02]"]["paired_roots"], 2)
        self.assertEqual(report["paired_location"]["(0,0.02]"]["foreground_minus_background_detection_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
