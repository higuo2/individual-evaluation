import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fyp_eval import (ROOT_FIELDS, SAMPLE_FIELDS, actual_area_band, audit,
                      mask_measurement, measure_samples, read_csv,
                      read_grayscale_mask, write_csv)


class MeasurementTests(unittest.TestCase):
    def test_area_boundaries(self):
        self.assertEqual(actual_area_band(.02), "(0,0.02]")
        self.assertEqual(actual_area_band(.020001), "(0.02,0.05]")
        self.assertEqual(actual_area_band(.05), "(0.02,0.05]")
        self.assertEqual(actual_area_band(.4), "(0.2,0.4]")
        self.assertEqual(actual_area_band(1), "(0.4,1]")

    def test_empty_and_one_pixel_masks(self):
        empty = np.zeros((10, 10), dtype=np.uint8)
        self.assertEqual(mask_measurement(empty, (10, 10))["location"], "excluded_empty_mask")
        empty[2, 3] = 255
        self.assertEqual(mask_measurement(empty, (10, 10))["r"], .01)
        self.assertEqual(mask_measurement(np.full((10, 10), 255, dtype=np.uint8),
                                          (10, 10))["location"], "excluded_full_image_mask")

    def test_location(self):
        operation = np.zeros((10, 10), dtype=np.uint8)
        operation[0:2, 0:5] = 255
        foreground = np.zeros_like(operation)
        foreground[0:2, 0:3] = 255
        self.assertEqual(mask_measurement(operation, (10, 10), foreground)["location"], "foreground")
        foreground[:] = 0
        foreground[0:2, 0:1] = 255
        self.assertEqual(mask_measurement(operation, (10, 10), foreground)["location"], "background")
        foreground[0:2, 0:3] = 255
        foreground[0, 2] = 0
        self.assertEqual(mask_measurement(operation, (10, 10), foreground)["location"], "mixed")

    def test_geometry_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            mask_measurement(np.zeros((9, 10)), (10, 10))

    def test_browser_png_mask_is_accepted_only_if_opaque_monochrome(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            pixels = np.zeros((2, 2, 4), dtype=np.uint8)
            pixels[:, :, 3] = 255
            pixels[0, 0, :3] = 255
            Image.fromarray(pixels, "RGBA").save(path)
            self.assertEqual(int(read_grayscale_mask(path)[0, 0]), 255)
            pixels[0, 1, :3] = (255, 0, 0)
            Image.fromarray(pixels, "RGBA").save(path)
            with self.assertRaisesRegex(ValueError, "identical"):
                read_grayscale_mask(path)

    def test_manifest_and_real_image_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            Image.new("RGB", (10, 10), "white").save(d / "edited.png")
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[0:2, 0:5] = 255
            Image.fromarray(mask).save(d / "mask.png")
            foreground = np.zeros((10, 10), dtype=np.uint8)
            foreground[0:2, 0:3] = 255
            Image.fromarray(foreground).save(d / "fg.png")
            roots = d / "roots.csv"
            samples = d / "samples.csv"
            write_csv(roots, [{"root_id": "r1", "split": "development",
                               "candidate_cluster": "r1", "sha256": "a" * 64,
                               "width": 10, "height": 10,
                               "foreground_status": "valid",
                               "foreground_mask_path": "fg.png"}], ROOT_FIELDS)
            write_csv(samples, [{"sample_id": "s1", "root_id": "r1",
                                 "split": "development",
                                 "edited_image_path": "edited.png",
                                 "operation_mask_path": "mask.png"}], SAMPLE_FIELDS)
            result = measure_samples(roots, samples, d / "out.csv")
            self.assertEqual(result["valid"], 1)
            row = read_csv(d / "out.csv")[0]
            self.assertEqual(row["location"], "foreground")
            self.assertEqual(float(row["r"]), .1)
            self.assertTrue(audit(roots, samples)["pass"])
            Image.fromarray(np.zeros((9, 10), dtype=np.uint8)).save(d / "mask.png")
            self.assertEqual(measure_samples(roots, samples, d / "out2.csv")["excluded"], 1)


if __name__ == "__main__":
    unittest.main()
