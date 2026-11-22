import json
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from baseline_forecast import run_baseline, weighted_rmse
from forecast_visualization import write_forecast_svg


class BaselineForecastTests(unittest.TestCase):
    def test_baseline_writes_metrics_archive_and_svg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            (data_dir / "test").mkdir(parents=True)
            (data_dir / "train").mkdir()
            values = np.arange(8 * 2 * 3, dtype=np.float32).reshape(8, 1, 2, 3)
            np.savez_compressed(data_dir / "test" / "2016_0.npz", geopotential_500=values)
            np.savez_compressed(
                data_dir / "train" / "climatology.npz",
                geopotential_500=np.full((1, 2, 3), 5.0, dtype=np.float32),
            )
            np.save(data_dir / "lat.npy", np.array([45.0, -45.0], dtype=np.float32))
            np.save(data_dir / "lon.npy", np.array([0.0, 120.0, 240.0], dtype=np.float32))
            output_dir = root / "out"

            metrics = run_baseline(
                data_dir=data_dir,
                output_dir=output_dir,
                lead_hours=2,
                num_samples=2,
                sample_stride=2,
            )

            self.assertEqual(metrics["kind"], "baseline_only_not_a_ClimaX_forecast")
            self.assertEqual(metrics["evaluation"]["prediction_shape"], [2, 1, 2, 3])
            self.assertEqual(metrics["evaluation"]["target_indices"], [2, 4])
            svg_path = output_dir / "baseline-forecast.svg"
            self.assertTrue(svg_path.is_file())
            self.assertEqual(
                ElementTree.parse(svg_path).getroot().tag,
                "{http://www.w3.org/2000/svg}svg",
            )
            with np.load(output_dir / "baseline-forecast.npz") as archive:
                self.assertEqual(archive["persistence"].shape, (2, 1, 2, 3))
                self.assertEqual(archive["climatology"].shape, (2, 1, 2, 3))
            saved_metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(saved_metrics["evaluation"]["split"], "test")

    def test_rejects_forecast_samples_that_exceed_shard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "test").mkdir()
            (root / "train").mkdir()
            np.savez(root / "test" / "2016_0.npz", geopotential_500=np.zeros((4, 1, 2, 3)))
            np.savez(
                root / "train" / "climatology.npz",
                geopotential_500=np.zeros((1, 2, 3)),
            )
            np.save(root / "lat.npy", np.array([30.0, -30.0]))
            np.save(root / "lon.npy", np.array([0.0, 180.0, 300.0]))

            with self.assertRaisesRegex(ValueError, "shard has 4 time steps"):
                run_baseline(
                    data_dir=root,
                    output_dir=root / "out",
                    lead_hours=3,
                    num_samples=2,
                    sample_stride=1,
                )

    def test_latitude_weighted_rmse(self):
        prediction = np.ones((2, 2, 2), dtype=np.float32)
        target = np.zeros_like(prediction)
        score = weighted_rmse(prediction, target, np.array([60.0, -60.0]))
        self.assertAlmostEqual(score, 1.0)

    def test_svg_rejects_mismatched_panel_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "same two-dimensional shape"):
                write_forecast_svg(
                    Path(temp_dir) / "plot.svg",
                    [("first", np.zeros((2, 2))), ("second", np.zeros((2, 3)))],
                )


if __name__ == "__main__":
    unittest.main()
