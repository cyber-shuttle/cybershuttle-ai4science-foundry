import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
WORKFLOW_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

from run_experiment import cosine_retrieve, run_experiment, weighted_rmse
import argparse


class AnalogExperimentTests(unittest.TestCase):
    def test_cosine_retrieval_and_rmse(self):
        indices, scores = cosine_retrieve(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            k=2,
        )
        self.assertEqual(indices.tolist(), [[0, 1], [1, 0]])
        self.assertAlmostEqual(float(scores[0, 0]), 1.0)
        prediction = np.ones((2, 2, 2), dtype=np.float32)
        target = np.zeros_like(prediction)
        self.assertAlmostEqual(
            weighted_rmse(prediction, target, np.array([60.0, -60.0])), 1.0
        )

    def test_synthetic_raw_and_embedding_experiment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            (data / "train").mkdir(parents=True)
            (data / "test").mkdir()
            train = np.arange(30 * 2 * 3, dtype=np.float32).reshape(30, 1, 2, 3)
            test = np.arange(30 * 2 * 3, dtype=np.float32).reshape(30, 1, 2, 3)
            np.savez_compressed(data / "train" / "2013.npz", geopotential_500=train)
            np.savez_compressed(data / "test" / "2016.npz", geopotential_500=test)
            np.savez_compressed(
                data / "train" / "climatology.npz",
                geopotential_500=np.full((1, 2, 3), 10.0, dtype=np.float32),
            )
            np.save(data / "lat.npy", np.array([45.0, -45.0], dtype=np.float32))
            np.save(data / "lon.npy", np.array([0.0, 120.0, 240.0], dtype=np.float32))

            np.savez_compressed(
                root / "train_embeddings.npz",
                pooled_embeddings=np.ones((30, 4), dtype=np.float32),
                sample_indices=np.arange(30, dtype=np.int64),
                variable=np.asarray(["geopotential_500"]),
            )
            np.savez_compressed(
                root / "test_embeddings.npz",
                pooled_embeddings=np.ones((2, 4), dtype=np.float32),
                sample_indices=np.arange(0, 2, dtype=np.int64),
                variable=np.asarray(["geopotential_500"]),
            )

            args = argparse.Namespace(
                data_dir=data,
                output_dir=root / "out",
                variable="geopotential_500",
                lead_hours=12,
                time_step_hours=6,
                reference_split="train",
                evaluation_split="test",
                time_index=0,
                num_samples=2,
                sample_stride_hours=6,
                k_values=[1, 3],
                reference_embeddings=root / "train_embeddings.npz",
                evaluation_embeddings=root / "test_embeddings.npz",
                seed=0,
            )
            metrics = run_experiment(args)
            self.assertEqual(len(metrics["results"]), 6)
            self.assertEqual(metrics["configuration"]["lead_hours"], 12)
            self.assertTrue((root / "out" / "forecast-comparison.svg").is_file())
            self.assertTrue((root / "out" / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
