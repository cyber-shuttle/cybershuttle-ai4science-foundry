import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import prepare_geopotential
from prepare_geopotential import select_geopotential_500


class PrepareGeopotentialTests(unittest.TestCase):
    def test_selects_500_hpa_from_pressure_level_dataset(self):
        source = np.empty((2, 2, 32, 64), dtype=np.float32)
        source[:, 0] = 850.0
        source[:, 1] = 500.0
        dataset = xr.Dataset(
            {"z": (("time", "level", "latitude", "longitude"), source)},
            coords={
                "time": np.array(["2016-01-01", "2016-01-01T01"], dtype="datetime64[h]"),
                "level": [850, 500],
                "latitude": np.linspace(87.1875, -87.1875, 32),
                "longitude": np.linspace(0, 354.375, 64),
            },
        )

        field, time_dim, lat_name, lon_name = select_geopotential_500(dataset)

        self.assertEqual((time_dim, lat_name, lon_name), ("time", "latitude", "longitude"))
        self.assertEqual(field.shape, (2, 32, 64))
        np.testing.assert_array_equal(field.values, source[:, 1])

    def test_accepts_single_level_geopotential_500_dataset(self):
        values = np.full((2, 32, 64), 500.0, dtype=np.float32)
        dataset = xr.Dataset(
            {"geopotential_500": (("time", "lat", "lon"), values)},
            coords={
                "time": np.array(["2016-01-01", "2016-01-01T01"], dtype="datetime64[h]"),
                "lat": np.linspace(87.1875, -87.1875, 32),
                "lon": np.linspace(0, 354.375, 64),
            },
        )

        field, time_dim, lat_name, lon_name = select_geopotential_500(dataset)

        self.assertEqual((time_dim, lat_name, lon_name), ("time", "lat", "lon"))
        self.assertEqual(field.shape, (2, 32, 64))
        np.testing.assert_array_equal(field.values, values)

    def test_rejects_pressure_level_dataset_without_500_hpa(self):
        dataset = xr.Dataset(
            {"z": (("time", "level", "lat", "lon"), np.zeros((1, 1, 32, 64)))},
            coords={
                "time": np.array(["2016-01-01"], dtype="datetime64[h]"),
                "level": [850],
                "lat": np.linspace(87.1875, -87.1875, 32),
                "lon": np.linspace(0, 354.375, 64),
            },
        )

        with self.assertRaisesRegex(ValueError, "500 hPa"):
            select_geopotential_500(dataset)

    def test_writes_upstream_compatible_shards_and_climatology(self):
        times = np.array(["2013-01-01", "2013-01-01T01"], dtype="datetime64[h]")
        values = np.stack(
            [np.ones((2, 2), dtype=np.float32), np.full((2, 2), 3.0, dtype=np.float32)]
        )
        field = xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": [1.0, -1.0], "lon": [0.0, 180.0]},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(prepare_geopotential, "HOURS_PER_YEAR", 2), \
                    mock.patch.object(prepare_geopotential, "GRID_SHAPE", (2, 2)):
                mean, std, hours = prepare_geopotential.write_split(
                    field,
                    "time",
                    "lat",
                    "lon",
                    range(2013, 2014),
                    "train",
                    Path(temp_dir),
                    1,
                )
            self.assertEqual((mean, std, hours), (2.0, 1.0, 2))
            with np.load(Path(temp_dir) / "train" / "2013_0.npz") as shard:
                self.assertEqual(shard[prepare_geopotential.VARIABLE_NAME].shape, (2, 1, 2, 2))
            with np.load(Path(temp_dir) / "train" / "climatology.npz") as climate:
                self.assertEqual(climate[prepare_geopotential.VARIABLE_NAME].shape, (1, 2, 2))

    def test_prepare_writes_training_statistics_and_all_splits(self):
        times = np.array(
            [
                "2013-01-01",
                "2013-01-01T01",
                "2014-01-01",
                "2014-01-01T01",
                "2015-01-01",
                "2015-01-01T01",
            ],
            dtype="datetime64[h]",
        )
        values = np.stack(
            [np.full((2, 2), value, dtype=np.float32) for value in (1, 3, 5, 7, 9, 11)]
        )
        dataset = xr.Dataset(
            {"z": (("time", "level", "lat", "lon"), values[:, None])},
            coords={
                "time": times,
                "level": [500],
                "lat": [1.0, -1.0],
                "lon": [0.0, 180.0],
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "prepared"
            args = argparse.Namespace(
                input_dir=Path(temp_dir) / "raw",
                output_dir=output_dir,
                train_start_year=2013,
                val_start_year=2014,
                test_start_year=2015,
                end_year=2016,
                num_shards=1,
            )
            with mock.patch.object(prepare_geopotential, "HOURS_PER_YEAR", 2), \
                    mock.patch.object(prepare_geopotential, "GRID_SHAPE", (2, 2)), \
                    mock.patch.object(
                        prepare_geopotential,
                        "find_netcdf_files",
                        return_value=[Path("synthetic.nc")],
                    ), \
                    mock.patch.object(prepare_geopotential.xr, "open_dataset", return_value=dataset):
                summary = prepare_geopotential.prepare(args)

            self.assertEqual(summary["train_hours"], 2)
            with np.load(output_dir / "normalize_mean.npz") as archive:
                self.assertEqual(archive[prepare_geopotential.VARIABLE_NAME].shape, (1,))
                self.assertEqual(float(archive[prepare_geopotential.VARIABLE_NAME][0]), 2.0)
            with np.load(output_dir / "normalize_std.npz") as archive:
                self.assertEqual(archive[prepare_geopotential.VARIABLE_NAME].shape, (1,))
                self.assertEqual(float(archive[prepare_geopotential.VARIABLE_NAME][0]), 1.0)
            with np.load(output_dir / "test" / "climatology.npz") as archive:
                self.assertEqual(archive[prepare_geopotential.VARIABLE_NAME].shape, (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
