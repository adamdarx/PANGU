"""Unit tests for the NumPy PANGU GRMHD PHDF5 reader."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from pangu_grmhd import GRMHDFrame, GRMHDSeries  # noqa: E402


def write_frame(path: Path, metric: str) -> None:
    ni, nj, nk = 3, 2, 1
    with h5py.File(path, "w") as handle:
        info = handle.create_group("Info")
        info.attrs["Time"] = 12.5
        info.attrs["NCycle"] = 7
        info.attrs["OutputDatasetNames"] = np.asarray(
            ["mhd.b_cell", "mhd.prim"], dtype=h5py.string_dtype()
        )
        info.attrs["NumComponents"] = np.asarray([3, 5], dtype=np.uint64)
        info.attrs["ComponentNames"] = np.asarray(
            [
                "mhd.b_cell_B1",
                "mhd.b_cell_B2",
                "mhd.b_cell_B3",
                "mhd.prim_density",
                "mhd.prim_velocity_1",
                "mhd.prim_velocity_2",
                "mhd.prim_velocity_3",
                "mhd.prim_internal_energy_density",
            ],
            dtype=h5py.string_dtype(),
        )
        params = handle.create_group("Params")
        params.attrs["geometry/metric"] = metric
        params.attrs["geometry/mode"] = "static" if metric == "mks" else "dynamic"
        params.attrs["geometry/background"] = 1
        params.attrs["geometry/bh_spin"] = 0.5
        params.attrs["mhd/gamma"] = 4.0 / 3.0
        input_group = handle.create_group("Input")
        input_group.attrs["File"] = "<geometry>\nhslope = 0.3\n"

        locations = handle.create_group("VolumeLocations")
        if metric == "mks":
            locations.create_dataset("x", data=np.asarray([[np.log(2.0), np.log(3.0), np.log(4.0)]]))
            locations.create_dataset("y", data=np.asarray([[0.25, 0.75]]))
            locations.create_dataset("z", data=np.asarray([[np.pi]]))
        else:
            locations.create_dataset("x", data=np.asarray([[2.0, 3.0, 4.0]]))
            locations.create_dataset("y", data=np.asarray([[-0.5, 0.5]]))
            locations.create_dataset("z", data=np.asarray([[0.25]]))

        handle.create_dataset("Levels", data=np.asarray([0]))
        handle.create_dataset("LogicalLocations", data=np.asarray([[0, 0, 0]]))
        primitive = np.zeros((1, 5, nk, nj, ni), dtype=np.float64)
        primitive[:, 0] = 1.25
        primitive[:, 1] = 0.08
        primitive[:, 2] = -0.03
        primitive[:, 3] = 0.02
        primitive[:, 4] = 0.4
        magnetic = np.zeros((1, 3, nk, nj, ni), dtype=np.float64)
        magnetic[:, 0] = 0.2
        magnetic[:, 1] = -0.1
        magnetic[:, 2] = 0.05
        handle.create_dataset("mhd.prim", data=primitive)
        handle.create_dataset("mhd.b_cell", data=magnetic)


class ReaderTest(unittest.TestCase):
    def check_metric(self, metric: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "test.prim.00000.phdf"
            second = Path(directory) / "test.prim.00001.phdf"
            write_frame(first, metric)
            write_frame(second, metric)
            with GRMHDFrame(first) as frame:
                self.assertEqual(frame.metric_name, metric)
                self.assertEqual(frame.rest_mass_density().shape, (1, 1, 2, 3))
                self.assertEqual(frame.four_velocity().shape, (1, 1, 2, 3, 4))
                self.assertEqual(frame.four_magnetic_field().shape, (1, 1, 2, 3, 4))
                errors = frame.constraint_errors()
                self.assertLess(errors["metric_inverse"], 2.0e-13)
                self.assertLess(errors["four_velocity_normalization"], 2.0e-13)
                self.assertLess(errors["magnetic_orthogonality"], 2.0e-13)
                np.testing.assert_allclose(frame.energy_density(), 0.4)
                np.testing.assert_allclose(frame.pressure(), 0.4 / 3.0)
            series = GRMHDSeries(directory)
            self.assertEqual(len(series), 2)
            np.testing.assert_allclose(series.times, [12.5, 12.5])

    def test_cks(self) -> None:
        self.check_metric("cks")

    def test_mks(self) -> None:
        self.check_metric("mks")


if __name__ == "__main__":
    unittest.main()

