#!/usr/bin/env python3
"""Check nonlinear TOV magnetic initialization and CT across AMR interfaces."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import resource
import subprocess
import time

import h5py
import numpy as np


def inspect(path):
    with h5py.File(path) as f:
        fields = {name: f[name][:] for name in (
            "nr.z4c", "nr.adm", "nr.tmunu", "mhd.prim", "mhd.cons",
            "mhd.b_face", "mhd.b_cell", "mhd.divb")}
        assert all(np.isfinite(a).all() for a in fields.values()), str(path)
        assert fields["mhd.prim"][:, 0].min() > 0
        assert fields["mhd.prim"][:, 4].min() > 0
        faces = [f["Locations/" + a][:] for a in "xyz"]
        spacing = [np.diff(x, axis=1) for x in faces]
        nz, ny, nx = fields["mhd.divb"].shape[1:]
        b = fields["mhd.b_face"]
        # Independently recompute the Cartesian face-flux divergence rather
        # than accepting an initializer's zeroed diagnostic array.
        div = (np.diff(b[:, 0, :nz, :ny, :nx+1], axis=3) / spacing[0][:, None, None, :] +
               np.diff(b[:, 1, :nz, :ny+1, :nx], axis=2) / spacing[1][:, None, :, None] +
               np.diff(b[:, 2, :nz+1, :ny, :nx], axis=1) / spacing[2][:, :, None, None])
        np.testing.assert_allclose(div, fields["mhd.divb"], rtol=0, atol=1e-14)
        volume = spacing[2][:, :, None, None] * spacing[1][:, None, :, None] * spacing[0][:, None, None, :]
        levels = f["Levels"][:]
        return dict(time=float(f["Info"].attrs["Time"]), blocks=len(levels),
                    level_counts={str(l): int(np.sum(levels == l)) for l in sorted(set(levels))},
                    independently_recomputed_divb=float(abs(div).max()),
                    magnetic_max=float(abs(fields["mhd.b_cell"]).max()),
                    baryon_mass=float(np.sum(fields["mhd.cons"][:, 0] * volume))), fields


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pangu", required=True, type=Path)
    p.add_argument("--reference-pangu", type=Path, help="Optional old executable: uniform TOV must agree")
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--cases", nargs="+", choices=["amr", "nonbinary", "zero_field", "uniform"],
                   default=["amr", "nonbinary", "zero_field", "uniform"])
    p.add_argument("--timeout", type=float, default=240)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("tov_amr_input", repo / "tst/regression/compact_object_plasma.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    binary, root = a.pangu.resolve(), a.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  runs={}, acceptance=dict(max_abs_divb=1e-12, baryon_drift=1e-5,
                                          restart_scaled_difference=5e-12,
                                          uniform_reference_scaled_difference=5e-13))

    def save():
        temp = root / "summary.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(root / "summary.json")

    def run(command, directory):
        start = time.monotonic()
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=a.timeout)
        assert result.returncode == 0 and "Driver completed." in (directory / "run.log").read_text()
        return time.monotonic() - start

    def difference(left, right):
        return {name: float(abs(left[name]-right[name]).max() / max(1., float(abs(left[name]).max())))
                for name in left}

    save()
    try:
        for case in a.cases:
            refined = case != "uniform"
            duration = .1 if refined else .001
            directory = root / case
            directory.mkdir()
            text = (repo / "input/numerical_relativity/sync_grmhd_tov.in").read_text()
            changes = {("parthenon/mesh", "refinement"): "static" if refined else "none",
                       ("parthenon/mesh", "nghost"): 4,
                       ("parthenon/time", "tlim"): duration,
                       ("parthenon/time", "ncycle_out"): 1,
                       ("parthenon/output1", "dt"): duration / 2,
                       ("problem", "rhoc"): .007993, ("problem", "v_pert"): -.01,
                       ("problem", "b_norm"): 0 if case == "zero_field" else 1}
            for axis, shift in zip((1, 2, 3), (-.3, .2, -.1)):
                offset = shift if case == "nonbinary" else 0
                changes["parthenon/mesh", f"nx{axis}"] = 32
                changes["parthenon/meshblock", f"nx{axis}"] = 8 if refined else 16
                changes["parthenon/mesh", f"x{axis}min"] = -16 + offset
                changes["parthenon/mesh", f"x{axis}max"] = 16 + offset
            if refined:
                for level, extent in ((1, 8), (2, 4), (3, 2)):
                    text += f"\n<parthenon/static_refinement{level-1}>\nlevel = {level}\n" + "".join(
                        f"x{axis}min = {-extent}\nx{axis}max = {extent}\n" for axis in (1, 2, 3))
            if case == "amr":
                text += "\n<parthenon/output3>\nfile_type = rst\ndt = 0.05\nhdf5_compression_level = 0\n"
            deck = directory / "input.in"
            deck.write_text(helper.edit(text, changes))
            entry = report["runs"][case] = dict(status="running")
            save()
            entry["wall_seconds"] = run([str(binary), "-i", str(deck)], directory)
            samples = []
            for path in sorted(directory.glob("*.phdf")):
                value, _ = inspect(path)
                samples.append(value)
            samples.sort(key=lambda s: s["time"])
            entry["samples"] = samples
            save()
            assert samples[0]["time"] == 0 and abs(samples[-1]["time"]-duration) < 1e-12
            assert max(s["independently_recomputed_divb"] for s in samples) < 1e-12
            entry["relative_baryon_drift"] = abs(samples[-1]["baryon_mass"]/samples[0]["baryon_mass"]-1)
            assert entry["relative_baryon_drift"] < 1e-5
            if refined:
                assert len(samples[-1]["level_counts"]) == 4
            if case == "zero_field":
                assert max(s["magnetic_max"] for s in samples) == 0
            final = next(directory.glob("*.final.phdf"))
            _, actual = inspect(final)
            if case == "amr":
                checkpoints = []
                for path in directory.glob("*.rhdf"):
                    with h5py.File(path) as f:
                        if 0 < float(f["Info"].attrs["Time"]) < duration:
                            checkpoints.append(path)
                assert checkpoints
                resumed = root / "amr_restart"
                resumed.mkdir()
                run([str(binary), "-i", str(deck), "-r", str(sorted(checkpoints)[-1])], resumed)
                last, fields = inspect(next(resumed.glob("*.final.phdf")))
                entry["restart_scaled_differences"] = difference(actual, fields)
                assert max(entry["restart_scaled_differences"].values()) < 5e-12
                assert last["independently_recomputed_divb"] < 1e-12
            if case == "uniform" and a.reference_pangu:
                reference = root / "uniform_reference"
                reference.mkdir()
                run([str(a.reference_pangu.resolve()), "-i", str(deck)], reference)
                _, fields = inspect(next(reference.glob("*.final.phdf")))
                entry["reference_scaled_differences"] = difference(actual, fields)
                assert max(entry["reference_scaled_differences"].values()) < 5e-13
            entry["status"] = "passed"
            save()
            print(case, json.dumps(entry), flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", failure=repr(error))
        save()
        raise
    save()


if __name__ == "__main__":
    main()
