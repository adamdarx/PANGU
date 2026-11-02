#!/usr/bin/env python3
"""Short CUDA qualification of the native Z4c/GRMHD plugin initial-data hook."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import resource
import subprocess
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugins.compact_objects.surface_flux import extract as extract_sphere_flux

_diagnostic_path = Path(__file__).resolve().parents[2] / "plugins/compact_objects/flux_diagnostics.py"
_spec = importlib.util.spec_from_file_location("plasma_flux_diagnostics", _diagnostic_path)
_diagnostics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diagnostics)


def edit(text, changes):
    for (section, name), value in changes.items():
        start = text.index("<" + section + ">") + len(section) + 2
        end = text.find("\n<", start)
        if end < 0:
            end = len(text)
        body = text[start:end]
        pattern = r"(?m)^[ \t]*" + re.escape(name) + r"[ \t]*=.*$"
        body = re.sub(pattern, f"{name} = {value}", body) if re.search(pattern, body) else body + f"\n{name} = {value}\n"
        text = text[:start] + body + text[end:]
    return text


def inspect(path):
    with h5py.File(path) as f:
        names = ["nr.z4c", "nr.adm", "nr.tmunu", "nr.constraints", "mhd.prim", "mhd.cons",
                 "mhd.b_cell", "mhd.divb", "mhd.recovery", "mhd.fofc"]
        fields = {name: f[name][:] for name in names}
        for name, values in fields.items():
            assert np.isfinite(values).all(), (path, name)
        assert np.max(abs(fields["nr.tmunu"])) == 0, "test plasma must not gravitate"
        rho, energy = fields["mhd.prim"][:, 0], fields["mhd.prim"][:, 4]
        assert np.min(rho) > 0 and np.min(energy) > 0
        h = {axis: np.diff(f["Locations/" + axis][:], axis=-1) for axis in "xyz"}
        volume = h["z"][:, :, None, None] * h["y"][:, None, :, None] * h["x"][:, None, None, :]
        max_w, max_sigma, max_em_flux = 1., 0., 0.
        # Process blocks separately to avoid duplicating an entire 3D AMR
        # snapshot as 4-tensors in a laptop's host memory.
        for block in range(len(rho)):
            metric = np.empty(rho[block].shape + (3, 3))
            for component, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
                metric[..., i, j] = metric[..., j, i] = fields["nr.adm"][block, component]
            diagnostic = _diagnostics.flux_densities(
                rho[block], np.moveaxis(fields["mhd.prim"][block, 1:4], 0, -1),
                np.moveaxis(fields["mhd.b_cell"][block], 0, -1), metric,
                fields["nr.z4c"][block, 18], np.moveaxis(fields["nr.z4c"][block, 19:22], 0, -1))
            max_w = max(max_w, float(np.max(diagnostic["lorentz_factor"])))
            max_sigma = max(max_sigma, float(np.max(diagnostic["magnetization"])))
            max_em_flux = max(max_em_flux, float(np.max(abs(diagnostic["em_energy_flux_density"]))))
        return dict(time=float(f["Info"].attrs["Time"]), blocks=int(f["Info"].attrs["NumMeshBlocks"]),
                    min_dx=float(min(np.min(s) for s in h.values())),
                    rho_min=float(rho.min()), rho_max=float(rho.max()),
                    divb_max=float(np.max(abs(fields["mhd.divb"]))),
                    magnetic_max=float(np.max(abs(fields["mhd.b_cell"]))),
                    baryon_mass=float(np.sum(fields["mhd.cons"][:, 0] * volume)),
                    recovery_mass_correction=float(np.sum(abs(fields["mhd.recovery"][:, 0]) * volume)),
                    protection_cells=int(np.count_nonzero(fields["mhd.recovery"][:, 2])),
                    fofc_cells=int(np.count_nonzero(fields["mhd.fofc"])),
                    lorentz_factor_max=max_w, magnetization_max=max_sigma,
                    coordinate_em_energy_flux_density_max=max_em_flux)


def check_single_inflow(path):
    """Reject missing fluid gravity, even when fields remain finite and div B=0."""
    result = extract_sphere_flux(path, [.6, .8], ntheta=32, nphi=64)
    assert result["time"] > 0
    assert all(s["mass_inflow_rate"] > 0 for s in result["spheres"]), \
        "initially resting single-BH plasma must fall inward outside the horizon"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pangu", required=True, type=Path)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--time", type=float, default=.02)
    p.add_argument("--launcher-json", type=json.loads, default=[],
                   help='MPI launch argv as JSON, e.g. ["mpirun", "-np", "1"]')
    p.add_argument("--cases", nargs="+", choices=["single_b0", "single_b1", "binary_b1"],
                   default=["single_b0", "single_b1", "binary_b1"])
    args = p.parse_args()
    if not isinstance(args.launcher_json, list) or not all(isinstance(x, str) for x in args.launcher_json):
        p.error("--launcher-json must be a JSON array of strings")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    binary, root = args.pangu.resolve(), args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    report = dict(status="running", executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), runs={})

    def save():
        temporary = root / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(root / "summary.json")

    try:
        for case in args.cases:
            d = root / case
            d.mkdir()
            kind = "binary" if case.startswith("binary") else "single"
            template = (repo / f"plugins/compact_objects/examples/{kind}_plasma_admission.in").read_text()
            deck = d / "input.in"
            changes = {("parthenon/time", "tlim"): args.time, ("parthenon/time", "ncycle_out"): 1,
                       ("parthenon/output1", "dt"): args.time / 2,
                       ("parthenon/output2", "dt"): args.time / 2}
            if case == "single_b0":
                changes[("plugin/compact_objects", "btilde3")] = 0
            deck.write_text(edit(template, changes))
            command = args.launcher_json + [str(binary), "-i", str(deck)]
            with (d / "mesh.log").open("w") as out:
                subprocess.run(command + ["-m", "1"], cwd=d, stdout=out, stderr=subprocess.STDOUT, check=True)
            start = time.monotonic()
            with (d / "run.log").open("w") as out:
                rc = subprocess.run(command, cwd=d, stdout=out, stderr=subprocess.STDOUT).returncode
            report["runs"][case] = dict(command=command, exit_code=rc, elapsed_seconds=time.monotonic() - start)
            save()
            assert rc == 0 and "Driver completed." in (d / "run.log").read_text(), case
            snapshots = sorted(x for x in d.glob("*.phdf") if ".final." not in x.name)
            assert snapshots, "initial output missing"
            samples = [inspect(x) for x in snapshots + [next(d.glob("*.final.phdf"))]]
            first, final = samples[0], samples[-1]
            assert abs(final["time"] - args.time) < 1e-10
            assert final["min_dx"] < .05
            expected_bz = 0 if case == "single_b0" else .1
            with h5py.File(snapshots[0]) as f:
                assert np.max(abs(f["mhd.b_cell"][:, :2])) == 0
                assert np.max(abs(f["mhd.b_cell"][:, 2] - expected_bz)) < 1e-14
            divb = max(x["divb_max"] for x in samples)
            assert divb < 1e-10, divb
            if case == "single_b0":
                assert final["magnetic_max"] == 0
            if kind == "single":
                report["runs"][case]["exterior_flux"] = check_single_inflow(
                    next(d.glob("*.final.phdf")))
            report["runs"][case].update(status="smoke_passed", initial=first, final=final, max_divb=divb, samples=samples)
            save()
            print(json.dumps({case: report["runs"][case]}), flush=True)
        if {"single_b0", "single_b1"}.issubset(args.cases):
            with h5py.File(next((root / "single_b0").glob("*.final.phdf"))) as a, h5py.File(next((root / "single_b1").glob("*.final.phdf"))) as b:
                difference = float(np.max(abs(a["nr.z4c"][:] - b["nr.z4c"][:])))
            report["zero_backreaction_metric_difference"] = difference
            assert difference < 1e-12, difference
        report["status"] = "smoke_passed"
    except BaseException as exc:
        report["status"], report["failure"] = "failed", repr(exc)
        save()
        raise
    save()


if __name__ == "__main__":
    main()
