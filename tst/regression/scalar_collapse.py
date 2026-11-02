#!/usr/bin/env python3
"""Qualify the regular-center Einstein-scalar plugin on a supplied CUDA binary.

Each run keeps its input, full log, elapsed time and executable hash. Weak-field
errors use the independent analytic spherical wave solution, including passage
through r=0. Strong-field tests require horizon formation and causal excision.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import resource
import subprocess
import time

import h5py
import numpy as np


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pangu", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=["zero", "weak", "weak_massive", "dispersion", "collapse", "restart", "controls"],
                        default=["zero", "weak", "dispersion", "collapse", "restart"])
    parser.add_argument("--strong-n", type=int, default=512)
    parser.add_argument("--strong-time", type=float, default=40)
    parser.add_argument("--reference", type=Path, help="Completed N512 R40 T20 strong-field qualification for controls")
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    binary, root = args.pangu.resolve(), args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    template = (repo / "plugins/scalar_collapse/examples/dispersion_admission.in").read_text()
    results = {"status": "running", "executable": str(binary),
               "executable_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "runs": {}}

    def save():
        temporary = root / "summary.tmp"
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(root / "summary.json")

    def run(name, n, amplitude, final_time, changes=None, restart=None):
        directory = root / name
        directory.mkdir()
        deck = directory / "input.in"
        base = {("parthenon/mesh", "nx1"): n, ("parthenon/meshblock", "nx1"): n,
                ("parthenon/time", "tlim"): final_time,
                ("plugin/scalar_collapse", "amplitude"): amplitude,
                ("parthenon/output1", "dt"): final_time / 2,
                ("parthenon/output2", "dt"): final_time / 2}
        deck.write_text(edit(template, base | (changes or {})))
        command = [str(binary), "-i", str(deck)]
        if restart:
            command += ["-r", str(restart)]
        start = time.monotonic()
        with (directory / "run.log").open("w") as stream:
            completed = subprocess.run(command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT)
        results["runs"][name] = {"command": command, "n": n, "amplitude": amplitude,
                                  "elapsed_seconds": time.monotonic() - start, "exit_code": completed.returncode}
        save()
        log = (directory / "run.log").read_text()
        assert completed.returncode == 0 and "Driver completed." in log, name
        final = next(directory.glob("*.final.phdf"))
        with h5py.File(final) as stream:
            assert "Default Device: Cuda" in str(stream["Info"].attrs["KokkosConfig"]), "qualification requires CUDA"
            actual = float(stream["Info"].attrs["Time"])
            assert abs(actual - final_time) < 1e-10, (name, actual)
            u = stream["scalar.state"][:].reshape(3, n)
            g = stream["collapse.geometry"][:].reshape(4, n)
            active = stream["collapse.active"][:].reshape(n) > .5
            radius = stream["VolumeLocations/x"][:].reshape(n)
            assert np.isfinite(u).all() and np.isfinite(g).all(), name
            assert np.all(g[0, active] > 0), name
            first = int(np.flatnonzero(active)[0])
            assert first == 0 or g[1, first] > 1, "excision boundary is not causally trapped"
        history = np.genfromtxt(directory / "collapse_history.csv", names=True, delimiter=",")
        assert np.isfinite(history.view(float)).all(), name
        assert np.all(np.diff(history["time"]) > 0), name
        rec = results["runs"][name]
        rec.update(final_time=actual, max_compactness=float(np.max(history["max_compactness"])),
                   horizon_formed=bool(np.any(history["horizon_found"] > .5)),
                   final_horizon_radius=float(history["horizon_radius"][-1]),
                   final_excision_cells=first, initial_mass=float(history["outer_mass"][0]),
                   final_mass=float(history["outer_mass"][-1]),
                   mass_flux_balance=float(history["outer_mass"][-1] - history["outer_mass"][0] +
                                           np.trapz(history["outgoing_luminosity"], history["time"])),
                   max_mass_constraint=float(np.max(history["mass_constraint_max"])),
                   max_lapse_constraint=float(np.max(history["lapse_constraint_max"])))
        save()
        print(json.dumps({name: rec}), flush=True)
        return directory, radius, u, g, active, history

    try:
        save()
        if "zero" in args.cases:
            _, r, u, g, _, h = run("zero", 64, 0, 1, {("plugin/scalar_collapse", "constraint_iterations"): 8})
            assert np.max(abs(u)) == 0 and np.max(abs(g[0] - 1)) == 0
            assert not np.any(h["horizon_found"])
        if "weak" in args.cases:
            errors = []
            for n in (256, 512, 1024):
                _, r, u, _, _, h = run(f"weak_n{n}", n, 1e-10, 20,
                                      {("plugin/scalar_collapse", "constraint_iterations"): 8,
                                       ("plugin/scalar_collapse", "dissipation"): 0})
                def odd_profile(s):
                    return s * 1e-10 * (np.exp(-((s - 8) / 1.5)**2) + np.exp(-((s + 8) / 1.5)**2))
                exact = (odd_profile(r - 20) + odd_profile(r + 20)) / (2 * r)
                error = float(np.linalg.norm((u[0] - exact) * r) / np.linalg.norm(exact * r))
                assert not np.any(h["horizon_found"])
                errors.append(error)
                results["runs"][f"weak_n{n}"]["relative_radial_L2_phi_error"] = error
                save()
            orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
            results["weak_convergence_orders"] = orders.tolist()
            save()
            assert np.all(orders > 3.5), (errors, orders)
        if "weak_massive" in args.cases:
            # Independent Fourier solution for U=r*phi: U_tt=U_rr-mu^2*U.
            # The odd periodic extension is many causal light-crossing times
            # away; all evaluation points coincide with this fine FFT lattice.
            count, length, mu, final_time = 65536, 160., .2, 20.
            x = np.arange(count) * length / count - length / 2
            initial = x * 1e-10 * (np.exp(-((x - 8) / 1.5)**2) + np.exp(-((x + 8) / 1.5)**2))
            frequency = np.sqrt((2 * np.pi * np.fft.rfftfreq(count, length / count))**2 + mu**2)
            reference = np.fft.irfft(np.fft.rfft(initial) * np.cos(frequency * final_time), n=count)
            errors = []
            for n in (256, 512, 1024):
                _, radius, u, _, _, h = run(f"massive_n{n}", n, 1e-10, final_time,
                    {("plugin/scalar_collapse", "mass"): mu,
                     ("plugin/scalar_collapse", "constraint_iterations"): 8,
                     ("plugin/scalar_collapse", "dissipation"): 0})
                index = np.rint((radius + length / 2) * count / length).astype(int)
                assert np.max(abs(x[index] - radius)) < 1e-12
                exact = reference[index]
                error = float(np.linalg.norm(u[0] * radius - exact) / np.linalg.norm(exact))
                assert not np.any(h["horizon_found"])
                errors.append(error)
                results["runs"][f"massive_n{n}"]["relative_radial_L2_phi_error"] = error
                save()
            orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
            results["massive_weak_convergence_orders"] = orders.tolist()
            save()
            assert np.all(orders > 3.5), (errors, orders)
        if "dispersion" in args.cases:
            _, _, _, _, _, h = run("dispersion", 512, .02, 40)
            assert not np.any(h["horizon_found"])
            assert h["max_compactness"][-1] < np.max(h["max_compactness"]) / 5
        if "collapse" in args.cases or "restart" in args.cases:
            strong = run("collapse", args.strong_n, .3, args.strong_time)
            directory, _, u, _, active, h = strong
            assert h["max_compactness"][0] < 1, "initial data already trapped"
            assert np.any(h["horizon_found"] > .5) and h["horizon_found"][-1] > .5
            assert np.any(~active) and h["outer_mass"][-1] > 0
            assert h["horizon_radius"][-1] / 2 <= h["outer_mass"][-1] * 1.001
            if "restart" in args.cases:
                checkpoint = sorted(p for p in directory.glob("*.rhdf") if ".final." not in p.name and ".00000." not in p.name)[-1]
                _, _, resumed, _, resumed_active, _ = run("resumed", args.strong_n, .3, args.strong_time, restart=checkpoint)
                difference = float(np.max(abs(u[:, active] - resumed[:, active])))
                assert np.array_equal(active, resumed_active) and difference < 1e-10, difference
                results["restart_max_difference"] = difference
        if "controls" in args.cases:
            assert args.reference, "controls require an already completed --reference"
            previous = json.loads((args.reference / "summary.json").read_text())
            assert previous["status"] == "passed"
            assert previous["executable_sha256"] == results["executable_sha256"], "reference binary differs"
            reference = previous["runs"]["collapse"]
            assert reference["n"] == 512 and reference["final_time"] == 20 and reference["amplitude"] == .3
            changes = [
                ("cfl_half", 512, {("plugin/scalar_collapse", "cfl"): .1}),
                ("boundary_double", 1024, {("parthenon/mesh", "x1max"): 80})]
            for name, n, config in changes:
                run(name, n, .3, 20, config)
                rec = results["runs"][name]
                shift = abs(rec["final_horizon_radius"] / reference["final_horizon_radius"] - 1)
                rec["relative_horizon_change_vs_reference"] = shift
                save()
                # Fixed project tolerance, below one tenth of the measured
                # N512->N1024 relative spatial-resolution difference.
                assert shift < 2e-6, (name, shift)
        results["status"] = "passed"
    except BaseException as exc:
        results["status"] = "failed"
        results["failure"] = repr(exc)
        save()
        raise
    save()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
