#!/usr/bin/env python3
"""Measure three-resolution EMS convergence on one common exterior grid.

This consumes completed runs and their analysis_ems.json files. Passing these
checks does not replace the independent outer-boundary and restart controls.
"""
from pathlib import Path
import argparse
import importlib.util
import json

import h5py
import numpy as np
from scipy.interpolate import make_interp_spline

_spec = importlib.util.spec_from_file_location("ems_static_analysis", Path(__file__).with_name("analyze_ems.py"))
_analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analysis)


def load(directory):
    directory = Path(directory).resolve()
    summary = json.loads((directory / "analysis_ems.json").read_text())
    runtime = json.loads((directory / "result.json").read_text())
    if "Driver completed." not in (directory / "run.log").read_text():
        raise ValueError(f"incomplete evolution: {directory}")
    config = _analysis.parameters(directory / "input.in")
    integration = {"integrator": config["parthenon/time", "integrator"]}
    for name, default in [("cfl", .3), ("dissipation", .05), ("constraint_iterations", 12),
                          ("constraint_tolerance", 1e-11)]:
        integration[name] = float(config.get(("plugin/scalar_field", name), default))
    paths = list(directory.glob("*.final.phdf"))
    if len(paths) != 1:
        raise ValueError(f"expected one final snapshot: {directory}")
    with h5py.File(paths[0]) as f:
        if int(f["Info"].attrs["NumMeshBlocks"]) != 1:
            raise ValueError("one radial block is required")
        time = float(f["Info"].attrs["Time"])
        radius = f["VolumeLocations/x"][:].reshape(-1)
        values = np.stack((f["scalar.state"][:].reshape(3, -1)[0],
                           f["ems.geometry"][:].reshape(4, -1)[2]))
        faces = f["Locations/x"][:].reshape(-1)
    spacing = float(np.mean(np.diff(radius)))
    if (not np.isfinite(values).all() or spacing <= 0 or
            not np.allclose(np.diff(radius), spacing, rtol=1e-10, atol=1e-12)):
        raise ValueError("nonfinite data or a nonuniform radial mesh")
    if abs(time - summary["time"]) > 1e-10 or abs(time - runtime["time"]) > 1e-10:
        raise ValueError("snapshot/analysis/runtime times disagree")
    return dict(path=str(directory), summary=summary, integration=integration,
                time=time, radius=radius, fields=values, h=spacing,
                bounds=[float(faces[0]), float(faces[-1])],
                n=len(radius), recorded_wall_seconds=runtime.get("wall_seconds"))


def order(coarse, fine):
    return float(np.log2(coarse / fine)) if coarse > 0 and fine > 0 else None


def compare(directories, inner_factor=1.05, outer_radius=30.):
    runs = sorted([load(d) for d in directories], key=lambda r: r["n"])
    first = runs[0]
    for run in runs[1:]:
        if (run["summary"]["action"] != first["summary"]["action"] or
                run["integration"] != first["integration"] or run["bounds"] != first["bounds"] or
                abs(run["time"] - first["time"]) > 1e-10):
            raise ValueError("action, integration settings, domain or final time differs across resolutions")
    if not all(b["n"] == 2 * a["n"] for a, b in zip(runs, runs[1:])):
        raise ValueError("three factor-two resolutions are required")
    if inner_factor <= 1:
        raise ValueError("comparison must be strictly outside every measured horizon")
    inner = inner_factor * max(r["summary"]["horizon_radius"] for r in runs)
    selected = (first["radius"] > inner) & (first["radius"] < outer_radius)
    x = first["radius"][selected]
    if len(x) < 8:
        raise ValueError("insufficient common exterior points")
    # A degree-five interpolant has O(h^6) error for smooth exterior fields,
    # above the evolution's fourth-order target. Never extrapolate the data.
    samples = [first["fields"][:, selected]]
    for run in runs[1:]:
        if x[0] < run["radius"][0] or x[-1] > run["radius"][-1]:
            raise ValueError("comparison would extrapolate")
        samples.append(make_interp_spline(run["radius"], run["fields"].T, k=5)(x).T)
    differences = [np.sqrt(np.mean((a - b)**2, axis=1)) for a, b in zip(samples, samples[1:])]
    constraints = [r["summary"]["final_mass_constraint"] for r in runs]
    static_errors = [r["summary"]["static_phi_max_difference"] for r in runs]
    checks = dict(exterior_differences_decrease=bool(np.all(differences[1] < differences[0])),
                  mass_constraint_decreases=all(b < a for a, b in zip(constraints, constraints[1:])),
                  charge_residual=all(r["summary"]["max_charge_error"] < 1e-12 for r in runs),
                  finest_static_phi=static_errors[-1] < 1e-4)
    return dict(status="passed_convergence_checks" if all(checks.values()) else "needs_review",
                checks=checks, time=first["time"], n=[r["n"] for r in runs],
                h=[r["h"] for r in runs], action=first["summary"]["action"],
                integration=first["integration"], common_interval=[float(x[0]), float(x[-1])],
                common_points=len(x), inner_horizon_factor=inner_factor,
                interpolation="degree-five spline, O(h^6) for smooth data; no extrapolation",
                norm="RMS on identical coarse-grid exterior points; no spherical r^2 weighting",
                phi_adjacent_rms=[float(d[0]) for d in differences],
                metric_N_adjacent_rms=[float(d[1]) for d in differences],
                phi_self_convergence_order=order(differences[0][0], differences[1][0]),
                metric_N_self_convergence_order=order(differences[0][1], differences[1][1]),
                mass_constraints=constraints,
                mass_constraint_orders=[order(a, b) for a, b in zip(constraints, constraints[1:])],
                static_phi_max=static_errors, paths=[r["path"] for r in runs],
                boundary_and_restart_controls="separate requirements; not asserted by this tool",
                recorded_wall_seconds=[r["recorded_wall_seconds"] for r in runs],
                timing_note="Recorded run durations require separate hardware/concurrency provenance before performance comparisons")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs=3, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inner-factor", type=float, default=1.05)
    parser.add_argument("--outer-radius", type=float, default=30.)
    args = parser.parse_args()
    result = compare(args.directories, args.inner_factor, args.outer_radius)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed_convergence_checks" else 1)
