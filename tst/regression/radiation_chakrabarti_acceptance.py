#!/usr/bin/env python3
"""Audit a cooled, tilted Chakrabarti-torus trajectory.

This is a statistical acceptance test, not a pointwise trajectory comparison.
It verifies the radiation history ledger and evaluates the disk angular-
momentum direction and thickness directly on the native CKS/SMR leaf cells.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np


SPIN = 15.0 / 16.0
GAMMA = 13.0 / 9.0
DISK_DENSITY = 1.0e-4
REGIONS = (("inner", 3.0, 10.0), ("middle", 10.0, 25.0), ("outer", 25.0, 50.0))
RADIAL_EDGES = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
HISTORY_COLUMNS = (
    "time_M",
    "dt_M",
    "cycle",
    "mesh_blocks",
    "mass",
    "momentum_1",
    "momentum_2",
    "momentum_3",
    "energy",
    "fofc_cells",
    "cooling_power",
    "removed_energy",
    "boundary_energy",
    "max_abs_divb",
    "max_cooling_fraction",
)
NUMBERED_OUTPUT = re.compile(r"^gr_chakrabarti_torus\.prim\.(\d+)\.phdf$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--target-time", required=True, type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all-snapshots", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--divb-limit", type=float, default=1.0e-10)
    parser.add_argument("--minimum-disk-mass-fraction", type=float, default=0.2)
    parser.add_argument("--require-bp", action="store_true")
    parser.add_argument("--minimum-bp-contrast-deg", type=float, default=10.0)
    parser.add_argument("--minimum-bp-inner-mass", type=float, default=1.0)
    return parser.parse_args()


def history(path: Path) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            rows.append([float(value) for value in stripped.split()])
    if not rows:
        raise RuntimeError(f"{path}: no history records")
    values = np.asarray(rows, dtype=np.float64)
    errors: list[str] = []
    if values.shape[1] != len(HISTORY_COLUMNS):
        errors.append(
            f"history has {values.shape[1]} columns, expected {len(HISTORY_COLUMNS)}"
        )
        return values, errors
    if not np.isfinite(values).all():
        errors.append("history contains non-finite values")
    if np.any(np.diff(values[:, 0]) <= 0.0) or np.any(np.diff(values[:, 2]) <= 0.0):
        errors.append("history time or cycle is not strictly increasing")
    if np.any(values[:, 1] <= 0.0):
        errors.append("history contains a non-positive timestep")
    for column, name in (
        (10, "cooling power"),
        (11, "removed energy"),
        (13, "absolute divB"),
        (14, "cooling fraction"),
    ):
        if np.any(values[:, column] < 0.0):
            errors.append(f"history contains negative {name}")
    # Boundary energy is the signed integral of the outward numerical flux;
    # net inflow is therefore negative and cumulative values need not be
    # monotonic.  Only the positive-definite removed-energy ledger is.
    scale = max(float(np.max(np.abs(values[:, 11]))), 1.0)
    if np.any(np.diff(values[:, 11]) < -64.0 * np.finfo(float).eps * scale):
        errors.append("cumulative removed energy is not monotonic")
    return values, errors


def kerr_radius(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    cartesian2 = x * x + y * y + z * z
    spin2 = SPIN * SPIN
    return np.sqrt(
        0.5
        * (cartesian2 - spin2 + np.sqrt((cartesian2 - spin2) ** 2 + 4.0 * spin2 * z * z))
    )


def stress_momentum(
    primitive: np.ndarray,
    magnetic: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = primitive[0]
    ux_tilde, uy_tilde, uz_tilde = primitive[1:4]
    internal = primitive[4]
    bx, by, bz = magnetic
    safe_radius = np.maximum(radius, 1.0e-14)
    radius2 = safe_radius * safe_radius
    denominator = radius2 + SPIN * SPIN
    null_x = (safe_radius * x + SPIN * y) / denominator
    null_y = (safe_radius * y - SPIN * x) / denominator
    null_z = z / safe_radius
    metric_factor = 2.0 * radius2 * safe_radius / (
        radius2 * radius2 + SPIN * SPIN * z * z
    )
    alpha = 1.0 / np.sqrt(1.0 + metric_factor)
    null_dot_tilde = null_x * ux_tilde + null_y * uy_tilde + null_z * uz_tilde
    spatial_u2 = (
        ux_tilde * ux_tilde
        + uy_tilde * uy_tilde
        + uz_tilde * uz_tilde
        + metric_factor * null_dot_tilde * null_dot_tilde
    )
    lorentz = np.sqrt(np.maximum(1.0 + spatial_u2, 1.0))
    u0 = lorentz / alpha
    shift = alpha * lorentz * metric_factor
    ux = ux_tilde - shift * null_x
    uy = uy_tilde - shift * null_y
    uz = uz_tilde - shift * null_z
    null_dot_u = null_x * ux + null_y * uy + null_z * uz
    u_cov_x = ux + metric_factor * null_x * (u0 + null_dot_u)
    u_cov_y = uy + metric_factor * null_y * (u0 + null_dot_u)
    u_cov_z = uz + metric_factor * null_z * (u0 + null_dot_u)
    b0 = u_cov_x * bx + u_cov_y * by + u_cov_z * bz
    bcx = (bx + b0 * ux) / u0
    bcy = (by + b0 * uy) / u0
    bcz = (bz + b0 * uz) / u0
    null_dot_b = null_x * bcx + null_y * bcy + null_z * bcz
    bcov0 = (-1.0 + metric_factor) * b0 + metric_factor * null_dot_b
    bcovx = bcx + metric_factor * null_x * (b0 + null_dot_b)
    bcovy = bcy + metric_factor * null_y * (b0 + null_dot_b)
    bcovz = bcz + metric_factor * null_z * (b0 + null_dot_b)
    magnetic2 = b0 * bcov0 + bcx * bcovx + bcy * bcovy + bcz * bcovz
    pressure = (GAMMA - 1.0) * internal
    enthalpy = rho + GAMMA / (GAMMA - 1.0) * pressure + magnetic2
    return (
        enthalpy * u0 * u_cov_x - b0 * bcovx,
        enthalpy * u0 * u_cov_y - b0 * bcovy,
        enthalpy * u0 * u_cov_z - b0 * bcovz,
        lorentz,
        magnetic2,
    )


def region_record(
    name: str,
    lower: float,
    upper: float,
    blocks: list[dict[str, np.ndarray]],
) -> dict[str, float | str | None]:
    angular_momentum = np.zeros(3, dtype=np.float64)
    mass = 0.0
    positions: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for block in blocks:
        radius = block["radius"]
        mask = (
            (radius >= lower)
            & (radius < upper)
            & (block["density"] > DISK_DENSITY)
        )
        if not np.any(mask):
            continue
        weight = block["mass_weight"][mask]
        sx, sy, sz = block["stress"]
        x, y, z = block["position"]
        angular_momentum += (
            np.sum((y[mask] * sz[mask] - z[mask] * sy[mask]) * block["volume"][mask]),
            np.sum((z[mask] * sx[mask] - x[mask] * sz[mask]) * block["volume"][mask]),
            np.sum((x[mask] * sy[mask] - y[mask] * sx[mask]) * block["volume"][mask]),
        )
        mass += float(np.sum(weight))
        positions.append((x[mask], y[mask], z[mask], weight))
    norm = float(np.linalg.norm(angular_momentum))
    if norm <= np.finfo(float).tiny or mass <= np.finfo(float).tiny:
        return {
            "region": name,
            "r_min_M": lower,
            "r_max_M": upper,
            "disk_mass": mass,
            "tilt_deg": None,
            "twist_deg": None,
            "h_over_r": None,
        }
    normal = angular_momentum / norm
    height2 = 0.0
    cylindrical2 = 0.0
    for x, y, z, weight in positions:
        height = normal[0] * x + normal[1] * y + normal[2] * z
        cartesian2 = x * x + y * y + z * z
        height2 += float(np.sum(weight * height * height))
        cylindrical2 += float(np.sum(weight * np.maximum(cartesian2 - height * height, 0.0)))
    return {
        "region": name,
        "r_min_M": lower,
        "r_max_M": upper,
        "disk_mass": mass,
        "tilt_deg": math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0)))),
        "twist_deg": math.degrees(math.atan2(float(normal[1]), float(normal[0]))),
        "h_over_r": math.sqrt(height2 / max(cylindrical2, np.finfo(float).tiny)),
    }


def snapshot(path: Path) -> dict[str, object]:
    blocks: list[dict[str, np.ndarray]] = []
    density_min = math.inf
    density_max = 0.0
    internal_min = math.inf
    internal_max = 0.0
    lorentz_max = 1.0
    sigma_max = 0.0
    with h5py.File(path, "r") as stream:
        primitive = np.asarray(stream["mhd.prim"], dtype=np.float64)
        magnetic = np.asarray(stream["mhd.b_cell"], dtype=np.float64)
        if not np.isfinite(primitive).all() or not np.isfinite(magnetic).all():
            raise RuntimeError(f"{path}: non-finite fluid or magnetic value")
        if np.any(primitive[:, 0] <= 0.0) or np.any(primitive[:, 4] <= 0.0):
            raise RuntimeError(f"{path}: non-positive density or internal energy")
        x_centres = np.asarray(stream["VolumeLocations/x"], dtype=np.float64)
        y_centres = np.asarray(stream["VolumeLocations/y"], dtype=np.float64)
        z_centres = np.asarray(stream["VolumeLocations/z"], dtype=np.float64)
        x_faces = np.asarray(stream["Locations/x"], dtype=np.float64)
        y_faces = np.asarray(stream["Locations/y"], dtype=np.float64)
        z_faces = np.asarray(stream["Locations/z"], dtype=np.float64)
        for index in range(primitive.shape[0]):
            z, y, x = np.meshgrid(
                z_centres[index], y_centres[index], x_centres[index], indexing="ij"
            )
            volume = (
                np.diff(z_faces[index])[:, None, None]
                * np.diff(y_faces[index])[None, :, None]
                * np.diff(x_faces[index])[None, None, :]
            )
            radius = kerr_radius(x, y, z)
            sx, sy, sz, lorentz, magnetic2 = stress_momentum(
                primitive[index], magnetic[index], x, y, z, radius
            )
            safe_radius = np.maximum(radius, 1.0e-14)
            radius2 = safe_radius * safe_radius
            metric_factor = 2.0 * radius2 * safe_radius / (
                radius2 * radius2 + SPIN * SPIN * z * z
            )
            alpha = 1.0 / np.sqrt(1.0 + metric_factor)
            blocks.append(
                {
                    "radius": radius,
                    "density": primitive[index, 0],
                    "position": (x, y, z),
                    "volume": volume,
                    "mass_weight": primitive[index, 0] * lorentz / alpha * volume,
                    "stress": (sx, sy, sz),
                }
            )
            physical = radius >= 1.0
            density_min = min(density_min, float(np.min(primitive[index, 0][physical])))
            density_max = max(density_max, float(np.max(primitive[index, 0][physical])))
            internal_min = min(internal_min, float(np.min(primitive[index, 4][physical])))
            internal_max = max(internal_max, float(np.max(primitive[index, 4][physical])))
            lorentz_max = max(lorentz_max, float(np.max(lorentz[physical])))
            sigma_max = max(
                sigma_max,
                float(
                    np.max(
                        np.maximum(magnetic2[physical], 0.0)
                        / np.maximum(primitive[index, 0][physical], 1.0e-300)
                    )
                ),
            )
        info = stream["Info"].attrs
        record: dict[str, object] = {
            "path": str(path),
            "time_M": float(info["Time"]),
            "cycle": int(info["NCycle"]),
            "mesh_blocks": int(info["NumMeshBlocks"]),
            "density_min": density_min,
            "density_max": density_max,
            "internal_energy_min": internal_min,
            "internal_energy_max": internal_max,
            "lorentz_max": lorentz_max,
            "sigma_max": sigma_max,
            "regions": [region_record(name, lower, upper, blocks) for name, lower, upper in REGIONS],
            "radial_profile": [
                region_record(f"radial_{index:02d}", lower, upper, blocks)
                for index, (lower, upper) in enumerate(zip(RADIAL_EDGES[:-1], RADIAL_EDGES[1:]))
            ],
        }
    return record


def outputs(run_dir: Path, all_snapshots: bool) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    for path in run_dir.glob("gr_chakrabarti_torus.prim.*.phdf"):
        match = NUMBERED_OUTPUT.match(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort()
    if not numbered:
        raise RuntimeError(f"{run_dir}: no numbered PHDF output")
    paths = [path for _, path in numbered]
    final = run_dir / "gr_chakrabarti_torus.prim.final.phdf"
    if final.is_file():
        with h5py.File(paths[-1], "r") as stream:
            last_time = float(stream["Info"].attrs["Time"])
        with h5py.File(final, "r") as stream:
            final_time = float(stream["Info"].attrs["Time"])
        if final_time > last_time:
            paths.append(final)
    return paths if all_snapshots or len(paths) == 1 else [paths[0], paths[-1]]


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    history_path = run_dir / "gr_chakrabarti_torus.out1.hst"
    values, errors = history(history_path)
    records = [snapshot(path) for path in outputs(run_dir, args.all_snapshots)]
    initial = records[0]
    latest = records[-1]
    initial_disk_mass = sum(float(row["disk_mass"]) for row in initial["regions"])
    latest_disk_mass = sum(float(row["disk_mass"]) for row in latest["regions"])
    disk_mass_fraction = latest_disk_mass / max(initial_disk_mass, np.finfo(float).tiny)
    if disk_mass_fraction < args.minimum_disk_mass_fraction:
        errors.append(
            f"disk mass fraction {disk_mass_fraction:.6e} is below "
            f"{args.minimum_disk_mass_fraction:.6e}"
        )
    max_divb = float(np.max(values[:, 13])) if values.shape[1] > 13 else math.inf
    if max_divb > args.divb_limit:
        errors.append(f"max|divB|={max_divb:.6e} exceeds {args.divb_limit:.6e}")
    if values.shape[1] > 14 and np.any(values[:, 14] > 1.0 + 1.0e-12):
        errors.append("cooling fraction exceeds unity")
    if values.shape[1] > 11 and values[-1, 11] <= 0.0:
        errors.append("cumulative removed energy is not positive")
    last_time = float(values[-1, 0])
    complete = math.isclose(last_time, args.target_time, rel_tol=0.0, abs_tol=1.0e-8)
    if args.require_complete and not complete:
        errors.append(f"trajectory ends at {last_time:g}M, expected {args.target_time:g}M")
    inner = latest["regions"][0]
    outer = latest["regions"][-1]
    inner_tilt = inner["tilt_deg"]
    outer_tilt = outer["tilt_deg"]
    bp_contrast = None
    bp_aligned = False
    if inner_tilt is not None and outer_tilt is not None:
        bp_contrast = float(outer_tilt) - float(inner_tilt)
        bp_aligned = (
            float(inner["disk_mass"]) >= args.minimum_bp_inner_mass
            and bp_contrast >= args.minimum_bp_contrast_deg
        )
    if args.require_bp and not bp_aligned:
        errors.append(
            "BP gate failed: require inner disk mass >= "
            f"{args.minimum_bp_inner_mass:.6e} and outer-minus-inner tilt >= "
            f"{args.minimum_bp_contrast_deg:.6e} deg; measured mass="
            f"{float(inner['disk_mass']):.6e}, contrast={bp_contrast}"
        )
    report = {
        "schema_version": 1,
        "case": "45-degree tilted, magnetized Chakrabarti torus with target-thickness cooling",
        "status": "pass" if not errors and (complete or not args.require_complete) else "fail",
        "target_time_M": args.target_time,
        "last_history_time_M": last_time,
        "complete": complete,
        "history_rows": int(values.shape[0]),
        "max_abs_divb": max_divb,
        "maximum_cooling_fraction": float(np.max(values[:, 14])),
        "removed_energy": float(values[-1, 11]),
        "boundary_energy": float(values[-1, 12]),
        "disk_mass_fraction": disk_mass_fraction,
        "bp_alignment": {
            "required": args.require_bp,
            "aligned": bp_aligned,
            "inner_tilt_deg": inner_tilt,
            "outer_tilt_deg": outer_tilt,
            "outer_minus_inner_tilt_deg": bp_contrast,
            "minimum_contrast_deg": args.minimum_bp_contrast_deg,
            "inner_disk_mass": float(inner["disk_mass"]),
            "minimum_inner_disk_mass": args.minimum_bp_inner_mass,
        },
        "snapshots": records,
        "errors": errors,
    }
    output = args.output or run_dir / "analysis/radiation_chakrabarti_acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"status={report['status']} t={last_time:g}M "
        f"disk_mass_fraction={disk_mass_fraction:.6e} max|divB|={max_divb:.6e}"
    )
    print(f"report={output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
