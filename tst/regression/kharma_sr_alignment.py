#!/usr/bin/env python3
"""Run and audit the KHARMA-SR-2/3 matched CUDA campaign.

All generated products are confined to --data-root (data/kharma_sr_alignment
by default).  The KHARMA comparison uses its face-CT/GS05 path so both codes
advance the same staggered magnetic topology.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
from typing import Iterable

import h5py
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODE_OMEGA = {0: 0.0, 1: 2.41024185339, 2: 3.44144232573, 3: 5.53726217331}
SHOCKS = {
    "fast": (2.5, 0.4, (1.0, 1.0, 25.0, 0.0, 0.0, 20.0, 25.02, 0.0),
             (25.48, 367.5, 1.091, 0.3923, 0.0, 20.0, 49.0, 0.0)),
    "slow": (2.0, 0.5, (1.0, 10.0, 1.53, 0.0, 0.0, 10.0, 18.28, 0.0),
             (3.323, 55.36, 0.9571, -0.6822, 0.0, 10.0, 14.49, 0.0)),
    "switch_on": (2.0, 0.5, (1.78e-3, 0.1, -0.765, -1.386, 0.0, 1.0, 1.022, 0.0),
                  (0.01, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    "switch_off": (1.0, 0.5, (0.1, 1.0, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0),
                   (0.562, 10.0, -0.212, -0.590, 0.0, 2.0, 4.71, 0.0)),
    "collision": (1.22, 0.5, (1.0, 1.0, 5.0, 0.0, 0.0, 10.0, 10.0, 0.0),
                  (1.0, 1.0, -5.0, 0.0, 0.0, 10.0, -10.0, 0.0)),
    "shock_1": (1.0, 0.5, (1.0, 1000.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                (0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    "shock_2": (1.0, 0.5, (1.0, 30.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0),
                (0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
}
FIXTURES = {
    "b0_rest_cold": (1.0, 0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "b0_rest_hot": (1.0, 10.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "b0_fast_flow": (0.7, 0.3, (3.0, -0.2, 0.1), (0.0, 0.0, 0.0)),
    "parallel_field": (1.0, 1.0, (0.4, 0.1, -0.2), (2.0, 0.0, 0.0)),
    "perpendicular_field": (1.0, 1.0, (0.4, 0.1, -0.2), (0.0, 2.0, 0.0)),
    "general_field": (0.8, 0.2, (1.2, -0.7, 0.3), (1.7, -0.4, 0.9)),
    "weak_field": (1.0, 0.1, (0.2, 0.0, 0.0), (1.0e-5, 0.0, 0.0)),
    "strong_field": (1.0, 0.1, (4.0, 0.3, -0.2), (31.622776601683793, 2.0, -1.0)),
}


def run(command: list[str], directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {directory / 'run.log'}")


def final_file(directory: pathlib.Path) -> pathlib.Path:
    files = sorted(directory.glob("*.final.phdf"))
    if len(files) != 1:
        raise RuntimeError(f"expected one final PHDF in {directory}, found {len(files)}")
    return files[0]


def load(path: pathlib.Path, code: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    with h5py.File(path, "r") as h5:
        x = np.asarray(h5["VolumeLocations/x"][0], dtype=np.float64)
        info = {name: float(h5["Info"].attrs[name]) for name in ("Time", "NCycle", "dt")}
        if code == "pangu":
            p = np.asarray(h5["mhd.prim"][0, :, 0], dtype=np.float64)
            b = np.asarray(h5["mhd.b_cell"][0, :, 0], dtype=np.float64)
            values = np.stack((p[0], p[4], p[1], p[2], p[3], b[0], b[1], b[2]))
        else:
            rho = np.asarray(h5["prims.rho"][0, 0], dtype=np.float64)
            internal = np.asarray(h5["prims.u"][0, 0], dtype=np.float64)
            uvec = np.asarray(h5["prims.uvec"][0, :, 0], dtype=np.float64)
            b = np.asarray(h5["prims.B"][0, :, 0], dtype=np.float64)
            values = np.concatenate((rho[None], internal[None], uvec, b), axis=0)
    if not np.all(np.isfinite(values)) or np.min(values[0]) <= 0.0 or np.min(values[1]) <= 0.0:
        raise RuntimeError(f"nonphysical output in {path}")
    return x, values, info


def common_overrides(resolution: int) -> list[str]:
    return [f"parthenon/mesh/nx1={resolution}", f"parthenon/mesh/nx2={resolution}",
            "parthenon/mesh/nx3=1", f"parthenon/meshblock/nx1={resolution}",
            f"parthenon/meshblock/nx2={resolution}", "parthenon/meshblock/nx3=1"]


def fixture_runs(args: argparse.Namespace) -> list[dict]:
    records = []
    for name, (rho, pressure, velocity, magnetic) in FIXTURES.items():
        base = args.data_root / "fixtures" / name
        pdir, kdir = base / "pangu", base / "kharma"
        shared = common_overrides(8)
        internal = pressure / (4.0 / 3.0 - 1.0)
        run([str(args.pangu), "-i", str(ROOT / "input/relativity/sr_mhd_modes.in"),
             "-d", str(pdir), *shared, "problem/kharma_mode=0", "problem/amplitude=0",
             f"problem/mode_density0={rho:.17g}", f"problem/mode_internal0={internal:.17g}",
             f"problem/mode_u10={velocity[0]:.17g}", f"problem/mode_u20={velocity[1]:.17g}",
             f"problem/mode_u30={velocity[2]:.17g}", f"problem/mode_b10={magnetic[0]:.17g}",
             f"problem/mode_b20={magnetic[1]:.17g}", f"problem/mode_b30={magnetic[2]:.17g}",
             "mhd/rsolver=llf", "mhd/cfl=0.9", "parthenon/time/integrator=rk1",
             "parthenon/time/dt=1e-4", "parthenon/time/dt_init=1e-4",
             "parthenon/time/dt_init_force=true", "parthenon/time/dt_floor=1e-4",
             "parthenon/time/dt_factor=2", "parthenon/time/tlim=1e-4",
             "parthenon/time/nlim=1",
             "parthenon/output2/dt=100", "parthenon/output2/single_precision_output=false",
             "parthenon/output2/variables=mhd.prim,mhd.b_cell,mhd.cmax,mhd.cmin"], pdir)
        run([str(args.kharma), "-i", str(ROOT / "kharma/pars/tests/mhdmodes.par"),
             "-d", str(kdir), *shared, "mhdmodes/nmode=0", "mhdmodes/dir=3",
             "mhdmodes/amp=0", "mhdmodes/one_period=false", f"mhdmodes/rho0={rho:.17g}",
             f"mhdmodes/u0={internal:.17g}", f"mhdmodes/u10={velocity[0]:.17g}",
             f"mhdmodes/u20={velocity[1]:.17g}", f"mhdmodes/u30={velocity[2]:.17g}",
             f"b_field/B10={magnetic[0]:.17g}", f"b_field/B20={magnetic[1]:.17g}",
             f"b_field/B30={magnetic[2]:.17g}", "GRMHD/gamma=1.3333333333333333",
             "GRMHD/cfl=0.9", "flux/type=llf", "flux/reconstruction=linear_mc",
             # One RK1 stage makes the saved characteristic cache a direct function
             # of the common initial primitive state.  RK2 would run a KHARMA UtoP
             # between stages and contaminate this estimator-only fixture.
             "driver/type=kharma", "parthenon/time/integrator=rk1",
             "b_field/solver=face_ct", "b_field/ct_scheme=gs05_c",
             "parthenon/time/dt_min=1e-4", "parthenon/time/tlim=1e-4",
             "parthenon/time/nlim=1", "parthenon/output0/dt=100",
             "parthenon/output0/single_precision_output=false",
             "parthenon/output0/variables=prims,Flux.cmax,Flux.cmin"], kdir)
        pf, kf = final_file(pdir), final_file(kdir)
        with h5py.File(pf, "r") as ph, h5py.File(kf, "r") as kh:
            pmax = np.asarray(ph["mhd.cmax"][0, :, 0], dtype=np.float64)
            pmin = np.asarray(ph["mhd.cmin"][0, :, 0], dtype=np.float64)
            kmax = np.asarray(kh["Flux.cmax"][0, :, 0], dtype=np.float64)
            kmin = np.asarray(kh["Flux.cmin"][0, :, 0], dtype=np.float64)
        records.append({"case": name,
                        "cmax_linf": float(np.max(np.abs(pmax - kmax))),
                        "cmin_linf": float(np.max(np.abs(pmin - kmin))),
                        "pangu_dt": load(pf, "pangu")[2]["dt"],
                        "kharma_dt": load(kf, "kharma")[2]["dt"]})
    return records


def mode_runs(args: argparse.Namespace, modes: Iterable[int], resolutions: Iterable[int]) -> list[dict]:
    records = []
    for mode in modes:
        tlim = 1.0 if mode == 0 else 2.0 * math.pi / abs(MODE_OMEGA[mode])
        for n in resolutions:
            base = args.data_root / "modes" / f"mode{mode}_N{n}"
            pdir, kdir = base / "pangu", base / "kharma"
            shared = common_overrides(n)
            run([str(args.pangu), "-i", str(ROOT / "input/relativity/sr_mhd_modes.in"),
                 "-d", str(pdir), *shared, f"problem/kharma_mode={mode}",
                 f"parthenon/time/tlim={tlim:.17g}"], pdir)
            run([str(args.kharma), "-i", str(ROOT / "kharma/pars/tests/mhdmodes.par"),
                 "-d", str(kdir), *shared, f"mhdmodes/nmode={mode}", "mhdmodes/dir=3",
                 "mhdmodes/amp=1e-4", "mhdmodes/one_period=false",
                 f"parthenon/time/tlim={tlim:.17g}", "parthenon/time/dt_min=1e-4",
                 "GRMHD/gamma=1.3333333333333333", "GRMHD/cfl=0.9",
                 "flux/type=llf", "flux/reconstruction=linear_mc", "driver/type=kharma",
                 "b_field/solver=face_ct", "b_field/ct_scheme=gs05_c",
                 "parthenon/output0/dt=100", "parthenon/output0/single_precision_output=false"], kdir)
            xp, vp, ip = load(final_file(pdir), "pangu")
            xk, vk, ik = load(final_file(kdir), "kharma")
            if not np.array_equal(xp, xk):
                raise RuntimeError(f"native grids differ for mode {mode}, N={n}")
            delta = vp - vk
            records.append({"mode": mode, "resolution": n, "pangu": ip, "kharma": ik,
                            "l1": np.mean(np.abs(delta), axis=(1, 2)).tolist(),
                            "linf": np.max(np.abs(delta), axis=(1, 2)).tolist()})
    return records


def shock_runs(args: argparse.Namespace) -> list[dict]:
    records = []
    pinput = ROOT / "input/relativity/sr_mhd_komissarov.in"
    for name, (tlim, cfl, left, right) in SHOCKS.items():
        base = args.data_root / "shocks" / name
        pdir, kdir = base / "pangu", base / "kharma"
        pnames = ("density", "pressure", "velocity_1", "velocity_2", "velocity_3",
                  "b1", "b2", "b3")
        overrides = [f"problem/{key}_left={value:.17g}" for key, value in zip(pnames, left)]
        overrides += [f"problem/{key}_right={value:.17g}" for key, value in zip(pnames, right)]
        # The normal CT field is single-valued; all canonical Komissarov decks satisfy B1L=B1R.
        overrides = [item for item in overrides if not item.startswith("problem/b1_")]
        overrides += [f"problem/b1={left[5]:.17g}", f"mhd/cfl={cfl:.17g}",
                      f"parthenon/time/tlim={tlim:.17g}", "parthenon/mesh/nghost=4",
                      "parthenon/mesh/ix1_bc=periodic", "parthenon/mesh/ox1_bc=periodic"]
        run([str(args.pangu), "-i", str(pinput), "-d", str(pdir), *overrides], pdir)
        kinput = ROOT / "kharma/pars/shocks" / f"komissarov_{name}.par"
        run([str(args.kharma), "-i", str(kinput), "-d", str(kdir),
             # Preserve the canonical 1D shock configuration.  In particular,
             # KHARMA intentionally uses flux_ct here because face_ct rejects 1D.
             "parthenon/output0/single_precision_output=false"], kdir)
        xp, vp, ip = load(final_file(pdir), "pangu")
        xk, vk, ik = load(final_file(kdir), "kharma")
        if not np.allclose(xp, xk, rtol=0.0, atol=8.0 * np.finfo(np.float64).eps):
            raise RuntimeError(f"native grids differ for shock {name}")
        delta = vp - vk
        gp = np.abs(np.diff(vp[0, 0]))
        gk = np.abs(np.diff(vk[0, 0]))
        # Several canonical tubes produce a symmetric pair of equally strong
        # fronts.  Comparing one argmax from each code can pair opposite/tied
        # fronts, so compare all peaks within 90% of the strongest gradient.
        front_p = 0.5 * (xp[:-1] + xp[1:])
        front_k = 0.5 * (xk[:-1] + xk[1:])
        candidates_p = front_p[gp >= 0.9 * np.max(gp)]
        candidates_k = front_k[gk >= 0.9 * np.max(gk)]
        dx = float(xp[1] - xp[0])
        front_cells = float(np.min(np.abs(candidates_p[:, None] - candidates_k[None, :])) / dx)
        scales = np.maximum(np.maximum(np.max(np.abs(vp), axis=(1, 2)),
                                       np.max(np.abs(vk), axis=(1, 2))), 1.0e-300)
        l1 = np.mean(np.abs(delta), axis=(1, 2))
        records.append({"case": name, "pangu": ip, "kharma": ik,
                        "front_cell_difference": front_cells,
                        "l1": l1.tolist(),
                        "relative_l1": (l1 / scales).tolist(),
                        "linf": np.max(np.abs(delta), axis=(1, 2)).tolist(),
                        "pangu_gamma_max": float(np.max(np.sqrt(1.0 + np.sum(vp[2:5] ** 2, axis=0)))),
                        "kharma_gamma_max": float(np.max(np.sqrt(1.0 + np.sum(vk[2:5] ** 2, axis=0))))})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", type=pathlib.Path,
                        default=ROOT / "build-nr-cuda128/src/pangu")
    parser.add_argument("--kharma", type=pathlib.Path,
                        default=ROOT / "kharma/build/kharma/kharma.cuda")
    parser.add_argument("--data-root", type=pathlib.Path,
                        default=ROOT / "data/kharma_sr_alignment")
    parser.add_argument("--stage", choices=("fixtures", "modes", "shocks", "all"), default="all")
    parser.add_argument("--modes", default="0,1,2,3")
    parser.add_argument("--resolutions", default="32,64,128,256")
    args = parser.parse_args()
    args.pangu = args.pangu.resolve()
    args.kharma = args.kharma.resolve()
    args.data_root = args.data_root.resolve()
    report: dict[str, object] = {"component_order": ["rho", "u", "u1", "u2", "u3", "B1", "B2", "B3"]}
    if args.stage in ("fixtures", "all"):
        report["fixtures"] = fixture_runs(args)
    if args.stage in ("modes", "all"):
        modes = [int(value) for value in args.modes.split(",")]
        resolutions = [int(value) for value in args.resolutions.split(",")]
        report["modes"] = mode_runs(args, modes, resolutions)
    if args.stage in ("shocks", "all"):
        report["shocks"] = shock_runs(args)
    args.data_root.mkdir(parents=True, exist_ok=True)
    output = args.data_root / f"{args.stage}_acceptance.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
