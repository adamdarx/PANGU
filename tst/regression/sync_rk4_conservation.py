#!/usr/bin/env python3
"""Check synchronized RK4 fluid/CT registers against constant-state invariants."""

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
    for (section, key), value in changes.items():
        start = text.index("<" + section + ">") + len(section) + 2
        end = text.find("\n<", start)
        if end < 0:
            end = len(text)
        body = text[start:end]
        pattern = r"(?m)^" + re.escape(key) + r"\s*=.*$"
        body = (re.sub(pattern, f"{key} = {value}", body) if re.search(pattern, body)
                else body + f"\n{key} = {value}\n")
        text = text[:start] + body + text[end:]
    return text


def measure(path, fluid):
    with h5py.File(path) as f:
        fields = [f"{fluid}.cons", f"{fluid}.prim", "nr.z4c", "nr.adm", "nr.tmunu"]
        if fluid == "mhd":
            fields += ["mhd.b_cell", "mhd.b_face", "mhd.divb"]
        for name in fields:
            if not np.isfinite(f[name][:]).all():
                raise RuntimeError(f"nonfinite {name}: {path}")
        h = [np.diff(f["Locations/" + a][:], axis=-1) for a in "xyz"]
        volume = h[2][:, :, None, None] * h[1][:, None, :, None] * h[0][:, None, None, :]
        cons = f[f"{fluid}.cons"][:]
        return {
            "time": float(f["Info"].attrs["Time"]),
            "blocks": int(f["Info"].attrs["NumMeshBlocks"]),
            "integrals": np.sum(cons * volume[:, None], axis=(0, 2, 3, 4)).tolist(),
            "max_divb": float(abs(f["mhd.divb"][:]).max()) if fluid == "mhd" else 0.0,
            "max_tmunu": float(abs(f["nr.tmunu"][:]).max()),
            "rho_min": float(f[f"{fluid}.prim"][:, 0].min()),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pangu", required=True, type=Path)
    p.add_argument("--workdir", required=True, type=Path)
    cases = ["grhd_constant", "mhd_constant", "mhd_wave_amr", "mhd_constant_rk3"]
    p.add_argument("--cases", choices=cases, nargs="+", default=cases)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    binary, root = args.pangu.resolve(), args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    report = {"status": "running", "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "schema": "pangu.sync-rk4-conservation.v1", "runs": {}}

    def save():
        temporary = root / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(root / "summary.json")

    save()
    for case in args.cases:
        fluid = "hydro" if case.startswith("grhd") else "mhd"
        filename = "sync_grhd_linear_wave.in" if fluid == "hydro" else "sync_grmhd_linear_wave_3d.in"
        text = (repo / "input/numerical_relativity" / filename).read_text()
        refined = case == "mhd_wave_amr"
        n = 32 if refined else 16
        changes = {
            ("parthenon/mesh", "refinement"): "static" if refined else "none",
            ("parthenon/mesh", "nghost"): 4,
            ("parthenon/time", "integrator"): "rk3" if case.endswith("rk3") else "rk4",
            ("parthenon/time", "tlim"): 0.025,
            ("parthenon/time", "nlim"): -1,
            ("parthenon/time", "ncycle_out"): 1,
            ("numerical_relativity", "finite_difference_order"): 4,
            ("numerical_relativity", "rk4_variant"): "athenak",
            ("numerical_relativity", "matter_source"): "zero",
            ("numerical_relativity", "cfl"): 0.2,
            (fluid, "cfl"): 0.2,
            (fluid, "fofc"): "false" if fluid == "hydro" else "true",
            ("parthenon/output1", "dt"): 0.01,
        }
        for axis in (1, 2, 3):
            changes["parthenon/mesh", f"nx{axis}"] = n
            changes["parthenon/mesh", f"x{axis}min"] = -0.5
            changes["parthenon/mesh", f"x{axis}max"] = 0.5
            changes["parthenon/meshblock", f"nx{axis}"] = 8
        for key in ["tracker_enabled", "horizon_enabled", "weyl_enabled", "waveform_enabled"]:
            changes["numerical_relativity", key] = "false"
        if fluid == "mhd":
            changes.update({("problem", "amp"): 1e-6 if refined else 0,
                            ("problem", "along_x1"): "true"})
        else:
            changes.update({("problem", "amplitude"): 0,
                            ("problem", "ambient_density"): 4,
                            ("problem", "ambient_pressure"): 1})
        text = edit(text, changes)
        if refined:
            text += "\n<parthenon/static_refinement0>\nlevel = 1\n" + "".join(
                f"x{a}min = -0.25\nx{a}max = 0.25\n" for a in (1, 2, 3))
        directory = root / case
        directory.mkdir()
        deck = directory / "input.in"
        deck.write_text(text)
        result = report["runs"][case] = {"status": "running"}
        save()
        start = time.monotonic()
        try:
            with (directory / "run.log").open("w") as output:
                completed = subprocess.run([str(binary), "-i", str(deck)], cwd=directory,
                                           stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout)
            result.update(exit_code=completed.returncode, wall_seconds=time.monotonic() - start)
            assert completed.returncode == 0, "evolution failed"
            assert "Driver completed." in (directory / "run.log").read_text()
            paths = list(directory.glob("*.phdf"))
            samples = sorted([measure(path, fluid) for path in paths], key=lambda x: x["time"])
            first, final = samples[0], samples[-1]
            result["samples"] = samples
            assert first["time"] == 0 and abs(final["time"] - .025) < 1e-12
            scale = max(abs(x) for x in first["integrals"])
            result["integral_scaled_error"] = max(
                max(abs(a - b) for a, b in zip(s["integrals"], first["integrals"])) / scale
                for s in samples)
            assert result["integral_scaled_error"] < 5e-12, "periodic conserved integrals changed"
            assert max(s["max_divb"] for s in samples) < 1e-10, "CT divergence"
            assert max(s["max_tmunu"] for s in samples) == 0
            assert min(s["rho_min"] for s in samples) > 0
            initial = next(path for path in paths if ".00000." in path.name)
            last = next(directory.glob("*.final.phdf"))
            with h5py.File(initial) as a, h5py.File(last) as b:
                result["metric_change"] = float(abs(a["nr.z4c"][:] - b["nr.z4c"][:]).max())
                assert result["metric_change"] < 1e-12
                if not refined:
                    fields = [f"{fluid}.cons", f"{fluid}.prim"]
                    if fluid == "mhd":
                        fields += ["mhd.b_cell", "mhd.b_face"]
                    result["constant_state_scaled_error"] = max(
                        float(abs(a[name][:] - b[name][:]).max()) /
                        max(1.0, float(abs(a[name][:]).max())) for name in fields)
                    assert result["constant_state_scaled_error"] < 5e-12, "constant state changed"
            if refined:
                assert final["blocks"] == 120, "coarse/fine interface not exercised"
            result["status"] = "passed"
        except Exception as exc:
            result.update(status="failed", error=repr(exc), wall_seconds=time.monotonic() - start)
        save()
        print(case, json.dumps(result), flush=True)
    report["status"] = "passed" if all(r["status"] == "passed" for r in report["runs"].values()) else "failed"
    save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
