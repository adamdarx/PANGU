#!/usr/bin/env python3
"""Small CUDA/Slurm admission cases; a smoke pass is not a production result."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

import h5py
import numpy as np


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def inspect(path):
    with h5py.File(path) as stream:
        fields = {name: stream[name][:] for name in (
            "mhd.prim", "mhd.cons", "mhd.b_cell", "mhd.divb", "mhd.fofc",
            "nr.z4c", "nr.adm", "nr.tmunu", "nr.constraints")}
        for name, values in fields.items():
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite {name} in {path}")
        primitive = fields["mhd.prim"]
        core = primitive[:, 0] > .01 * primitive[:, 0].max()
        speed = np.sqrt(np.sum(primitive[:, 1:4]**2, axis=1))
        dx = np.diff(stream["Locations/x"][:], axis=-1)
        dy = np.diff(stream["Locations/y"][:], axis=-1)
        dz = np.diff(stream["Locations/z"][:], axis=-1)
        volume = dz[:, :, None, None] * dy[:, None, :, None] * dx[:, None, None, :]
        return dict(time=float(stream["Info"].attrs["Time"]),
                    rho_max=float(primitive[:, 0].max()),
                    core_projected_speed_max=float(speed[core].max()),
                    baryon_mass=float(np.sum(fields["mhd.cons"][:, 0] * volume)),
                    magnetic_max=float(np.sqrt(np.sum(fields["mhd.b_cell"]**2, axis=1)).max()),
                    divb_max=float(np.abs(fields["mhd.divb"]).max()),
                    hamiltonian_max=float(np.abs(fields["nr.constraints"][:, 1]).max()),
                    fofc_flagged=int(np.count_nonzero(fields["mhd.fofc"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["stable", "collapse"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    job = os.environ["SLURM_JOB_ID"]
    assert int(os.environ["SLURM_NTASKS"]) == 1
    binary = root / "bin/pangu-a2bbf14"
    names = ["stable_b0", "stable_b1"] if args.family == "stable" else ["unstable_b1"]
    work = root / "runs" / args.family / job
    work.mkdir(parents=True, exist_ok=False)
    report = dict(family=args.family, job=job, status="running", cases=[],
                  binary_sha256=sha(binary), source_commit=(root / "source-commit.txt").read_text().strip(),
                  physical_acceptance="pending formation, long-time and resolution validation")
    def save():
        temporary = work / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(work / "summary.json")
    save()
    try:
        for name in names:
            folder = work / name
            folder.mkdir()
            deck = root / "inputs" / (name + ".in")
            command = ["mpirun", "-np", "1", "--map-by", "ppr:1:node", "--bind-to", "none",
                       "--mca", "io", "ompio", "bash", str(root / "scripts/rank.sh"),
                       str(binary), "-i", str(deck)]
            (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            start = time.monotonic()
            with (folder / "run.log").open("w") as stream:
                subprocess.run(["/usr/bin/time", "-v", "-o", str(folder / "time.txt")] + command,
                               cwd=folder, stdout=stream, stderr=subprocess.STDOUT, check=True)
            log = (folder / "run.log").read_text()
            assert "Driver completed." in log, "driver completion missing"
            snapshots = sorted(folder.glob("*.phdf"))
            assert len(snapshots) >= 3, "insufficient snapshots"
            samples = [inspect(path) for path in snapshots if ".final." not in path.name]
            final = list(folder.glob("*.final.phdf"))
            assert len(final) == 1
            first, last = samples[0], inspect(final[0])
            target = 2 if args.family == "stable" else 10
            assert abs(last["time"] - target) < 1e-10
            density_drift = abs(last["rho_max"] / first["rho_max"] - 1)
            mass_drift = abs(last["baryon_mass"] / first["baryon_mass"] - 1)
            assert max(s["divb_max"] for s in samples + [last]) < 1e-12
            if args.family == "stable":
                assert density_drift < .01, (name, "density", density_drift)
                assert mass_drift < 1e-5, (name, "baryon mass", mass_drift)
                assert last["core_projected_speed_max"] < .01
            row = dict(case=name, status="smoke_passed", time=last["time"],
                       wall_seconds=time.monotonic() - start, input_sha256=sha(deck),
                       relative_density_change=density_drift, relative_baryon_mass_change=mass_drift,
                       first=first, final=last, samples=samples)
            report["cases"].append(row)
            save()
            print(json.dumps(row), flush=True)
        report["status"] = "smoke_passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
