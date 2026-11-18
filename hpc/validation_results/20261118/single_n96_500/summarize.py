"""Summarize the two archived PANGU N96 500M runs without new evolution."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np


def archived_bytes(entry, zstd):
    payload = subprocess.check_output([zstd, "-dc", entry["archive"]])
    if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"Archive integrity check failed: {entry['archive']}")
    return payload


def final_constraints(entry, zstd, scratch_dir):
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=scratch_dir, suffix=".phdf") as scratch:
        with subprocess.Popen([zstd, "-dc", entry["archive"]], stdout=subprocess.PIPE) as decoder:
            for part in iter(lambda: decoder.stdout.read(8 * 1024**2), b""):
                scratch.write(part)
                digest.update(part)
                size += len(part)
            if decoder.wait():
                raise RuntimeError(f"Failed to decode {entry['archive']}")
        if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Final archive integrity check failed: {entry['archive']}")
        scratch.flush()
        with h5py.File(scratch.name, "r") as field:
            attrs = field["Info"].attrs
            timestamp = float(attrs["Time"])
            if not math.isclose(timestamp, 500, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Unexpected final time {timestamp}")
            ghosts = int(attrs.get("NGhost", 0)) if int(attrs.get("IncludesGhost", 0)) else 0
            crop = (slice(ghosts, -ghosts),) * 3 if ghosts else (slice(None),) * 3
            active_cells = 0
            h_squared = 0.0
            m_squared = 0.0
            for start in range(0, field["nr.constraints"].shape[0], 16):
                slab = slice(start, start + 16)
                active = field["nr.constraint_mask"][(slab, 0, *crop)].astype(bool)
                hamiltonian = field["nr.constraints"][(slab, 1, *crop)][active]
                momentum = field["nr.constraints"][(slab, 2, *crop)][active]
                if not np.isfinite(hamiltonian).all() or not np.isfinite(momentum).all():
                    raise ValueError("Nonfinite constraint in final active cells")
                active_cells += int(active.sum())
                h_squared += float(np.square(hamiltonian).sum(dtype=np.float64))
                m_squared += float(np.square(momentum).sum(dtype=np.float64))
            if not active_cells:
                raise ValueError("Final field contains no active constraint cells")
    return dict(time=timestamp, constraint_mask_cells=active_cells,
                hamiltonian_RMS=math.sqrt(h_squared / active_cells),
                momentum_RMS=math.sqrt(m_squared / active_cells), field_sha256=entry["sha256"])


def read_diagnostics(entries, zstd):
    entry = next(e for e in entries if e["path"].endswith("/nr_horizon.csv"))
    source = archived_bytes(entry, zstd).decode()
    rows = csv.DictReader(line for line in io.StringIO(source) if not line.startswith("#"))
    fields = {"time": "time", "irreducible_mass": "irreducible_mass",
              "christodoulou_mass": "christodoulou_mass", "spin": "spin",
              "scaled_rms_expansion": "scaled_rms_expansion", "expansion_tolerance": "expansion_tolerance"}
    diagnostics = []
    total = 0
    for row in rows:
        total += 1
        data = {key: float(row[column]) for key, column in fields.items()}
        if (int(row["found"]) == 1 and all(math.isfinite(value) for value in data.values())
                and data["scaled_rms_expansion"] <= data["expansion_tolerance"]):
            diagnostics.append(data)
    if not diagnostics:
        raise ValueError("No qualified horizon measurements")
    return diagnostics, total, entry["sha256"]


def archived_constraints(entries, receipt_dir):
    rows = []
    for entry in entries:
        if not entry["path"].endswith(".phdf") or entry.get("snapshot_validation") != "passed":
            continue
        receipt = entry.get("validation_receipt")
        if receipt:
            receipt = Path(receipt)
        else:
            key = hashlib.sha256(entry["absolute_source"].encode()).hexdigest()
            receipt = receipt_dir / f"{key}.json"
        proof = json.loads(receipt.read_text())
        if (proof["source_sha256"] != entry["sha256"] or proof["status"] != "snapshot_checks_passed"
                or abs(proof["time"] - entry["time"]) > 1e-9):
            raise ValueError(f"Constraint receipt disagrees with archive index: {entry['path']}")
        rows.append(dict(time=proof["time"], constraint_mask_cells=proof["constraint_mask_cells"],
                         hamiltonian_RMS=proof["hamiltonian_L2"], momentum_RMS=proof["momentum_L2"],
                         field_sha256=entry["sha256"]))
    return sorted(rows, key=lambda row: row["time"])


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zstd", default="zstd")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.index.read_text())["files"]
    summary = dict(analysis="PANGU N96 stationary and spinning 500M archive",
                   constraints="Cell-count RMS over native active physical cells with ghost cells excluded.",
                   horizon="Each series uses its measured initial horizon as the mass and spin reference. The stationary case uses absolute spin changes.",
                   cases={})
    for kind, job in (("stationary", "17313453"), ("spinning", "17313451")):
        entries = [e for k, e in index.items() if f"/{job}/attempt-0/pangu/" in k]
        phase_entry = next(e for e in entries if e["path"].endswith("/phase-validated.json"))
        phase = json.loads(archived_bytes(phase_entry, args.zstd))
        if phase["status"] != "completed" or phase["returncode"] != 0 or phase["last_logged_time"] != 500:
            raise ValueError(f"PANGU phase {job} lacks completion evidence")
        horizons, total, horizon_sha = read_diagnostics(entries, args.zstd)
        if (len(horizons), total) != (phase["valid_horizons"], phase["horizon_samples"]):
            raise ValueError(f"Horizon counts disagree with phase receipt for {job}")
        reference = horizons[0]
        for row in horizons:
            row["irreducible_mass_relative_change"] = row["irreducible_mass"] / reference["irreducible_mass"] - 1
            row["christodoulou_mass_relative_change"] = row["christodoulou_mass"] / reference["christodoulou_mass"] - 1
            row["spin_change"] = row["spin"] - reference["spin"]
        final_entry = next(e for e in entries if e["path"].endswith(".final.phdf"))
        constraints = archived_constraints(entries, args.index.parent / "snapshot-validation")
        constraints.append(final_constraints(final_entry, args.zstd, args.output))
        write_csv(args.output / f"{kind}_horizons.csv", horizons)
        write_csv(args.output / f"{kind}_constraints.csv", constraints)
        summary["cases"][kind] = dict(job=job, valid_horizon_samples=len(horizons), total_horizon_samples=total,
            horizon_source_sha256=horizon_sha, phase_receipt_sha256=phase_entry["sha256"],
            horizon_first=reference, horizon_last=horizons[-1],
            max_abs_relative_irreducible_mass_change=max(abs(r["irreducible_mass_relative_change"]) for r in horizons),
            max_abs_relative_christodoulou_mass_change=max(abs(r["christodoulou_mass_relative_change"]) for r in horizons),
            max_abs_spin_change=max(abs(r["spin_change"]) for r in horizons),
            constraint_samples=len(constraints), first_constraints=constraints[0], final_constraints=constraints[-1])
        print(kind, summary["cases"][kind]["final_constraints"], flush=True)
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
