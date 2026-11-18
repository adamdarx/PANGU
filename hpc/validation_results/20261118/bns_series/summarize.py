"""Compare the archived PANGU head-on BNS series and a live N128 snapshot."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def history(directory):
    path = directory / "sync_plugin_grmhd.out2.hst"
    result = []
    lines = path.read_text().splitlines(keepends=True)
    for line in lines:
        if line.startswith("#") or not line.strip() or not line.endswith("\n"):
            continue
        row = [float(value) for value in line.split()]
        if len(row) != 14 or not all(math.isfinite(value) for value in row):
            raise ValueError(f"Malformed or nonfinite history in {path}")
        result.append({"time": row[0], "cycle": int(row[2]), "baryon_mass": row[4]})
    return result


def horizons(directory):
    path = directory / "nr_horizon.csv"
    with path.open() as file:
        rows = []
        for row in csv.DictReader(line for line in file if not line.startswith("#") and line.endswith("\n")):
            record = {name: float(row[name]) for name in (
                "time", "irreducible_mass", "christodoulou_mass", "max_radius",
                "scaled_rms_expansion", "expansion_tolerance")}
            record["cycle"] = int(row["cycle"])
            record["qualified"] = (
                int(row["found"]) == 1 and all(math.isfinite(x) for x in record.values())
                and 0 <= record["scaled_rms_expansion"] <= record["expansion_tolerance"]
            )
            rows.append(record)
        return rows


def write_csv(path, rows):
    with path.open("w") as file:
        writer = csv.DictWriter(file, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.source / "PROTOCOL.json").read_text())
    adm = protocol["adm_mass"]
    summary = {
        "model": protocol["model"],
        "initial_solved_ADM_mass": adm,
        "separation": protocol["separation"],
        "initial_table_sha256": protocol["source_table_sha256"],
        "histories": "Baryon mass drift uses each run's own time-zero mass. The complete volume includes the black hole interior.",
        "horizon_measurement": "PANGU found flag and native scaled-expansion tolerance. Failed searches are retained in the source snapshots.",
        "source_hashes": {
            str(path.relative_to(args.source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.source.rglob("*")) if path.is_file()
        },
        "cases": {},
    }
    for n in (64, 96, 128):
        key = f"n{n}"
        directory = args.source / key
        original = json.loads((directory / "summary.json").read_text())
        masses = history(directory)
        attempts = horizons(directory)
        if n == 96:
            restart_cycle = original["restart_cycle"]
            prefix = args.source / "n96-prefix"
            masses = ([row for row in history(prefix) if row["cycle"] <= restart_cycle]
                      + [row for row in masses if row["cycle"] > restart_cycle])
            attempts = ([row for row in horizons(prefix) if row["cycle"] <= restart_cycle]
                        + [row for row in attempts if row["cycle"] > restart_cycle])
        if masses[0]["time"] != 0 or any(a["time"] >= b["time"] for a, b in zip(masses, masses[1:])):
            raise ValueError(f"{key} history is missing time zero or has overlapping segments")
        if n != 128:
            proof = json.loads((directory / "phase-validated.json").read_text())
            if proof["final"]["time"] != 200 or masses[-1]["time"] != 200:
                raise ValueError(f"{key} completion evidence disagrees with history")
        initial_baryon_mass = masses[0]["baryon_mass"]
        for row in masses:
            row["relative_baryon_change"] = row["baryon_mass"] / initial_baryon_mass - 1
        good = [row for row in attempts if row["qualified"]]
        result = dict(job=original["job_id"], status=original["status"], latest_history_time=masses[-1]["time"],
                      initial_baryon_mass=initial_baryon_mass,
                      latest_relative_baryon_change=masses[-1]["relative_baryon_change"],
                      max_absolute_relative_baryon_change_through_80=max(
                          abs(row["relative_baryon_change"]) for row in masses if row["time"] <= 80
                      ), horizon_attempts=len(attempts), qualified_horizons=len(good))
        if good:
            for row in good:
                row["relative_christodoulou_mass_minus_initial_ADM"] = row["christodoulou_mass"] / adm - 1
            previous_failure = max(row["time"] for row in attempts if not row["qualified"] and row["time"] < good[0]["time"])
            shell = original["coordinate_shells_final"]["regions"]["8.0"]
            if good[-1]["max_radius"] >= 8:
                raise ValueError("Reported horizon extends into the comparison shell")
            result.update(horizon_sampling_bracket=[previous_failure, good[0]["time"]],
                          last_qualified_horizon=good[-1], shell_8_to_30_at_200=shell)
            write_csv(args.output / f"{key}_qualified_horizons.csv", good)
        summary["cases"][key] = result
        write_csv(args.output / f"{key}_baryon_history.csv", masses)
    low, high = (summary["cases"][key] for key in ("n64", "n96"))
    x, y = (case["last_qualified_horizon"]["relative_christodoulou_mass_minus_initial_ADM"] for case in (low, high))
    summary["N64_to_N96_positive_AH_mass_excess_reduction_fraction"] = 1 - y / x
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
