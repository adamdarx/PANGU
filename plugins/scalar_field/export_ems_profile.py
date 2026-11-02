#!/usr/bin/env python3
"""Export a relaxed PG EMS snapshot with explicit action normalization."""
from pathlib import Path
import argparse
import hashlib
import re

import h5py
import numpy as np


def parameters(path):
    section = ""
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        header = re.fullmatch(r"<([^>]+)>", line)
        if header:
            section = header[1]
        elif section == "plugin/scalar_field" and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if values.get("backend") != "spherical_ems":
        raise ValueError("profile export requires a spherical_ems input deck")
    return values


def export(snapshot, deck, output):
    p = parameters(deck)
    potential = p.get("potential", "polynomial")
    coupling = p.get("em_coupling", "exponential")
    if potential not in ("polynomial", "axion") or coupling not in ("exponential", "quadratic"):
        raise ValueError("unsupported EMS profile model")
    with h5py.File(snapshot) as stream:
        coordinates = np.asarray(stream["VolumeLocations/x"])
        if coordinates.ndim != 2 or coordinates.shape[0] != 1:
            raise ValueError("only one radial MeshBlock can be exported")
        r = coordinates[0]
        field = np.asarray(stream["scalar.state"])
        if field.shape[0] != 1 or field.shape[-1] != r.size or field.size != 3*r.size:
            raise ValueError("expected phi,P,m in one radial MeshBlock")
        state = field.reshape(3, r.size)
        time = float(stream["Info"].attrs["Time"])
    if r.size < 4 or not np.all(np.diff(r) > 0):
        raise ValueError("profile coordinates are not strictly increasing")
    if not np.isfinite(state).all() or not np.isfinite(r).all() or np.any(state[2] <= 0):
        raise ValueError("nonfinite profile or nonpositive mass")
    constants = [float(p.get(key, default)) for key, default in (
        ("bh_charge", ".9"), ("em_alpha", "5"), ("mass", "0"),
        ("quartic", "0"), ("sextic", "0"), ("decay_constant", "1"))]
    constants[2] **= 2
    if not np.isfinite(constants).all():
        raise ValueError("nonfinite action coefficient")
    source_hash = hashlib.file_digest(snapshot.open("rb"), "sha256").hexdigest()
    with output.open("x") as stream:
        stream.write("# pangu_ems_profile_v1\n")
        stream.write("# " + potential + " " + coupling + " " +
                     " ".join(format(x, ".17g") for x in constants) + "\n")
        stream.write(f"# source_sha256 {source_hash} source_time {time:.17g}\n")
        stream.write("# columns: areal_radius phi P misner_sharp_mass\n")
        np.savetxt(stream, np.column_stack((r, state.T)), fmt="%.17e")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.snapshot, args.input, args.output))
