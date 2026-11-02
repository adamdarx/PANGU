#!/usr/bin/env python3
"""Exercise profile export/import, model rejection and restart on the supplied binary."""
from pathlib import Path
import argparse
import importlib.util
import json
import re
import resource
import subprocess

import h5py
import numpy as np


def edit(text, updates):
    for (section, name), value in updates.items():
        header = "<" + section + ">"
        start = text.index(header) + len(header)
        end = text.find("\n<", start)
        if end < 0:
            end = len(text)
        body = text[start:end]
        pattern = r"(?m)^[ \t]*" + re.escape(name) + r"[ \t]*=.*$"
        body = re.sub(pattern, f"{name} = {value}", body) if re.search(pattern, body) else body + f"\n{name} = {value}\n"
        text = text[:start] + body + text[end:]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    binary, root = args.pangu.resolve(), args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    template = (repo / "plugins/scalar_field/examples/ems_coupled_rn.in").read_text()
    base = {("parthenon/mesh", "nx1"): 1024, ("parthenon/meshblock", "nx1"): 1024,
            ("parthenon/mesh", "x1max"): 80, ("parthenon/time", "tlim"): 1,
            ("parthenon/output1", "dt"): .5, ("parthenon/output2", "dt"): .5,
            ("plugin/scalar_field", "amplitude"): .01,
            ("plugin/scalar_field", "constraint_iterations"): 16}
    spec = importlib.util.spec_from_file_location("export_ems", repo / "plugins/scalar_field/export_ems_profile.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    def run(name, changes=None, restart=None, rejection=None):
        directory = root / name
        directory.mkdir()
        deck = directory / "input.in"
        deck.write_text(edit(template, base | (changes or {})))
        command = [str(binary), "-i", str(deck)]
        if restart:
            command += ["-r", str(restart)]
        with (directory / "run.log").open("w") as stream:
            result = subprocess.run(command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT)
        log = (directory / "run.log").read_text()
        if rejection:
            assert result.returncode != 0 and rejection in log, (name, result.returncode)
            return directory
        assert result.returncode == 0 and "Driver completed." in log, name
        with h5py.File(next(directory.glob("*.final.phdf"))) as stream:
            assert abs(float(stream["Info"].attrs["Time"]) - 1) < 1e-12
            assert np.isfinite(stream["scalar.state"][:]).all()
        return directory

    seed = run("gaussian")
    profile = root / "gaussian.profile"
    exporter.export(next(seed.glob("*.final.phdf")), seed / "input.in", profile)
    imported = {("plugin/scalar_field", "initial_data"): "profile",
                ("plugin/scalar_field", "profile_file"): str(profile),
                ("plugin/scalar_field", "amplitude"): 0}
    control = run("imported", imported)
    checkpoint = next(p for p in control.glob("*.rhdf") if ".final." not in p.name and ".00000." not in p.name)
    resumed = run("resumed", imported, checkpoint)
    with h5py.File(next(control.glob("*.final.phdf"))) as a, h5py.File(next(resumed.glob("*.final.phdf"))) as b:
        difference = float(np.max(abs(a["scalar.state"][:] - b["scalar.state"][:])))
    assert difference < 1e-12, difference
    bad = root / "wrong_charge.profile"
    lines = profile.read_text().splitlines()
    tokens = lines[1].split()
    tokens[3] = ".8"
    lines[1] = " ".join(tokens)
    bad.write_text("\n".join(lines) + "\n")
    run("wrong_charge", imported | {("plugin/scalar_field", "profile_file"): str(bad)},
        rejection="action coefficients do not match")
    result = dict(status="passed", restart_max_difference=difference,
                  checks=["finite imported evolution", "restart", "wrong-model rejection"])
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
