#!/usr/bin/env python3
"""GPU AMR regression: local/cross-rank restriction, prolongation and tracker order."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import re
import subprocess
import tempfile

import h5py
import numpy as np


def execute(command, directory, env):
    directory.mkdir()
    with (directory / "run.log").open("w") as log:
        subprocess.run(list(map(str, command)), cwd=directory, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    text = (directory / "run.log").read_text()
    if "Driver completed." not in text:
        raise RuntimeError(f"Incomplete run: {directory}")
    return text


def snapshot(directory):
    path, = directory.glob("*.state.final.phdf")
    with h5py.File(path) as data:
        state = np.asarray(data["nr.z4c"])
        if not np.isfinite(state).all():
            raise RuntimeError(f"Nonfinite state: {path}")
        return dict(blocks=int(data["Info"].attrs["NumMeshBlocks"]),
                    levels=np.asarray(data["Levels"]),
                    locations=np.asarray(data["LogicalLocations"]), state=state)


def restart(directory):
    path, = directory.glob("*.restart.final.rhdf")
    return path


def wave_case(binary, launcher, directory, env, template):
    # Nine roots yield 72 fine blocks. With two ranks the 36-block split cuts
    # a sibling family, so coarsening/refining exercises real cross-rank AMR
    # transfers, not merely two independent local remeshes.
    common = ["parthenon/mesh/nx1=12", "parthenon/mesh/nx2=4", "parthenon/mesh/nx3=12",
              "parthenon/mesh/derefine_count=2", "parthenon/time/tlim=1",
              "parthenon/time/ncycle_out=1", "parthenon/time/ncycle_out_mesh=1",
              "numerical_relativity/amr_method=normalized_dchi",
              "numerical_relativity/refine_tolerance=0",
              "numerical_relativity/derefine_tolerance=-1",
              "parthenon/output1/id=state", "parthenon/output1/dt=1e9",
              "parthenon/output2/file_type=rst", "parthenon/output2/id=restart",
              "parthenon/output2/dt=1e9"]
    stages = []
    previous = None
    for name, nlim, tolerances in [
            ("fine", 2, []),
            ("coarsen", 4, ["numerical_relativity/refine_tolerance=1e30",
                             "numerical_relativity/derefine_tolerance=1e29"]),
            ("refine", 6, [])]:
        target = directory / name
        command = [*launcher, binary]
        command += ["-i", template] if previous is None else ["-r", restart(previous)]
        text = execute([*command, *common, f"parthenon/time/nlim={nlim}", *tolerances],
                       target, env)
        if name == "coarsen":
            history = [int(n) for n in re.findall(
                r"Total number of MeshBlocks = (\d+)", text)]
            # Exactly one AMR tag per step: threshold two must not coarsen
            # after the first request, or wait until a third completed step.
            if history != [72, 72, 9]:
                raise RuntimeError(f"AMR tagging/hysteresis cadence differs: {history}")
        stages.append(snapshot(target))
        previous = target
    counts = [s["blocks"] for s in stages]
    if counts != [72, 9, 72]:
        raise RuntimeError(f"Missing restriction/prolongation: {directory}: {counts}")
    return stages


def compare(left, right):
    differences = []
    for a, b in zip(left, right):
        if a["blocks"] != b["blocks"] or not np.array_equal(a["levels"], b["levels"]) or \
                not np.array_equal(a["locations"], b["locations"]):
            raise RuntimeError("AMR topology differs")
        error = float(np.max(np.abs(a["state"] - b["state"])))
        differences.append(error)
        if not np.allclose(a["state"], b["state"], rtol=2e-13, atol=2e-13):
            raise RuntimeError(f"AMR state mismatch: Linf={error}")
    return differences


def tracker_command(binary, template):
    return [binary, "-i", template,
               "parthenon/mesh/refinement=adaptive", "parthenon/mesh/numlevel=3",
               "parthenon/mesh/nx1=16", "parthenon/mesh/nx2=16", "parthenon/mesh/nx3=16",
               "parthenon/meshblock/nx1=8", "parthenon/meshblock/nx2=8",
               "parthenon/meshblock/nx3=8", "parthenon/time/integrator=rk4",
               "parthenon/time/tlim=0.05", "parthenon/time/nlim=1",
               "parthenon/time/ncycle_out=1", "numerical_relativity/tracker_mode=ode",
               "numerical_relativity/amr_method=tracker",
               "numerical_relativity/tracker_0_refinement_radius=4",
               "numerical_relativity/tracker_0_refinement_level=2",
               "numerical_relativity/chi_div_floor=1e-5"]


def tracker_case(binary, directory, env, template):
    text = execute(tracker_command(binary, template), directory, env)
    counts = [int(n) for n in re.findall(r"Total number of MeshBlocks = (\d+)", text)]
    if len(counts) < 2 or counts[-1] <= counts[0]:
        raise RuntimeError(f"Tracker AMR used stale center on first step: {counts}")
    if "cycle=1 " not in text or "Puncture tracker mode = ode" not in text:
        raise RuntimeError("Missing tracker mode/step evidence")
    return dict(initial_blocks=counts[0], first_completed_step_blocks=counts[-1])


def stationary_tangency_case(binary, directory, env, template):
    directory.mkdir()
    cases = [("origin", None, 0.)]
    cases += [(f"axis-{axis}-{sign:+d}", axis, sign * 2.3e-16)
              for axis in range(3) for sign in (-1, 1)]
    reference = None
    records = []
    for name, axis, offset in cases:
        overrides = ["parthenon/time/tlim=0", "parthenon/time/nlim=0",
                     "problem/punc_velocity_x1=0", "numerical_relativity/horizon_enabled=false"]
        if axis is not None:
            overrides += [f"numerical_relativity/tracker_0_center_x{axis+1}={offset:.17g}"]
        target = directory / name
        execute([*tracker_command(binary, template), *overrides,
                 "parthenon/output2/id=state"], target, env)
        state = snapshot(target)
        if state["blocks"] != 120:
            raise RuntimeError(f"Roundoff-sized stationary offset changes mesh: {name}: "
                               f"{state['blocks']}")
        if reference is None:
            reference = state
        error, = compare([reference], [state])
        if error != 0.:
            raise RuntimeError(f"Stationary initial state changes with tracker jitter: {name}")
        records.append(dict(case=name, offset=offset, blocks=state["blocks"], state_linf=error))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pangu", required=True, type=Path)
    parser.add_argument("--baseline", type=Path,
                        help="Optional frozen pre-optimization binary; compared without tracker")
    parser.add_argument("--mpiexec", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    case = Path(tempfile.mkdtemp(prefix="nr-amr-transfer-", dir=args.workdir)).resolve()
    source = Path(__file__).resolve().parents[1]
    wave = source / "input/nr4_amr_gauge_wave_crossing.in"
    tracker = source / "input/nr5_boosted_puncture.in"
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("AWA_"):
            env.pop(key)
    # Exercise CUDA-aware MPI, including remote AMR messages on the shared GPU.
    env.update(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,smcuda,tcp",
               OMPI_MCA_io="ompio", OMPI_MCA_opal_cuda_support="true", OMP_NUM_THREADS="1")
    binary = args.pangu.resolve()
    one = case / "one-rank"
    one.mkdir()
    single = wave_case(binary, [], one, env, wave)
    two = case / "two-rank"
    two.mkdir()
    paired = wave_case(binary, [args.mpiexec.resolve(), "--oversubscribe", "-n", "2"],
                       two, env, wave)
    result = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  block_counts=[s["blocks"] for s in single],
                  one_vs_two_rank_linf=compare(single, paired),
                  tracker=tracker_case(binary, case / "tracker", env, tracker),
                  stationary_tangency=stationary_tangency_case(
                      binary, case / "stationary-tangency", env, tracker))
    if args.baseline:
        reference = case / "baseline"
        reference.mkdir()
        old = wave_case(args.baseline.resolve(), [], reference, env, wave)
        result["baseline_vs_optimized_linf"] = compare(old, single)
        result["baseline_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    (case / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print("NR AMR local/remote transfer and tracker-order PASS", case, result)


if __name__ == "__main__":
    main()
