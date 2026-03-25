#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


def replace_or_die(text: str, pattern: str, repl: str, min_count: int = 1) -> str:
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n < min_count:
        raise RuntimeError(f"Pattern not found enough times: {pattern}")
    return new_text


def replace_nth_or_die(text: str, pattern: str, repl: str, nth: int) -> str:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if nth < 0 or nth >= len(matches):
        raise RuntimeError(f"Pattern occurrence {nth} not found: {pattern}")
    m = matches[nth]
    return text[: m.start()] + re.sub(pattern, repl, m.group(0), flags=re.MULTILINE) + text[m.end() :]


def prepare_athena_input(
    template: str,
    nx1: int,
    tlim: str,
    cfl: float,
    gamma: float,
    bx: float,
    output_dt: float,
) -> str:
    text = template
    block_nx1 = 32
    text = replace_or_die(
        text,
        r"(^nx1\s*=\s*)\d+(\s*# Number of zones in X1-direction.*$)",
        rf"\g<1>{nx1}\g<2>",
    )
    text = replace_or_die(
        text,
        r"(^nx1\s*=\s*)\d+(\s*# Number of cells in each MeshBlock, X1-dir.*$)",
        rf"\g<1>{block_nx1}\g<2>",
    )
    if "<coord>" not in text:
        text = text + "\n<coord>\nspecial_rel = true\n"
    else:
        if re.search(r"^special_rel\s*=", text, flags=re.MULTILINE):
            text = replace_or_die(text, r"(^special_rel\s*=\s*).*$", r"\g<1>true")
        else:
            text = text.replace("<coord>", "<coord>\nspecial_rel = true")

    text = replace_or_die(text, r"(^integrator\s*=\s*).*$", r"\g<1>rk2")
    text = replace_or_die(text, r"(^tlim\s*=\s*).*$", rf"\g<1>{tlim}")
    text = replace_or_die(text, r"(^cfl_number\s*=\s*).*$", rf"\g<1>{cfl}")
    text = replace_or_die(text, r"(^reconstruct\s*=\s*).*$", r"\g<1>plm")
    text = replace_or_die(text, r"(^rsolver\s*=\s*).*$", r"\g<1>llf")
    text = replace_or_die(text, r"(^gamma\s*=\s*).*$", rf"\g<1>{gamma}")
    text = replace_or_die(text, r"(^bxl\s*=\s*).*$", rf"\g<1>{bx}")
    text = replace_or_die(text, r"(^bxr\s*=\s*).*$", rf"\g<1>{bx}")
    text = replace_or_die(text, r"(^dt\s*=\s*).*$", rf"\g<1>{output_dt}", min_count=1)
    return text


def prepare_pangu_input(
    template: str,
    nx1: int,
    tlim: str,
    cfl: float,
    gamma: float,
    output_dt: float,
) -> str:
    text = template
    text = replace_nth_or_die(text, r"^nx1\s*=\s*\d+", f"nx1 = {nx1}", nth=0)
    text = replace_nth_or_die(text, r"^nx1\s*=\s*\d+", "nx1 = 32", nth=1)
    text = replace_or_die(text, r"(^tlim\s*=\s*).*$", rf"\g<1>{tlim}")
    text = replace_or_die(text, r"(^CFLNumber\s*=\s*).*$", rf"\g<1>{cfl}")
    text = replace_or_die(text, r"(^AdiabaticIndex\s*=\s*).*$", rf"\g<1>{gamma}")
    text = replace_or_die(text, r"(^dt\s*=\s*).*$", rf"\g<1>{output_dt}", min_count=1)
    text = replace_or_die(
        text,
        r"(^variables\s*=\s*).*$",
        r"\g<1>Density,Energy,WeightedVelocity,MagneticField",
    )
    return text


def run_case(exe: Path, input_file: Path, run_dir: Path, tag: str) -> float:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    cmd = [str(exe), "-i", str(input_file)]
    t0 = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=run_dir, stdout=f, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = ""
        try:
            lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            tail = "\n".join(lines[-80:])
        except Exception:
            tail = "<failed to read log>"
        raise RuntimeError(f"Run failed ({proc.returncode}) for {tag}\n{tail}")
    return dt


def load_athena_profile(tab_w: Path, tab_b: Path, gamma: float):
    w = np.loadtxt(tab_w, comments="#")
    b = np.loadtxt(tab_b, comments="#")
    x = w[:, 2]
    density = w[:, 3]
    vx = w[:, 4]
    pressure = (gamma - 1.0) * w[:, 7]
    by = b[:, 4]
    order = np.argsort(x)
    return {
        "x": x[order],
        "density": density[order],
        "vx": vx[order],
        "pressure": pressure[order],
        "by": by[order],
    }


def load_pangu_profile(phdf: Path, gamma: float):
    with h5py.File(phdf, "r") as h:
        x = h["VolumeLocations"]["x"][...].reshape(-1)
        density = h["Density"][...].reshape(-1)
        energy = h["Energy"][...].reshape(-1)
        wv = h["WeightedVelocity"][...]
        mf = h["MagneticField"][...]
        ux = wv[:, 0, :, :, :].reshape(-1)
        by = mf[:, 1, :, :, :].reshape(-1)
    vx = ux / np.sqrt(1.0 + ux * ux)
    pressure = (gamma - 1.0) * energy
    order = np.argsort(x)
    return {
        "x": x[order],
        "density": density[order],
        "vx": vx[order],
        "ux_weighted": ux[order],
        "pressure": pressure[order],
        "by": by[order],
    }


def error_metrics(ref_x, ref_y, test_x, test_y):
    test_on_ref = np.interp(ref_x, test_x, test_y)
    d = test_on_ref - ref_y
    abs_d = np.abs(d)
    l1 = float(abs_d.mean())
    l2 = float(np.sqrt((d * d).mean()))
    linf = float(abs_d.max())
    rel_l1 = float(l1 / max(float(np.mean(np.abs(ref_y))), 1.0e-14))
    return {
        "l1": l1,
        "l2": l2,
        "linf": linf,
        "rel_l1": rel_l1,
    }


def latest_file(pattern: str) -> Path:
    files = sorted(Path().glob(pattern))
    if not files:
        raise RuntimeError(f"No files match: {pattern}")
    return files[-1]


def median_std(vals):
    med = float(statistics.median(vals))
    std = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return med, std


def main():
    parser = argparse.ArgumentParser(description="Brio-Wu SRMHD benchmark for AthenaK vs Pangu")
    parser.add_argument("--root", default=".", help="Repo root")
    parser.add_argument(
        "--athena-template",
        default="athenak/inputs/mhd/bw.athinput",
        help="Athena template path relative to root",
    )
    parser.add_argument(
        "--pangu-template",
        default="pangu/problem/BrioWuShocktube/inputfile",
        help="Pangu template path relative to root",
    )
    parser.add_argument("--athena-exe", default="athenak/build/src/athena")
    parser.add_argument("--pangu-exe", default="build_srmhd/pangu/src/pangu.cuda")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--tlim", default="1")
    parser.add_argument("--output-dt", type=float, default=0.01)
    parser.add_argument("--bx", type=float, default=0.5, help="Align Bx in Athena input to Pangu")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ath_exe = root / args.athena_exe
    pan_exe = root / args.pangu_exe
    if not ath_exe.exists():
        raise RuntimeError(f"Athena executable not found: {ath_exe}")
    if not pan_exe.exists():
        raise RuntimeError(f"Pangu executable not found: {pan_exe}")

    ath_template = (root / args.athena_template).read_text(encoding="utf-8")
    pan_template = (root / args.pangu_template).read_text(encoding="utf-8")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = root / "data" / f"brio_benchmark_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": ts,
        "resolutions": args.resolutions,
        "repeats": args.repeats,
        "gamma": args.gamma,
        "cfl": args.cfl,
        "tlim": args.tlim,
        "output_dt": args.output_dt,
        "athena_template": args.athena_template,
        "pangu_template": args.pangu_template,
        "cases": {},
    }

    for nx in args.resolutions:
        case_dir = out_root / f"nx{nx}"
        case_dir.mkdir(parents=True, exist_ok=True)
        ath_input = case_dir / "athena.in"
        pan_input = case_dir / "pangu.in"

        ath_input.write_text(
            prepare_athena_input(
                ath_template, nx, args.tlim, args.cfl, args.gamma, args.bx, args.output_dt
            ),
            encoding="utf-8",
        )
        pan_input.write_text(
            prepare_pangu_input(
                pan_template, nx, args.tlim, args.cfl, args.gamma, args.output_dt
            ),
            encoding="utf-8",
        )

        ath_times = []
        pan_times = []
        for rep in range(1, args.repeats + 1):
            ath_times.append(run_case(ath_exe, ath_input, case_dir / "athena" / f"run{rep}", "athena"))
            pan_times.append(run_case(pan_exe, pan_input, case_dir / "pangu" / f"run{rep}", "pangu"))

        ath_med, ath_std = median_std(ath_times)
        pan_med, pan_std = median_std(pan_times)

        ath_w = latest_file(str((case_dir / "athena" / f"run{args.repeats}" / "tab" / "*.mhd_w.*.tab").relative_to(root)))
        ath_b = latest_file(str((case_dir / "athena" / f"run{args.repeats}" / "tab" / "*.mhd_bcc.*.tab").relative_to(root)))

        pan_final = case_dir / "pangu" / f"run{args.repeats}" / "BrioWuShocktube.out0.final.phdf"
        if not pan_final.exists():
            pan_final = latest_file(str((case_dir / "pangu" / f"run{args.repeats}" / "*.phdf").relative_to(root)))

        ath = load_athena_profile(root / ath_w, root / ath_b, args.gamma)
        pan = load_pangu_profile(pan_final, args.gamma)

        errs = {}
        for v in ["density", "vx", "ux_weighted", "pressure", "by"]:
            ref_name = "vx" if v == "ux_weighted" else v
            errs[v] = error_metrics(ath["x"], ath[ref_name], pan["x"], pan[v])

        results["cases"][f"nx{nx}"] = {
            "athena_times": ath_times,
            "pangu_times": pan_times,
            "athena_median": ath_med,
            "athena_std": ath_std,
            "pangu_median": pan_med,
            "pangu_std": pan_std,
            "speed_ratio_pangu_over_athena": pan_med / ath_med if ath_med > 0 else math.nan,
            "errors": errs,
        }

    json_path = out_root / "summary.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = [
        "# Brio-Wu SRMHD Benchmark Summary",
        "",
        f"- Output dir: {out_root}",
        f"- Resolutions: {args.resolutions}",
        f"- Repeats: {args.repeats}",
        "",
        "## Speed",
        "",
        "| nx | Athena median (s) | Pangu median (s) | Pangu/Athena |",
        "|---:|---:|---:|---:|",
    ]

    for nx in args.resolutions:
        c = results["cases"][f"nx{nx}"]
        md_lines.append(
            f"| {nx} | {c['athena_median']:.4f} | {c['pangu_median']:.4f} | {c['speed_ratio_pangu_over_athena']:.3f} |"
        )

    md_lines += ["", "## Errors (Pangu vs Athena)", ""]
    for nx in args.resolutions:
        c = results["cases"][f"nx{nx}"]
        md_lines.append(f"### nx={nx}")
        md_lines.append("")
        md_lines.append("| Variable | L1 | L2 | Linf | Rel L1 |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for v in ["density", "vx", "ux_weighted", "pressure", "by"]:
            e = c["errors"][v]
            md_lines.append(
                f"| {v} | {e['l1']:.6e} | {e['l2']:.6e} | {e['linf']:.6e} | {e['rel_l1']:.6e} |"
            )
        md_lines.append("")

    md_path = out_root / "summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"DONE: {out_root}")
    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")


if __name__ == "__main__":
    main()
