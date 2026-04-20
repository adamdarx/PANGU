#!/usr/bin/env python3
import argparse
import os
from multiprocessing import Pool

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate x-z density-style frames from Parthenon PHDF files.")
    parser.add_argument("field", help="Dataset name to plot, e.g. density")
    parser.add_argument("files", nargs="+", help="Input PHDF files")
    parser.add_argument("--output-directory", required=True, help="Directory to write PNG frames")
    parser.add_argument("--prefix", default="xzplot", help="Output filename prefix")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes")
    parser.add_argument("--kerr-a", type=float, default=0.9375, help="Kerr spin a for horizon marker")
    parser.add_argument("--kerr-h", type=float, default=0.0, help="Modified polar mapping parameter h")
    parser.add_argument("--r0", type=float, default=0.0, help="Radial offset in r=exp(x1)+r0")
    parser.add_argument("--x-max", type=float, default=60.0, help="Maximum display radius in r_g; <=0 means auto")
    parser.add_argument("--level-min", type=float, default=-7.0, help="Lower contour level in log10")
    parser.add_argument("--level-max", type=float, default=0.0, help="Upper contour level in log10")
    parser.add_argument("--level-count", type=int, default=500, help="Number of contour levels")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    return parser.parse_args()


def _load_global_maps(phdf_file, field_name):
    with h5py.File(phdf_file, "r") as h:
        if field_name not in h:
            raise KeyError(f"Field '{field_name}' not found in {phdf_file}")

        data = h[field_name][:]  # (nb, nz, ny, nx)
        if data.ndim != 4:
            raise ValueError(f"Expected 4D dataset (nb,nz,ny,nx), got {data.shape}")

        rho_blocks = data.mean(axis=1)  # average over phi-like direction -> (nb, ny, nx)

        x1_blocks = h["VolumeLocations"]["x"][:]  # (nb, nx)
        x2_blocks = h["VolumeLocations"]["y"][:]  # (nb, ny)

        if "LogicalLocations" in h:
            ll = h["LogicalLocations"][:]
            bx = ll[:, 0].astype(int)
            by = ll[:, 1].astype(int)
        else:
            ll = h["Blocks"]["loc.lx123"][:]
            bx = ll[:, 0].astype(int)
            by = ll[:, 1].astype(int)

        nb, nyb, nxb = rho_blocks.shape
        nx_blocks = int(bx.max()) + 1
        ny_blocks = int(by.max()) + 1

        nx = nx_blocks * nxb
        ny = ny_blocks * nyb

        q = np.zeros((ny, nx), dtype=np.float64)
        x1 = np.zeros(nx, dtype=np.float64)
        x2 = np.zeros(ny, dtype=np.float64)

        for b in range(nb):
            i0 = bx[b] * nxb
            j0 = by[b] * nyb
            q[j0:j0 + nyb, i0:i0 + nxb] = rho_blocks[b]
            x1[i0:i0 + nxb] = x1_blocks[b]
            x2[j0:j0 + nyb] = x2_blocks[b]

        ix = np.argsort(x1)
        iy = np.argsort(x2)
        x1 = x1[ix]
        x2 = x2[iy]
        q = q[np.ix_(iy, ix)]

        info = h.get("Info", None)
        t = float(info.attrs.get("Time", 0.0)) if info is not None else 0.0

    return x1, x2, q, t


def _make_frame(task):
    file_index, file_path, args_dict = task
    field_name = args_dict["field"]

    x1, x2, q, sim_time = _load_global_maps(file_path, field_name)

    x1g, x2g = np.meshgrid(x1, x2, indexing="xy")
    r = np.exp(x1g) + args_dict["r0"]
    theta = 0.5 * np.pi * (x2g + 1.0) + 0.5 * args_dict["kerr_h"] * np.sin(np.pi * (x2g + 1.0))

    x_plot = r * np.sin(theta)
    z_plot = r * np.cos(theta)

    value = np.log10(np.abs(q) + 1e-30)
    levels = np.linspace(
        args_dict["level_min"],
        args_dict["level_max"],
        args_dict["level_count"],
    )

    x_max = args_dict["x_max"] if args_dict["x_max"] > 0.0 else float(np.nanmax(x_plot))
    r_h = 1.0 + np.sqrt(max(0.0, 1.0 - args_dict["kerr_a"] * args_dict["kerr_a"]))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    contour = ax.contourf(
        x_plot,
        z_plot,
        value,
        levels=levels,
        cmap=args_dict["cmap"],
        extend="both",
    )

    horizon = plt.Circle((0.0, 0.0), r_h, color="black", zorder=5)
    ax.add_patch(horizon)

    ax.set_xlim(0.0, x_max)
    ax.set_ylim(-x_max, x_max)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [r_g]")
    ax.set_ylabel("z [r_g]")
    ax.set_title(f"t = {sim_time:.2e}")

    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"log10({field_name})")

    os.makedirs(args_dict["output_directory"], exist_ok=True)
    out_name = f"{args_dict['prefix']}{file_index:04d}.png"
    out_path = os.path.join(args_dict["output_directory"], out_name)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def main():
    args = parse_args()
    os.makedirs(args.output_directory, exist_ok=True)

    files_sorted = sorted(args.files)
    shared = {
        "field": args.field,
        "output_directory": args.output_directory,
        "prefix": args.prefix,
        "kerr_a": args.kerr_a,
        "kerr_h": args.kerr_h,
        "r0": args.r0,
        "x_max": args.x_max,
        "level_min": args.level_min,
        "level_max": args.level_max,
        "level_count": args.level_count,
        "cmap": args.cmap,
    }

    tasks = [(i, f, shared) for i, f in enumerate(files_sorted)]

    if args.workers <= 1:
        for task in tasks:
            _make_frame(task)
    else:
        with Pool(processes=args.workers) as pool:
            pool.map(_make_frame, tasks)


if __name__ == "__main__":
    main()
