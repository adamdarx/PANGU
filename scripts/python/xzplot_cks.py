#!/usr/bin/env python3
"""Generate CKS x-z slice frames from Parthenon PHDF files with SMR/AMR support.

Each mesh block is plotted individually via pcolormesh at its native resolution.
The slice is taken at the equatorial plane (y ≈ 0), not each block's local mid-y.
Uses raw h5py — no dependency on parthenon_tools.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_block_data(h5f, field_name):
    """Return (q_all, xf_all, zf_all, y_idx_all, levels, time, ncycle).

    q_all      — ndarray [NumBlocks, Nz, Ny, Nx] (interior cells only)
    xf_all     — list of 1-D interior face x-coordinates per block
    zf_all     — list of 1-D interior face z-coordinates per block
    y_idx_all  — list of y-index (in interior coords) closest to y=0 per block,
                 or None if the block does not straddle the equatorial plane.
    levels     — ndarray [NumBlocks] refinement level per block
    """
    info = h5f["Info"]
    n_blocks = info.attrs["NumMeshBlocks"]
    includes_ghost = bool(info.attrs.get("IncludesGhost", 0))
    n_ghost = int(info.attrs.get("NGhost", 0)) if includes_ghost else 0

    time = float(info.attrs.get("Time", 0.0))
    try:
        ncycle = int(info.attrs["NCycle"])
    except KeyError:
        ncycle = -1

    # -- field data ----------------------------------------------------------
    dset = h5f[field_name]
    q_raw = dset[:]  # [NumBlocks, ..., Nz_raw, Ny_raw, Nx_raw]

    if includes_ghost and n_ghost > 0:
        g = n_ghost
        interior_slice = (..., slice(g, -g), slice(g, -g), slice(g, -g))
    else:
        interior_slice = (...,)

    q_raw_shape = q_raw.shape
    if len(q_raw_shape) == 4:
        q_all = q_raw[interior_slice] if includes_ghost and n_ghost > 0 else q_raw
    elif len(q_raw_shape) == 5:
        if includes_ghost and n_ghost > 0:
            g = n_ghost
            q_all = q_raw[:, :, g:-g, g:-g, g:-g]
        else:
            q_all = q_raw
    else:
        raise ValueError(f"Unexpected field shape: {q_raw_shape}")

    # -- refinement levels ---------------------------------------------------
    block_meta = h5f["/Blocks/loc.level-gid-lid-cnghost-gflag"][:]
    levels = block_meta[:, 0].astype(int)

    # -- coordinates ---------------------------------------------------------
    xf_raw = h5f["/Locations/x"][:]  # [Nb, Nxf]
    zf_raw = h5f["/Locations/z"][:]  # [Nb, Nzf]
    yf_raw = h5f["/Locations/y"][:]  # [Nb, Nyf]

    xf_all = []
    zf_all = []
    y_idx_all = []

    for ib in range(n_blocks):
        if includes_ghost and n_ghost > 0:
            g = n_ghost
            xf = xf_raw[ib, g : xf_raw.shape[1] - g]
            zf = zf_raw[ib, g : zf_raw.shape[1] - g]
            yf = yf_raw[ib, g : yf_raw.shape[1] - g]
        else:
            xf = xf_raw[ib]
            zf = zf_raw[ib]
            yf = yf_raw[ib]

        # Find the cell whose face interval straddles y = 0 (equatorial plane).
        # Use face coordinates because y = 0 may lie on a block boundary.
        if yf[0] <= 0.0 <= yf[-1]:
            # yf is monotonically increasing; find j such that yf[j] <= 0 <= yf[j+1]
            j = int(np.searchsorted(yf, 0.0, side="right")) - 1
            y_idx = max(0, min(len(yf) - 2, j))
        else:
            y_idx = None

        xf_all.append(xf)
        zf_all.append(zf)
        y_idx_all.append(y_idx)

    return q_all, xf_all, zf_all, y_idx_all, levels, time, ncycle


# ---------------------------------------------------------------------------
# per-frame worker
# ---------------------------------------------------------------------------

def _make_frame(task):
    file_index, file_path, args_dict = task
    field_name = args_dict["field"]

    with h5py.File(file_path, "r") as h5f:
        q_all, xf_list, zf_list, y_idx_list, levels, sim_time, ncycle = \
            _load_block_data(h5f, field_name)

    n_blocks = q_all.shape[0]

    # -- reduce tensor / vector components -----------------------------------
    tc = args_dict.get("tensor_component")
    vc = args_dict.get("vector_component")
    if tc is not None:
        q_all = q_all[:, tc[0], tc[1]]
    elif vc is not None:
        q_all = q_all[:, vc]
    elif q_all.ndim == 5:
        q_all = np.sqrt(np.sum(q_all[:, :3] ** 2, axis=1))

    # -- equatorial (y ≈ 0) slice per block ----------------------------------
    blocks_q = []
    blocks_xf = []
    blocks_zf = []
    blocks_lv = []

    for ib in range(n_blocks):
        y_idx = y_idx_list[ib]
        if y_idx is None:
            continue  # block does not cover equator → skip
        blocks_q.append(q_all[ib, :, y_idx, :])
        blocks_xf.append(xf_list[ib])
        blocks_zf.append(zf_list[ib])
        blocks_lv.append(int(levels[ib]))

    if not blocks_q:
        raise RuntimeError("No blocks cover the equatorial plane (y=0).")

    # -- colour range ---------------------------------------------------------
    q_flat = np.concatenate([b.ravel() for b in blocks_q])
    q_flat = q_flat[np.isfinite(q_flat)]
    if q_flat.size == 0:
        q_flat = np.array([1e-30])

    use_log = not args_dict.get("linear")
    if use_log:
        log_q = np.log10(np.abs(q_flat) + 1e-30)
    else:
        log_q = q_flat

    vmin = args_dict["level_min"] if args_dict["level_min"] is not None else log_q.min()
    vmax = args_dict["level_max"] if args_dict["level_max"] is not None else log_q.max()

    # -- auto x_max ----------------------------------------------------------
    x_max = args_dict["x_max"]
    if x_max <= 0.0:
        all_x = np.concatenate([xf.ravel() for xf in blocks_xf])
        all_z = np.concatenate([zf.ravel() for zf in blocks_zf])
        x_max = float(max(np.nanmax(np.abs(all_x)), np.nanmax(np.abs(all_z))))
        x_max *= 1.02

    # -- Kerr horizon --------------------------------------------------------
    r_h = 1.0 + np.sqrt(max(0.0, 1.0 - args_dict["kerr_a"] ** 2))

    # -- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar_label = f"log10(|{field_name}|)" if use_log else field_name

    for ib, (xf, zf, qb, lv) in enumerate(
        zip(blocks_xf, blocks_zf, blocks_q, blocks_lv)
    ):
        if use_log:
            qb_plot = np.log10(np.abs(qb) + 1e-30)
        else:
            qb_plot = qb

        # imshow with bicubic interpolation: fast raster render, smooth fill,
        # extent pins it exactly to block faces → seamless between neighbours.
        ax.imshow(
            qb_plot,
            extent=[xf[0], xf[-1], zf[0], zf[-1]],
            origin="lower",
            cmap=args_dict["cmap"],
            norm=norm,
            interpolation="none",
            aspect="auto",
        )

    # -- horizon -------------------------------------------------------------
    horizon = plt.Circle((0.0, 0.0), r_h, color="black", zorder=5)
    ax.add_patch(horizon)

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-x_max, x_max)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [r_g]")
    ax.set_ylabel("z [r_g]")
    ax.set_title(f"t = {sim_time:.2e}  |  ncycle = {ncycle}")

    # -- colourbar -----------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=args_dict["cmap"], norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    # -- save ----------------------------------------------------------------
    os.makedirs(args_dict["output_directory"], exist_ok=True)
    out_name = f"{args_dict['prefix']}{file_index:04d}.png"
    out_path = os.path.join(args_dict["output_directory"], out_name)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close("all")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CKS x-z frames from Parthenon PHDF files (SMR-aware)."
    )
    parser.add_argument("field", help="Dataset name to plot, e.g. density")
    parser.add_argument("files", nargs="+", help="Input PHDF files")
    parser.add_argument(
        "--output-directory", required=True, help="Directory to write PNG frames"
    )
    parser.add_argument(
        "--prefix", default="xzplot_cks", help="Output filename prefix"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Worker processes"
    )
    parser.add_argument(
        "--kerr-a", type=float, default=0.9375,
        help="Kerr spin parameter for horizon marker",
    )
    parser.add_argument(
        "--x-max", type=float, default=50.0,
        help="Maximum display radius; <= 0 means auto",
    )
    parser.add_argument(
        "--level-min", type=float, default=-8.0,
        help="Lower colour bound (log10 by default, raw if --linear)",
    )
    parser.add_argument(
        "--level-max", type=float, default=0.0,
        help="Upper colour bound (log10 by default, raw if --linear)",
    )
    parser.add_argument(
        "--cmap", default="viridis", help="Matplotlib colormap"
    )
    parser.add_argument(
        "--linear", action="store_true",
        help="Use linear colour scale instead of the default log10",
    )
    parser.add_argument(
        "--vector-component", type=int, default=None,
        help="Vector component index (0,1,2) to plot from a vector field",
    )
    parser.add_argument(
        "--tensor-component", type=int, nargs=2, default=None,
        help="Tensor component (i,j) to plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_directory, exist_ok=True)

    files_sorted = sorted(args.files)
    shared = {
        "field": args.field,
        "output_directory": args.output_directory,
        "prefix": args.prefix,
        "kerr_a": args.kerr_a,
        "x_max": args.x_max,
        "level_min": args.level_min,
        "level_max": args.level_max,
        "cmap": args.cmap,
        "linear": args.linear,
        "vector_component": args.vector_component,
        "tensor_component": args.tensor_component,
    }

    tasks = [(i, f, shared) for i, f in enumerate(files_sorted)]

    if args.workers <= 1:
        for task in tasks:
            out = _make_frame(task)
            print(out)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_make_frame, t) for t in tasks]
            wait(futures, return_when=ALL_COMPLETED)
            for future in futures:
                exc = future.exception()
                if exc:
                    raise exc
                print(future.result())


if __name__ == "__main__":
    main()
