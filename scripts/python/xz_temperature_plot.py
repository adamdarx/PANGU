#!/usr/bin/env python3
"""Generate CKS x-z ion/electron temperature frames from Parthenon PHDF files.

Reads density, entropy, and electron_entropy, computes ion and electron
temperatures, and renders dual-panel pcolormesh slices with optional GR overlays.

Style: mimics athenak/vis/python/plot_slice.py CLI and rendering conventions.
"""

import argparse
import os
from multiprocessing import Pool

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# -- shared CKS plotting utilities --------------------------------------------
from _cks_common import (
    add_common_args,
    cell_centers_to_faces,
    cks_to_physical,
    draw_ergosphere_xz,
    draw_horizon,
    get_plot_bounds,
    label_for,
    setup_plot_env,
)

# Pangu/KHARMA electron-heating temperatures are dimensionless:
#   Theta = k_B T / (m_p c^2)
# so T[K] = Theta * m_p c^2 / k_B.
PROTON_MASS_CGS = 1.67262171e-24
LIGHT_SPEED_CGS = 2.99792458e10
BOLTZMANN_CGS = 1.380649e-16
TEMPERATURE_UNIT_K = PROTON_MASS_CGS * LIGHT_SPEED_CGS ** 2 / BOLTZMANN_CGS


# ---------------------------------------------------------------------------
# PHDF I/O
# ---------------------------------------------------------------------------

def _scalar_attr(attrs, key, default):
    value = attrs.get(key, default)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def _load_temperature_data(phdf_file):
    """Load density/entropy/electron_entropy and compute temperatures.

    Returns:
        x1, x2: 1D sorted cell-center arrays
        ion_temp_k, electron_temp_k: 2D arrays (ny, nx) in Kelvin
        sim_time, gamma, gamma_p, gamma_e
    """
    with h5py.File(phdf_file, "r") as h:
        for field_name in ("density", "entropy", "electron_entropy"):
            if field_name not in h:
                raise KeyError(f"Field '{field_name}' not found in {phdf_file}")

        rho_blocks = h["density"][:].mean(axis=1)
        entropy_blocks = h["entropy"][:].mean(axis=1)
        ee_blocks = h["electron_entropy"][:].mean(axis=1)

        x1_blocks = h["VolumeLocations"]["x"][:]
        x2_blocks = h["VolumeLocations"]["y"][:]

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

        rho = np.zeros((ny, nx), dtype=np.float64)
        entropy = np.zeros((ny, nx), dtype=np.float64)
        electron_entropy = np.zeros((ny, nx), dtype=np.float64)
        x1 = np.zeros(nx, dtype=np.float64)
        x2 = np.zeros(ny, dtype=np.float64)

        for block_index in range(nb):
            i0 = bx[block_index] * nxb
            j0 = by[block_index] * nyb
            rho[j0:j0 + nyb, i0:i0 + nxb] = rho_blocks[block_index]
            entropy[j0:j0 + nyb, i0:i0 + nxb] = entropy_blocks[block_index]
            electron_entropy[j0:j0 + nyb, i0:i0 + nxb] = ee_blocks[block_index]
            x1[i0:i0 + nxb] = x1_blocks[block_index]
            x2[j0:j0 + nyb] = x2_blocks[block_index]

        ix = np.argsort(x1)
        iy = np.argsort(x2)
        x1 = x1[ix]
        x2 = x2[iy]
        rho = rho[np.ix_(iy, ix)]
        entropy = entropy[np.ix_(iy, ix)]
        electron_entropy = electron_entropy[np.ix_(iy, ix)]

        params = h["Params"].attrs if "Params" in h else {}
        gamma = float(_scalar_attr(params, "core/adiabatic_index", 5.0 / 3.0))
        gamma_e = float(_scalar_attr(params, "core/gamma_e", gamma))
        gamma_p = float(_scalar_attr(params, "core/gamma_p", gamma))
        sim_time = float(_scalar_attr(
            h["Info"].attrs if "Info" in h else {}, "Time", 0.0,
        ))

    # -- temperature calculation ----------------------------------------------
    rho_safe = np.maximum(rho, 1.0e-30)
    internal_energy = entropy * np.power(rho_safe, gamma) / (gamma - 1.0)
    ion_temp_code = np.maximum(
        (gamma_p - 1.0) * internal_energy / rho_safe, 0.0,
    )
    electron_temp_code = np.maximum(
        electron_entropy * np.power(rho_safe, gamma_e - 1.0), 0.0,
    )

    return (
        x1, x2,
        ion_temp_code, electron_temp_code,
        sim_time, gamma, gamma_p, gamma_e,
    )


# ---------------------------------------------------------------------------
# per-frame worker
# ---------------------------------------------------------------------------

def _make_frame(task):
    file_index, file_path, args_dict = task

    x1, x2, ion_code, electron_code, sim_time, gamma, gamma_p, gamma_e = \
        _load_temperature_data(file_path)

    temp_unit_k = args_dict.get("temperature_unit_k", TEMPERATURE_UNIT_K)

    ion_temp_k = np.maximum(ion_code * temp_unit_k, 1.0e-300)
    electron_temp_k = np.maximum(electron_code * temp_unit_k, 1.0e-300)

    # -- face coordinates for pcolormesh ------------------------------------
    x1_f = cell_centers_to_faces(x1)
    x2_f = cell_centers_to_faces(x2)
    X1f, X2f = np.meshgrid(x1_f, x2_f, indexing="xy")

    x_faces, z_faces = cks_to_physical(
        X1f, X2f,
        r0=args_dict.get("r0", 0.0),
        kerr_h=args_dict.get("kerr_h", 0.0),
    )

    # -- value transform ------------------------------------------------------
    use_log = args_dict.get("norm", "log") == "log"
    ion_value = np.log10(ion_temp_k) if use_log else ion_temp_k
    electron_value = np.log10(electron_temp_k) if use_log else electron_temp_k

    # -- colour range (shared across both panels) -----------------------------
    combined = np.concatenate([
        ion_value[np.isfinite(ion_value)].ravel(),
        electron_value[np.isfinite(electron_value)].ravel(),
    ])
    vmin = args_dict.get("vmin")
    vmax = args_dict.get("vmax")
    if vmin is None:
        vmin = float(np.nanmin(combined)) if combined.size > 0 else 0.0
    if vmax is None:
        vmax = float(np.nanmax(combined)) if combined.size > 0 else 1.0
    norm = matplotlib.colors.Normalize(vmin, vmax)

    # -- auto bounds ----------------------------------------------------------
    auto_max = float(max(np.nanmax(np.abs(x_faces)), np.nanmax(np.abs(z_faces)))) * 1.02
    x1_min, x1_max, x2_min, x2_max = get_plot_bounds(
        args_dict,
        -auto_max, auto_max,
        -auto_max, auto_max,
    )

    # -- matplotlib config ----------------------------------------------------
    setup_plot_env(notex=args_dict.get("notex", False), output_file=None)

    # -- figure (dual panel) --------------------------------------------------
    dpi = args_dict.get("dpi", 150)
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), dpi=dpi, sharex=True, sharey=True)

    plot_specs = [
        (axes[0], ion_value, "Ion temperature",
         rf"$\gamma_p={gamma_p:.3f}$"),
        (axes[1], electron_value, "Electron temperature",
         rf"$\gamma_e={gamma_e:.3f}$"),
    ]

    kerr_a = args_dict.get("kerr_a", 0.0)

    for ax, values, title, subtitle in plot_specs:
        mesh = ax.pcolormesh(
            x_faces, z_faces, values,
            cmap=args_dict["cmap"],
            norm=norm,
            shading="flat",
            rasterized=True,
        )

        # -- GR overlays per panel --------------------------------------------
        if kerr_a != 0.0:
            if args_dict.get("horizon_mask", False):
                draw_horizon(
                    ax, kerr_a, mask=True, outline=False,
                    mask_color=args_dict.get("horizon_mask_color", "k"),
                )
            if args_dict.get("horizon", False):
                draw_horizon(
                    ax, kerr_a, mask=False, outline=True,
                    outline_color=args_dict.get("horizon_color", "k"),
                )
            if args_dict.get("ergosphere", False):
                draw_ergosphere_xz(
                    ax, kerr_a,
                    color=args_dict.get("ergosphere_color", "gray"),
                )

        # -- axes -------------------------------------------------------------
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_aspect("equal", "box")

        if args_dict.get("notex", False):
            ax.set_xlabel("x [r_g]")
            ax.set_title(f"{title}\n{subtitle}")
        else:
            ax.set_xlabel(r"$x$ [$r_\mathrm{g}$]")
            ax.set_title(title + "\n" + subtitle)

        # -- colourbar per panel ----------------------------------------------
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        if use_log:
            cbar.set_label(r"$\log_{10}(T\ [\mathrm{K}])$")
        else:
            cbar.set_label(r"$T$ (K)")

    if args_dict.get("notex", False):
        axes[0].set_ylabel("z [r_g]")
        fig.suptitle(f"t = {sim_time:.2e}")
    else:
        axes[0].set_ylabel(r"$z$ [$r_\mathrm{g}$]")
        fig.suptitle(r"$t = {:.2e}$".format(sim_time))

    # -- save -----------------------------------------------------------------
    os.makedirs(args_dict["output_directory"], exist_ok=True)
    out_name = f"{args_dict['prefix']}{file_index:04d}.png"
    out_path = os.path.join(args_dict["output_directory"], out_name)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CKS x-z ion and electron temperature frames "
                    "from Parthenon PHDF files."
    )
    parser.add_argument(
        "files", nargs="+", help="Input PHDF files"
    )
    parser.add_argument(
        "--output-directory", required=True, help="Directory to write PNG frames"
    )
    parser.add_argument(
        "--prefix", default="xztemp", help="Output filename prefix"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Worker processes"
    )

    # CKS coordinate parameters
    parser.add_argument(
        "--kerr-a", type=float, default=0.0,
        help="Kerr spin parameter for horizon marker (0 = no BH)",
    )
    parser.add_argument(
        "--kerr-h", type=float, default=0.0,
        help="Modified polar mapping parameter h",
    )
    parser.add_argument(
        "--r0", type=float, default=0.0,
        help="Radial offset in r = exp(x1) + r0",
    )

    # Temperature specific
    parser.add_argument(
        "--temperature-unit-k",
        type=float,
        default=TEMPERATURE_UNIT_K,
        help="Conversion factor from code temperature Theta to Kelvin",
    )

    # -- common args (plot bounds, colour scale, GR, grid, output) ------------
    add_common_args(parser)

    parser.set_defaults(
        cmap="plasma",
        dpi=150,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_directory, exist_ok=True)

    files_sorted = sorted(args.files)
    shared = {
        "output_directory": args.output_directory,
        "prefix": args.prefix,
        "kerr_a": args.kerr_a,
        "kerr_h": args.kerr_h,
        "r0": args.r0,
        "temperature_unit_k": args.temperature_unit_k,
        "cmap": args.cmap,
        "norm": args.norm,
        "vmin": args.vmin,
        "vmax": args.vmax,
        "r_max": args.r_max,
        "x1_min": args.x1_min,
        "x1_max": args.x1_max,
        "x2_min": args.x2_min,
        "x2_max": args.x2_max,
        "dpi": args.dpi,
        "notex": args.notex,
        "horizon": args.horizon,
        "horizon_mask": args.horizon_mask,
        "horizon_color": args.horizon_color,
        "horizon_mask_color": args.horizon_mask_color,
        "ergosphere": args.ergosphere,
        "ergosphere_color": args.ergosphere_color,
        "grid": args.grid,
        "grid_color": args.grid_color,
        "grid_alpha": args.grid_alpha,
    }

    tasks = [(i, f, shared) for i, f in enumerate(files_sorted)]

    if args.workers <= 1:
        for task in tasks:
            out = _make_frame(task)
            print(out)
    else:
        with Pool(processes=args.workers) as pool:
            for out in pool.imap_unordered(_make_frame, tasks):
                print(out)


if __name__ == "__main__":
    main()
