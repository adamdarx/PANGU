#!/usr/bin/env python3
import argparse
import os
from multiprocessing import Pool

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Pangu/KHARMA electron-heating temperatures are dimensionless:
#   Theta = k_B T / (m_p c^2)
# so T[K] = Theta * m_p c^2 / k_B.
PROTON_MASS_CGS = 1.67262171e-24
LIGHT_SPEED_CGS = 2.99792458e10
BOLTZMANN_CGS = 1.380649e-16
TEMPERATURE_UNIT_K = PROTON_MASS_CGS * LIGHT_SPEED_CGS**2 / BOLTZMANN_CGS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate x-z ion and electron temperature frames from Parthenon PHDF files."
    )
    parser.add_argument("files", nargs="+", help="Input PHDF files")
    parser.add_argument("--output-directory", required=True, help="Directory to write PNG frames")
    parser.add_argument("--prefix", default="xztemp", help="Output filename prefix")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes")
    parser.add_argument("--kerr-a", type=float, default=0.9375, help="Kerr spin a for horizon marker")
    parser.add_argument("--kerr-h", type=float, default=0.0, help="Modified polar mapping parameter h")
    parser.add_argument("--r0", type=float, default=0.0, help="Radial offset in r=exp(x1)+r0")
    parser.add_argument("--x-max", type=float, default=60.0, help="Maximum display radius in r_g; <=0 means auto")
    parser.add_argument("--level-min", type=float, default=10.0, help="Lower contour level in log10(T/K)")
    parser.add_argument("--level-max", type=float, default=12.0, help="Upper contour level in log10(T/K)")
    parser.add_argument("--level-count", type=int, default=500, help="Number of contour levels")
    parser.add_argument("--cmap", default="plasma", help="Matplotlib colormap")
    parser.add_argument(
        "--temperature-unit-k",
        type=float,
        default=TEMPERATURE_UNIT_K,
        help="Conversion factor from code temperature Theta to Kelvin",
    )
    return parser.parse_args()


def _scalar_attr(attrs, key, default):
    value = attrs.get(key, default)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    array_value = np.asarray(value)
    if array_value.shape == ():
        return array_value.item()
    return value


def _load_global_maps(phdf_file):
    with h5py.File(phdf_file, "r") as h:
        for field_name in ("density", "entropy", "electron_entropy"):
            if field_name not in h:
                raise KeyError(f"Field '{field_name}' not found in {phdf_file}")

        rho_blocks = h["density"][:].mean(axis=1)
        entropy_blocks = h["entropy"][:].mean(axis=1)
        electron_entropy_blocks = h["electron_entropy"][:].mean(axis=1)

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
            rho[j0 : j0 + nyb, i0 : i0 + nxb] = rho_blocks[block_index]
            entropy[j0 : j0 + nyb, i0 : i0 + nxb] = entropy_blocks[block_index]
            electron_entropy[j0 : j0 + nyb, i0 : i0 + nxb] = electron_entropy_blocks[block_index]
            x1[i0 : i0 + nxb] = x1_blocks[block_index]
            x2[j0 : j0 + nyb] = x2_blocks[block_index]

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
        sim_time = float(_scalar_attr(h["Info"].attrs if "Info" in h else {}, "Time", 0.0))

    rho_safe = np.maximum(rho, 1.0e-30)
    internal_energy = entropy * np.power(rho_safe, gamma) / (gamma - 1.0)
    ion_temp_code = np.maximum((gamma_p - 1.0) * internal_energy / rho_safe, 0.0)
    electron_temp_code = np.maximum(electron_entropy * np.power(rho_safe, gamma_e - 1.0), 0.0)

    return x1, x2, ion_temp_code, electron_temp_code, sim_time, gamma, gamma_p, gamma_e


def _map_to_xz(x1, x2, q, args_dict):
    x1g, x2g = np.meshgrid(x1, x2, indexing="xy")
    r = np.exp(x1g) + args_dict["r0"]
    theta = 0.5 * np.pi * (x2g + 1.0) + 0.5 * args_dict["kerr_h"] * np.sin(np.pi * (x2g + 1.0))
    x_plot = r * np.sin(theta)
    z_plot = r * np.cos(theta)
    return x_plot, z_plot, q


def _make_frame(task):
    file_index, file_path, args_dict = task
    x1, x2, ion_temp_code, electron_temp_code, sim_time, gamma, gamma_p, gamma_e = _load_global_maps(file_path)

    x_plot, z_plot, ion_temp_code = _map_to_xz(x1, x2, ion_temp_code, args_dict)
    _, _, electron_temp_code = _map_to_xz(x1, x2, electron_temp_code, args_dict)
    temp_unit_k_use = args_dict["temperature_unit_k"]

    ion_temp_k = np.maximum(ion_temp_code * temp_unit_k_use, 1.0e-300)
    electron_temp_k = np.maximum(electron_temp_code * temp_unit_k_use, 1.0e-300)

    ion_value = np.log10(ion_temp_k)
    electron_value = np.log10(electron_temp_k)

    # Auto-select contour log10 levels when user didn't provide them
    lm = args_dict.get("level_min")
    lM = args_dict.get("level_max")
    combined_min = float(min(np.nanmin(ion_value), np.nanmin(electron_value)))
    combined_max = float(max(np.nanmax(ion_value), np.nanmax(electron_value)))
    if lm is None:
        # pad by 0.2 dex or 5% of dynamic range
        pad = max(0.2, 0.05 * (combined_max - combined_min))
        lm = combined_min - pad
    if lM is None:
        pad = max(0.2, 0.05 * (combined_max - combined_min))
        lM = combined_max + pad
    # If user provided levels that are outside data range, still use them; else use auto
    levels = np.linspace(lm, lM, args_dict["level_count"])

    x_max = args_dict["x_max"] if args_dict["x_max"] > 0.0 else float(np.nanmax(x_plot))
    r_h = 1.0 + np.sqrt(max(0.0, 1.0 - args_dict["kerr_a"] * args_dict["kerr_a"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 8), dpi=150, sharex=True, sharey=True)
    plot_specs = [
        (axes[0], ion_value, f"Ion temperature\n$\\gamma_p={gamma_p:.3f}$"),
        (axes[1], electron_value, f"Electron temperature\n$\\gamma_e={gamma_e:.3f}$"),
    ]

    for axis, values, title in plot_specs:
        contour = axis.contourf(
            x_plot,
            z_plot,
            values,
            levels=levels,
            cmap=args_dict["cmap"],
            extend="both",
        )
        horizon = plt.Circle((0.0, 0.0), r_h, color="black", zorder=5)
        axis.add_patch(horizon)
        axis.set_xlim(0.0, x_max)
        axis.set_ylim(-x_max, x_max)
        axis.set_aspect("equal", "box")
        axis.set_xlabel("x [r_g]")
        axis.set_title(title)
        cbar = fig.colorbar(contour, ax=axis, fraction=0.046, pad=0.04)
        cbar.set_label("log10(T [K])")

    axes[0].set_ylabel("z [r_g]")
    fig.suptitle(f"t = {sim_time:.2e}")
    os.makedirs(args_dict["output_directory"], exist_ok=True)
    out_name = f"{args_dict['prefix']}{file_index:04d}.png"
    out_path = os.path.join(args_dict["output_directory"], out_name)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


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
        "x_max": args.x_max,
        "level_min": args.level_min,
        "level_max": args.level_max,
        "level_count": args.level_count,
        "cmap": args.cmap,
        "temperature_unit_k": args.temperature_unit_k,
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
