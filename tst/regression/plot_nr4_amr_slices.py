#!/usr/bin/env python3
"""Render native-resolution AMR slices without interpolation or regridding."""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib.patches import Patch, Rectangle


def files_for(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(directory.glob("nr_gauge_wave.nr4_amr_crossing*.phdf"))


def read_frame(path: pathlib.Path, z_plane: float) -> dict:
    with h5py.File(path, "r") as data:
        info = data["Info"].attrs
        levels = np.asarray(data["Levels"])
        locations = {axis: np.asarray(data["Locations"][axis]) for axis in "xyz"}
        centers = {axis: np.asarray(data["VolumeLocations"][axis]) for axis in "xyz"}
        z4c = np.asarray(data["nr.z4c"])
        adm = np.asarray(data["nr.adm"])
        constraints = np.asarray(data["nr.constraints"])
        time = float(info["Time"])
        num_blocks = int(info["NumMeshBlocks"])

    blocks = []
    for block, level in enumerate(levels):
        z_edges = locations["z"][block]
        if z_edges[0] - 1.0e-12 > z_plane or z_edges[-1] + 1.0e-12 < z_plane:
            continue
        k = int(np.argmin(np.abs(centers["z"][block] - z_plane)))
        blocks.append({
            "level": int(level),
            "x": locations["x"][block],
            "y": locations["y"][block],
            "chi": z4c[block, 0, k],
            "gxx": adm[block, 0, k],
            "constraint": constraints[block, 0, k],
        })
    if not blocks:
        raise RuntimeError(f"no AMR block intersects native z-plane {z_plane}: {path}")
    return {"time": time, "blocks": blocks, "num_blocks": num_blocks}


def limits(frames: list[dict]) -> dict[str, tuple[float, float]]:
    values: dict[str, list[np.ndarray]] = {"chi": [], "gxx": [], "constraint": []}
    for frame in frames:
        for block in frame["blocks"]:
            for name in values:
                values[name].append(block[name].ravel())
    out = {}
    for name, arrays in values.items():
        merged = np.concatenate(arrays)
        if name == "constraint":
            positive = merged[merged > 0.0]
            out[name] = (float(np.min(positive)), float(np.max(positive)))
        else:
            out[name] = (float(np.min(merged)), float(np.max(merged)))
    return out


def draw(frame: dict, path: pathlib.Path, norm_limits: dict[str, tuple[float, float]],
         z_plane: float, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), layout="constrained")
    specs = (
        ("chi", r"$\chi$", "viridis", colors.Normalize(*norm_limits["chi"])),
        ("gxx", r"$\gamma_{xx}$", "plasma", colors.Normalize(*norm_limits["gxx"])),
        ("constraint", r"$C_{\rm combined}$", "magma",
         colors.LogNorm(*norm_limits["constraint"])),
    )
    mappables = []
    for ax, (name, label, cmap, norm) in zip(axes, specs):
        for block in frame["blocks"]:
            ax.pcolormesh(block["x"], block["y"], block[name], shading="flat",
                          cmap=cmap, norm=norm, edgecolors="none", antialiased=False)
            level_color = "#f4f4f4" if block["level"] == 0 else "#101010"
            ax.add_patch(Rectangle(
                (block["x"][0], block["y"][0]),
                block["x"][-1] - block["x"][0], block["y"][-1] - block["y"][0],
                fill=False, edgecolor=level_color, linewidth=0.75, zorder=5,
            ))
        mappable = ax.collections[-len(frame["blocks"])]
        mappables.append(mappable)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.046)
    axes[0].legend(handles=[
        Patch(facecolor="none", edgecolor="#f4f4f4", label="level 0"),
        Patch(facecolor="none", edgecolor="#101010", label="level 1"),
    ], loc="upper right", framealpha=0.85, fontsize=9)
    fig.suptitle(
        f"Z4c AMR native slice  |  t={frame['time']:.6f}  |  z={z_plane:.6f} "
        f"(nearest native cell)  |  leaf blocks={frame['num_blocks']}", fontsize=12,
    )
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--z-plane", type=float, default=0.1875)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    inputs = files_for(args.input_dir)
    if not inputs:
        raise RuntimeError(f"no nr4 AMR PHDF files in {args.input_dir}")
    frames = [read_frame(path, args.z_plane) for path in inputs]
    norm_limits = limits(frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.input_dir),
        "frames": [str(path) for path in inputs],
        "z_plane": args.z_plane,
        "slice_policy": "native cell nearest to z_plane; no interpolation or regridding",
        "mesh_encoding": "native pcolormesh patches with level-colored leaf-block outlines",
        "norm_limits": norm_limits,
        "dpi": args.dpi,
    }
    for index, frame in enumerate(frames):
        draw(frame, args.output_dir / f"slice_{index:05d}_t{frame['time']:.6f}.png",
             norm_limits, args.z_plane, args.dpi)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(frames)} native AMR slice figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
