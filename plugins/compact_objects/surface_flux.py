#!/usr/bin/env python3
"""Sample native Cartesian AMR output and integrate fixed coordinate-sphere fluxes.

The sampler uses each point's finest leaf and tensor-product cubic interpolation.
At a leaf edge it uses interior cell centers, with at most half a cell of
one-sided extrapolation. This is fourth order for smooth fields and does not
require saved ghost cells. Linear interpolation is also available explicitly.
Unphysical interpolated states are errors.
These fixed spheres do not measure flux through a moving apparent horizon.
"""
from pathlib import Path
import argparse
from functools import lru_cache
import hashlib
import json

import h5py
import numpy as np

if __package__:
    from .flux_diagnostics import coordinate_sphere_flux, flux_densities
else:
    from flux_diagnostics import coordinate_sphere_flux, flux_densities


def sphere_quadrature(radius, center=(0., 0., 0.), ntheta=32, nphi=64):
    if not np.isfinite(radius) or radius <= 0 or ntheta < 4 or nphi < 8:
        raise ValueError("positive radius and at least 4 by 8 angular points are required")
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("invalid sphere center")
    mu, w = np.polynomial.legendre.leggauss(ntheta)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    transverse = np.sqrt(1 - mu**2)[:, None]
    normals = np.stack(np.broadcast_arrays(transverse * np.cos(phi),
                                           transverse * np.sin(phi), mu[:, None]), axis=-1)
    weights = np.broadcast_to(w[:, None] * (2 * np.pi / nphi), (ntheta, nphi)).copy()
    return center + radius * normals, normals, weights


class CartesianLeafSampler:
    """A bounded block cache for cell-centered GRMHD and ADM output."""

    def __init__(self, path, degree=3):
        if degree not in (1, 3):
            raise ValueError("interpolation degree must be 1 or 3")
        self.degree = degree
        self.path = Path(path)
        self.file = h5py.File(self.path, "r")
        try:
            info = self.file["Info"].attrs
            coordinates = info["Coordinates"]
            if isinstance(coordinates, bytes):
                coordinates = coordinates.decode()
            if str(coordinates) != "UniformCartesian" or int(info["NumDims"]) != 3:
                raise ValueError("three-dimensional UniformCartesian output is required")
            if int(info["IncludesGhost"]) != 0:
                raise ValueError("save ghost_zones=false for unambiguous leaf ownership")
            self.time = float(info["Time"])
            if not np.isfinite(self.time):
                raise ValueError("nonfinite snapshot time")
            self.levels = np.asarray(self.file["Levels"], dtype=int)
            self.centers = [np.asarray(self.file["VolumeLocations/" + a]) for a in "xyz"]
            faces = [np.asarray(self.file["Locations/" + a]) for a in "xyz"]
            count = int(info["NumMeshBlocks"])
            shape = self.file["mhd.prim"].shape
            if shape[0] != count or len(self.levels) != count or shape[1] < 4:
                raise ValueError("inconsistent block or primitive layout")
            for axis, (x, f) in enumerate(zip(self.centers, faces)):
                if (x.shape != (count, shape[4 - axis]) or x.shape[1] < degree + 1 or
                        f.shape != (count, x.shape[1] + 1) or not np.isfinite(x).all() or
                        not np.isfinite(f).all() or np.any(np.diff(x, axis=1) <= 0) or
                        np.any(np.diff(f, axis=1) <= 0)):
                    raise ValueError("invalid cell coordinates")
                if not np.allclose(x, .5 * (f[:, :-1] + f[:, 1:]), rtol=1e-13, atol=1e-13):
                    raise ValueError("cell centers must match the saved Cartesian faces")
            self.bounds = np.stack([np.stack((f[:, 0], f[:, -1]), axis=1) for f in faces], axis=1)
            self.domain = np.stack((self.bounds[:, :, 0].min(axis=0),
                                    self.bounds[:, :, 1].max(axis=0)), axis=1)
            self.storage_dtypes = {name: str(self.file[name].dtype)
                                   for name in ("mhd.prim", "mhd.b_cell", "nr.adm", "nr.z4c")}
            for name, components in [("mhd.b_cell", 3), ("nr.adm", 6), ("nr.z4c", 22)]:
                s = self.file[name].shape
                if len(s) != 5 or s[0] != count or s[1] < components or s[2:] != shape[2:]:
                    raise ValueError(f"invalid {name} layout")
        except Exception:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._block.cache_clear()
        self.file.close()

    @lru_cache(maxsize=16)
    def _block(self, block):
        return np.concatenate((self.file["mhd.prim"][block, :4],
                               self.file["mhd.b_cell"][block, :3],
                               self.file["nr.adm"][block, :6],
                               self.file["nr.z4c"][block, 18:22]), axis=0)

    def sample(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("surface points must be a finite (n,3) array")
        owner = np.full(len(points), -1, dtype=int)
        tolerance = 32 * np.finfo(float).eps * max(1., float(abs(self.bounds).max()))
        # Highest-level-first assignment also makes shared-boundary ties
        # deterministic. No point is counted twice in the surface integral.
        for block in np.argsort(-self.levels, kind="stable"):
            mask = ((owner < 0) & np.all(points >= self.bounds[block, :, 0] - tolerance, axis=1)
                    & np.all(points <= self.bounds[block, :, 1] + tolerance, axis=1))
            owner[mask] = block
        if np.any(owner < 0):
            raise ValueError(f"{np.count_nonzero(owner < 0)} surface samples lie outside the saved mesh")
        output = np.empty((len(points), 17))
        one_sided = np.zeros(len(points), dtype=bool)
        for block in np.unique(owner):
            selected = np.flatnonzero(owner == block)
            indices, weights = [], []
            for axis in range(3):
                x = self.centers[axis][block]
                query = points[selected, axis]
                i = np.clip(np.searchsorted(x, query) - (self.degree + 1) // 2,
                            0, len(x) - self.degree - 1)
                indices.append(i)
                axis_weights = []
                for j in range(self.degree + 1):
                    w = np.ones(len(selected))
                    for k in range(self.degree + 1):
                        if k != j:
                            w *= (query - x[i + k]) / (x[i + j] - x[i + k])
                    axis_weights.append(w)
                weights.append(axis_weights)
                one_sided[selected] |= (query < x[0]) | (query > x[-1])
            values = self._block(int(block))
            local = np.zeros((len(selected), 17))
            for dx in range(self.degree + 1):
                for dy in range(self.degree + 1):
                    for dz in range(self.degree + 1):
                        weight = np.ones(len(selected))
                        for axis, offset in enumerate((dx, dy, dz)):
                            weight *= weights[axis][offset]
                        local += weight[:, None] * values[:, indices[2] + dz,
                                                          indices[1] + dy, indices[0] + dx].T
            output[selected] = local
        if not np.isfinite(output).all():
            raise ValueError("nonfinite sampled fields")
        gamma = np.empty((len(points), 3, 3))
        for component, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
            gamma[:, i, j] = gamma[:, j, i] = output[:, 7 + component]
        diagnostics = flux_densities(output[:, 0], output[:, 1:4], output[:, 4:7],
                                    gamma, output[:, 13], output[:, 14:17])
        diagnostics["densitized_magnetic_field"] = output[:, 4:7]
        return diagnostics, dict(points=len(points), blocks_used=int(len(np.unique(owner))),
                                 levels_used=sorted(int(x) for x in np.unique(self.levels[owner])),
                                 one_sided_points=int(np.count_nonzero(one_sided)))


def extract(path, radii, center=(0., 0., 0.), ntheta=32, nphi=64, degree=3):
    results = []
    with CartesianLeafSampler(path, degree) as sampler:
        for radius in radii:
            points, normals, weights = sphere_quadrature(radius, center, ntheta, nphi)
            tolerance = 32 * np.finfo(float).eps * max(1., float(abs(sampler.domain).max()))
            if (np.any(np.asarray(center) - radius < sampler.domain[:, 0] - tolerance) or
                    np.any(np.asarray(center) + radius > sampler.domain[:, 1] + tolerance)):
                raise ValueError("the full extraction sphere must lie inside the saved domain")
            values, sampling = sampler.sample(points.reshape(-1, 3))
            normal, weight = normals.reshape(-1, 3), weights.reshape(-1)
            mass = coordinate_sphere_flux(values["mass_flux_density"], normal, weight, radius)
            electromagnetic = coordinate_sphere_flux(values["em_energy_flux_density"], normal, weight, radius)
            magnetic = coordinate_sphere_flux(values["densitized_magnetic_field"], normal, weight, radius)
            results.append(dict(radius=float(radius), mass_flux_outward=mass, mass_inflow_rate=-mass,
                                electromagnetic_luminosity_outward=electromagnetic,
                                net_magnetic_flux=magnetic, sampling=sampling))
        return dict(file=str(Path(path).resolve()), time=sampler.time, center=list(center),
                    angular_points=[ntheta, nphi], spheres=results,
                    storage_dtypes=sampler.storage_dtypes,
                    interpolation=f"leaf-local degree-{degree} tensor product; one-sided at edges; O(h^{degree+1}) for smooth fields",
                    surface="fixed Cartesian coordinate spheres; positive flux is outward",
                    energy="coordinate energy current -sqrt(-g) T^i_t(EM), absorbed-4pi convention")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--radii", nargs="+", required=True, type=float)
    parser.add_argument("--center", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nphi", type=int, default=64)
    parser.add_argument("--degree", type=int, choices=[1, 3], default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = dict(status="completed", tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  snapshots=[extract(p, args.radii, args.center, args.ntheta, args.nphi, args.degree) for p in args.snapshots])
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.output)
    print(json.dumps(report, indent=2))
