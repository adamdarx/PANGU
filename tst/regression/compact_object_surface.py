#!/usr/bin/env python3
"""Validate AMR surface sampling with analytic Cartesian GRMHD currents."""
from pathlib import Path
import argparse
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugins.compact_objects.surface_flux import CartesianLeafSampler, extract, sphere_quadrature


def snapshot(path, n, radial=False):
    # 4^3 root blocks. Refine the central 2^3 blocks, yielding 56+64 leaves.
    block_cells = n // 4
    leaves = []
    for iz in range(4):
        for iy in range(4):
            for ix in range(4):
                low = -1 + .5 * np.array([ix, iy, iz])
                if all(i in (1, 2) for i in (ix, iy, iz)):
                    for dz in (0, 1):
                        for dy in (0, 1):
                            for dx in (0, 1):
                                leaves.append((1, low + .25 * np.array([dx, dy, dz]), .25))
                else:
                    leaves.append((0, low, .5))
    gamma = np.eye(3) if radial else np.array([[2., .1, .02], [.1, 1.5, .05], [.02, .05, .9]])
    lapse = 1. if radial else .8
    shift = np.zeros(3) if radial else np.array([.01, .02, -.03])
    sqrt_gamma = np.sqrt(np.linalg.det(gamma))
    chi = np.linalg.det(gamma)**(-1 / 3)
    with h5py.File(path, "w") as f:
        info = f.create_group("Info")
        for key, value in dict(Coordinates="UniformCartesian", NumDims=3, IncludesGhost=0,
                               Time=0., NumMeshBlocks=len(leaves), MeshBlockSize=[block_cells] * 3).items():
            info.attrs[key] = value
        f["Levels"] = [leaf[0] for leaf in leaves]
        for axis, name in enumerate("xyz"):
            faces = np.array([np.linspace(low[axis], low[axis] + width, block_cells + 1)
                              for _, low, width in leaves])
            f["Locations/" + name] = faces
            f["VolumeLocations/" + name] = .5 * (faces[:, :-1] + faces[:, 1:])
        arrays = {}
        for name, count in [("mhd.prim", 5), ("mhd.b_cell", 3), ("nr.adm", 13), ("nr.z4c", 22)]:
            arrays[name] = f.create_dataset(name, (len(leaves), count, block_cells, block_cells, block_cells),
                                           dtype="f8", chunks=(1, count, block_cells, block_cells, block_cells),
                                           compression="gzip", compression_opts=1)
        for b, _ in enumerate(leaves):
            x, y, z = np.broadcast_arrays(f["VolumeLocations/x"][b][None, None, :],
                                          f["VolumeLocations/y"][b][None, :, None],
                                          f["VolumeLocations/z"][b][:, None, None])
            primitive = np.empty((5,) + x.shape)
            if radial:
                radius = np.sqrt(x*x + y*y + z*z)
                primitive[0] = 1.3
                u = -.2 / np.sqrt(1 - .2**2)
                primitive[1:4] = u * np.array([x, y, z]) / radius
            else:
                primitive[0] = 1 + .1*x + .2*y + .3*z
                primitive[1:4] = np.array([.2, -.1, .05])[:, None, None, None]
            primitive[4] = 1
            arrays["mhd.prim"][b] = primitive
            magnetic = np.zeros((3,) + x.shape)
            magnetic[:] = np.array([0., 0., .4] if radial else [.3, -.2, .1])[:, None, None, None]
            arrays["mhd.b_cell"][b] = magnetic
            adm, nr = np.zeros((13,) + x.shape), np.zeros((22,) + x.shape)
            for c, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
                adm[c] = gamma[i, j]
                nr[1+c] = chi * gamma[i, j]
            nr[0], nr[18] = chi, lapse
            nr[19:22] = shift[:, None, None, None]
            arrays["nr.adm"][b], arrays["nr.z4c"][b] = adm, nr
    return gamma, lapse, shift, sqrt_gamma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "purpose": "Synthetic sampler/formula qualification, not evolved physical data"}

    def save():
        p = root / "summary.tmp"
        p.write_text(json.dumps(report, indent=2) + "\n")
        p.replace(root / "summary.json")

    save()
    try:
        path = root / "linear_curved.phdf"
        gamma, lapse, shift, sqrt_gamma = snapshot(path, 16)
        radius, center = .72, [.09, -.06, .04]
        point, _, _ = sphere_quadrature(radius, center, 16, 32)
        points = point.reshape(-1, 3)
        U, gradient = np.array([.2, -.1, .05]), np.array([.1, .2, .3])
        W = np.sqrt(1 + U @ gamma @ U)
        ui = U - W * shift / lapse
        rho = 1 + points @ gradient
        exact_density = lapse * sqrt_gamma * rho[:, None] * ui
        with CartesianLeafSampler(path) as sampler:
            values, sampling = sampler.sample(points)
            point_error = float(abs(values["mass_flux_density"] - exact_density).max())
            assert point_error < 1e-13, point_error
            assert sampling["one_sided_points"] > 0 and len(sampling["levels_used"]) == 2
            try:
                sampler.sample(np.array([[5., 0., 0.]]))
            except ValueError:
                pass
            else:
                raise AssertionError("outside-domain points were accepted")
        measured = extract(path, [radius], center, 16, 32)["spheres"][0]
        exact_mass = 4 * np.pi / 3 * radius**3 * lapse * sqrt_gamma * (ui @ gradient)
        assert abs(measured["mass_flux_outward"] - exact_mass) < 1e-13
        assert abs(measured["electromagnetic_luminosity_outward"]) < 1e-13
        assert abs(measured["net_magnetic_flux"]) < 1e-13
        try:
            extract(path, [1 + 1e-7], ntheta=4, nphi=8)
        except ValueError:
            pass
        else:
            raise AssertionError("a sphere protruding between angular nodes was accepted")
        report["linear_curved"] = dict(point_mass_density_error=point_error, exact_mass_flux=exact_mass,
                                         measured=measured)
        with h5py.File(path, "r+") as f:
            f["Info"].attrs["IncludesGhost"] = 1
        try:
            with CartesianLeafSampler(path):
                raise AssertionError("overlapping ghost blocks were accepted")
        except ValueError:
            pass
        finally:
            with h5py.File(path, "r+") as f:
                f["Info"].attrs["IncludesGhost"] = 0
        radius, rho, speed, field = .72, 1.3, .2, .4
        exact = np.array([-4*np.pi*radius**2*rho*speed/np.sqrt(1-speed**2),
                          -8*np.pi/3*radius**2*speed*field**2])
        rows, errors = [], []
        for n in (16, 32, 64):
            path = root / f"radial_n{n}.phdf"
            snapshot(path, n, radial=True)
            result = extract(path, [radius], ntheta=256, nphi=512)["spheres"][0]
            measured = np.array([result["mass_flux_outward"], result["electromagnetic_luminosity_outward"]])
            error = abs(measured / exact - 1)
            errors.append(error)
            rows.append(dict(n=n, exact=exact.tolist(), measured=measured.tolist(), relative_error=error.tolist(), sampling=result["sampling"]))
            report["radial_inflow"] = rows
            save()
        orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        assert np.min(orders) > 2.5, orders
        assert np.max(errors[-1]) < 1e-4, errors
        high = extract(path, [radius], ntheta=512, nphi=1024)["spheres"][0]
        high_values = np.array([high["mass_flux_outward"], high["electromagnetic_luminosity_outward"]])
        angular_change = abs(high_values / exact - np.asarray(rows[-1]["measured"]) / exact)
        assert np.all(angular_change < .3 * errors[-1]), (angular_change, errors[-1])
        report.update(status="passed", observed_orders=orders.tolist(), angular_refinement_change=angular_change.tolist())
    except Exception as error:
        report.update(status="failed", error=repr(error))
        save()
        raise
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
