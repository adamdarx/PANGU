#!/usr/bin/env python3
"""Measure a matched, uniform Cartesian TOV trajectory; never infer acceptance.

AthenaK's reader is loaded from the explicitly supplied reference checkout.
Both conserved density outputs already contain sqrt(gamma), so the baryon
integral uses coordinate cell volumes without a second metric factor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reader(source: Path):
    path = source / 'vis/python/bin_convert.py'
    spec = importlib.util.spec_from_file_location('athenak_reference_reader', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_binary


def ordered_density(snapshot, stream):
    """Return AthenaK density in the PANGU block order, checking geometry."""
    locations = np.asarray(stream['LogicalLocations'], dtype=int)
    levels = np.asarray(stream['Levels'], dtype=int)
    if np.any(levels != 0):
        raise ValueError('This pilot comparison requires a uniform root grid')
    target_keys = [(int(level), *map(int, loc))
                   for level, loc in zip(levels, locations, strict=True)]
    source_keys = [(int(row[3]), *map(int, row[:3]))
                   for row in np.asarray(snapshot['mb_logical'])]
    if (len(set(source_keys)) != len(source_keys)
            or set(source_keys) != set(target_keys)):
        raise ValueError('MeshBlock sets differ or contain duplicate blocks')
    index = {key: i for i, key in enumerate(source_keys)}
    order = np.asarray([index[key] for key in target_keys])
    geometry = np.asarray(snapshot['mb_geometry'])[order]
    expected = np.stack([np.asarray(stream['Locations/' + axis])[:, side]
                         for axis in 'xyz' for side in (0, -1)], axis=1)
    if not np.allclose(geometry, expected, rtol=0, atol=1e-12):
        raise ValueError('MeshBlock physical bounds differ')
    return np.asarray(snapshot['mb_data']['dens'], dtype=np.float64)[order]


def binary_variable_bytes(path: Path) -> int:
    with path.open('rb') as stream:
        stream.readline()
        count = int(stream.readline().split(b'=')[-1])
        header = {}
        for _ in range(count - 1):
            key, value = stream.readline().decode().split('=', 1)
            header[key.strip()] = value.strip()
    return int(header['size of variable'])


def native_mass_history(folder: Path, column: str, times: np.ndarray):
    matches = []
    for path in folder.rglob('*.hst'):
        with path.open() as stream:
            headers = ''.join(line for line in stream if line.startswith('#'))
        columns = {name: int(index) - 1
                   for index, name in re.findall(r'\[(\d+)\]=(\S+)', headers)}
        if column in columns:
            matches.append((path, columns))
    if len(matches) != 1:
        raise ValueError(f'Expected one native history with column {column}, found {len(matches)}')
    path, columns = matches[0]
    data = np.atleast_2d(np.loadtxt(path))
    if not np.isfinite(data[:, [columns['time'], columns[column]]]).all():
        raise ValueError('Nonfinite native history mass or time')
    indices = []
    for time in times:
        hits = np.flatnonzero(np.isclose(data[:, columns['time']], time, rtol=0, atol=1e-12))
        if len(hits) != 1:
            raise ValueError(f'Missing or duplicated native history at time {time}')
        indices.append(hits[0])
    return data[indices, columns[column]], dict(path=str(path), column=column, sha256=sha256(path))


def compare(pangu_dir: Path, athena_dir: Path, source: Path,
            target_time: float = 50, expected_samples: int = 11):
    read = reader(source)
    fields = sorted(pangu_dir.glob('*.phdf'))
    conserved = sorted((athena_dir / 'bin').glob('*.mhd_u.*.bin'))
    if len(fields) != expected_samples or len(conserved) != expected_samples:
        raise ValueError(f'Expected {expected_samples} samples per code; '
                         f'found PANGU={len(fields)}, AthenaK={len(conserved)}')
    rows, evidence, precisions = [], [], set()
    core = None
    for ppath, apath in zip(fields, conserved, strict=True):
        wpath = apath.with_name(apath.name.replace('.mhd_u.', '.mhd_w_bcc.'))
        u, w = read(apath), read(wpath)
        with h5py.File(ppath) as stream:
            precisions.add((str(stream['mhd.prim'].dtype), binary_variable_bytes(wpath),
                            binary_variable_bytes(apath)))
            time = float(stream['Info'].attrs['Time'])
            if (not np.isclose(time, u['time'], rtol=0, atol=1e-12)
                    or not np.isclose(time, w['time'], rtol=0, atol=1e-12)):
                raise ValueError('Output times differ')
            if stream['Info'].attrs['IncludesGhost'] != 0:
                raise ValueError('Ghost zones must be omitted from the integral')
            if 'Default Device: Cuda' not in stream['Info'].attrs['KokkosConfig']:
                raise ValueError('PANGU output does not identify a CUDA execution space')
            prho = np.asarray(stream['mhd.prim'][:, 0], dtype=np.float64)
            pdens = np.asarray(stream['mhd.cons'][:, 0], dtype=np.float64)
            arho, adens = ordered_density(w, stream), ordered_density(u, stream)
            if any(a.shape != prho.shape for a in (pdens, arho, adens)):
                raise ValueError('Field shapes differ')
            if any(not np.isfinite(a).all() for a in (prho, pdens, arho, adens)):
                raise ValueError('Nonfinite density')
            volume = np.ones(prho.shape, dtype=np.float64)
            for axis, shape in zip('xyz', ((-1, 1, 1, prho.shape[3]),
                                           (-1, 1, prho.shape[2], 1),
                                           (-1, prho.shape[1], 1, 1)), strict=True):
                volume *= np.diff(np.asarray(stream['Locations/' + axis]), axis=1).reshape(shape)
            if core is None:
                core = arho > 1e-4
                if not core.any():
                    raise ValueError('Empty initial stellar core')
            row = dict(time=time, pangu_rho_max=float(prho.max()),
                       athena_rho_max=float(arho.max()),
                       pangu_baryon_mass=float(np.sum(pdens * volume)),
                       athena_baryon_mass=float(np.sum(adens * volume)),
                       rho_core_relative_l1=float(np.sum(np.abs(prho[core] - arho[core]))
                                                  / np.sum(np.abs(arho[core]))))
            rows.append(row)
        evidence.append({str(p): sha256(p) for p in (ppath, apath, wpath)})
    times = np.asarray([row['time'] for row in rows])
    if (times[0] != 0 or np.any(np.diff(times) <= 0)
            or not np.isclose(times[-1], target_time, rtol=0, atol=1e-12)):
        raise ValueError('Incomplete, duplicated or unordered trajectory')
    p_native, p_proof = native_mass_history(pangu_dir, 'mhd_mass', times)
    a_native, a_proof = native_mass_history(athena_dir, 'mass', times)
    for row, pmass, amass in zip(rows, p_native, a_native, strict=True):
        row.update(pangu_native_baryon_mass=float(pmass), athena_native_baryon_mass=float(amass),
                   pangu_native_baryon_drift=float(pmass / p_native[0] - 1),
                   athena_native_baryon_drift=float(amass / a_native[0] - 1))
    initial = rows[0]
    for row in rows:
        for code in ('pangu', 'athena'):
            row[code + '_rho_perturbation'] = row[code + '_rho_max'] / initial[code + '_rho_max'] - 1
            row[code + '_baryon_drift'] = row[code + '_baryon_mass'] / initial[code + '_baryon_mass'] - 1
        row['rho_max_relative_error'] = row['pangu_rho_max'] / row['athena_rho_max'] - 1
        row['rho_perturbation_difference'] = row['pangu_rho_perturbation'] - row['athena_rho_perturbation']
    return dict(samples=rows, files_sha256=evidence,
                native_mass_history=dict(pangu=p_proof, athena=a_proof,
                                         note='Native float64 reductions at exactly the shared snapshot times; snapshot mass is also reported and may be quantized.'),
                stored_precision=[dict(pangu_density_dtype=p, athena_primitive_bytes=w,
                                       athena_conserved_bytes=u) for p, w, u in sorted(precisions)],
                target_time=target_time, independent_comparison_completed=True,
                full_physical_acceptance=False, performance_claim=False,
                summary={
                    'initial_rho_max_relative_bias': rows[0]['rho_max_relative_error'],
                    'initial_baryon_relative_bias': initial['pangu_baryon_mass'] / initial['athena_baryon_mass'] - 1,
                    'initial_native_baryon_relative_bias': float(p_native[0] / a_native[0] - 1),
                    **{'max_abs_' + key: max(abs(row[key]) for row in rows)
                       for key in ('rho_max_relative_error', 'rho_perturbation_difference',
                                   'pangu_baryon_drift', 'athena_baryon_drift',
                                   'pangu_native_baryon_drift', 'athena_native_baryon_drift',
                                   'rho_core_relative_l1')}},
                limitations=['Single resolution and bounded 50-time pilot; not a convergence claim.',
                             'Finite domain and output precision contribute to the measured difference.',
                             'Uniform Cartesian, zero magnetic field and no horizon only.',
                             'Both conserved densities already include sqrt(gamma).'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu-dir', type=Path, required=True)
    parser.add_argument('--athenak-dir', type=Path, required=True)
    parser.add_argument('--athenak-source', type=Path, required=True)
    parser.add_argument('--athenak-gpu-proof', type=Path, required=True)
    parser.add_argument('--target-time', type=float, default=50)
    parser.add_argument('--expected-samples', type=int, default=11)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    proof = json.loads(args.athenak_gpu_proof.read_text())
    if (proof.get('status') != 'gpu_evolution_completed' or proof.get('returncode') != 0
            or not proof.get('execution', '').startswith('CUDA ')):
        raise ValueError('Completed AthenaK GPU execution proof is required')
    result = compare(args.pangu_dir, args.athenak_dir, args.athenak_source,
                     args.target_time, args.expected_samples)
    result['athenak_gpu_execution'] = proof
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['samples'][0]))
        writer.writeheader()
        writer.writerows(result['samples'])
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
