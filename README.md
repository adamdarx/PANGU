# PANGU

[Chinese version | 中文版本](README_CN.md)

**PANGU** (**P**arthenon-based **A**ccretion and **N**umerical **G**eneral-relativistic **U**nified framework) is a GRMHD simulation code built on [Parthenon](https://github.com/parthenon-hpc-lab/parthenon) and [Kokkos](https://github.com/kokkos/kokkos).

PANGU is named after the Chinese primordial deity who created the universe by separating heaven and earth, reflecting our code's ambition to model the most extreme astrophysical phenomena from first principles. The name embodies the spirit of building order from chaos — transforming raw conservation laws into coherent simulations of black hole accretion, jet launching, and relativistic turbulence.

The code supports per-problem initializers, CPU/CUDA backends, multiple GR metrics (Minkowski/BL/CKS/MKS), multiple Riemann solvers, multi-level C2P recovery, constrained-transport magnetic field evolution, and two-temperature electron heating.

---

## Table of Contents

- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Build](#build)
- [Run](#run)
- [Input File](#input-file)
- [Analysis and Visualization](#analysis-and-visualization)
- [Example Problems](#example-problems)

---

## Code Structure

| Path | Description |
| --- | --- |
| `pangu/src/` | Solver core |
| `pangu/src/constrained_transport/` | Constrained transport (CT) for divergence-free B |
| `pangu/src/fixer/` | Primitive and recovery fixers |
| `pangu/src/initialization/` | Package registration, fields, variable mnemonics |
| `pangu/src/interpolation/` | Spatial reconstruction: MC, PPM4 |
| `pangu/src/metric/` | Metrics: Minkowski, BL, CKS, MKS |
| `pangu/src/physics/` | State calculation, contravariant fluxes, fast magnetosonic speed, two-temperature models |
| `pangu/src/recovery/` | C2P recovery: 1D, 1Dvsq, 2D, Kastaun |
| `pangu/src/riemann_solver/` | Riemann solvers: LAXF, HLL |
| `pangu/src/task_list/` | GRMHD task graph |
| `pangu/problem/` | Per-problem `problem_generator.cpp` and `inputfile` |
| `scripts/shell/` | Build, run, and analysis entry scripts |
| `scripts/python/` | Plotting scripts |
| `athenak/` | AthenaK reference implementation |
| `parthenon/` | Parthenon framework (submodule) |
| `data/` | Default run output directory |
| `pic/` | Default image output directory |

---

## Dependencies

- **CMake** 3.10+, recommended 3.16+
- **C++17** compiler
- **Python 3** + numpy, h5py, matplotlib
- **HDF5** (Parallel HDF5 recommended)
- Optional: **MPI** (multi-process), **CUDA Toolkit** (GPU), **OpenMP** (CPU multi-threading)

Install OpenMPI first, then Parallel HDF5 (which requires `mpicc`):

```bash
# OpenMPI
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.gz
tar -zxvf openmpi-5.0.10.tar.gz && cd openmpi-5.0.10
./configure --prefix=/openmpi --enable-shared
make all install
echo 'export PATH=$PATH:/openmpi/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/openmpi/lib' >> ~/.bashrc
source ~/.bashrc

# Parallel HDF5
wget https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_3/downloads/hdf5-1.14.3.tar.gz
tar -zxvf hdf5-1.14.3.tar.gz && cd hdfsrc
export CC=mpicc && export HDF5_MPI="ON"
./configure --enable-shared --enable-parallel --prefix=/hdf5
export HDF5_DIR="/hdf5"
make && make install && make check-install
echo 'export PATH=$PATH:/hdf5/bin' >> ~/.bashrc
echo 'export HDF5_ROOT=/hdf5' >> ~/.bashrc
source ~/.bashrc
```

> Change `--prefix` to a writable path if `/openmpi` or `/hdf5` are not accessible.

---

## Build

Entry point: [`make.sh`](https://github.com/adamdarx/Pangu/blob/master/scripts/shell/make.sh), builds with **CUDA** backend by default.

```bash
# GPU build
PROBLEM=fm_torus_mks BUILD_DIR=build bash ./scripts/shell/make.sh

# CPU build
PROBLEM=orszag_tang_vortex BUILD_DIR=build bash ./scripts/shell/make.sh cpu
```

### Build Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `PROBLEM` | `shock_tube` | Select `pangu/problem/<PROBLEM>` |
| `BUILD_DIR` | `build` | Build directory |
| `BUILD_TYPE` | `Release` | CMake build type |
| `BUILD_JOBS` | `4` | Parallel compile jobs |
| `ENABLE_CUDA` | `ON` | CUDA target |
| `ENABLE_OPENMP` | `OFF` | OpenMP |
| `HOST_ARCH` | `NATIVE` | CPU architecture |
| `DEVICE_ARCH` | empty | GPU architecture (`AMPERE80`/`HOPPER90`/`AMD_GFX90A`/`INTEL_PVC`, etc.) |
| `CMAKE_EXTRA_ARGS` | empty | Extra CMake arguments |

### Build Artifacts

| Backend | Executable |
| --- | --- |
| CPU | `build/pangu/src/pangu.host` |
| CUDA | `build/pangu/src/pangu.cuda` |

---

## Run

Entry point: [`run.sh`](https://github.com/adamdarx/Pangu/blob/master/scripts/shell/run.sh)

```bash
bash ./scripts/shell/run.sh
```

```bash
# CPU single-process FM torus
BUILD_DIR=build ENABLE_CUDA=OFF PROBLEM=fm_torus_mks bash ./scripts/shell/run.sh -n 1

# Custom input file
BUILD_DIR=build PROBLEM=fm_torus_mks bash ./scripts/shell/run.sh -i /path/to/custom.athinput
```

| Argument | Description |
| --- | --- |
| `-i, --input <path>` | Specify input file |
| `-b, --build-dir <path>` | Build directory |
| `-p, --problem <name>` | Problem name |
| `-n, --np <N>` | MPI process count |

Default output: `data/<problem>`

---

## Input File

PANGU uses Parthenon-style input files. Typical structure:

```ini
<parthenon/job>
problem_id = fm_torus_mks

<parthenon/mesh>
nx1 = 512
nx2 = 256
nx3 = 1

<parthenon/meshblock>
nx1 = 512
nx2 = 256
nx3 = 1

<parthenon/time>
integrator = rk2
tlim = 2000

<core>
cfl_number = 0.4
adiabatic_index = 1.666666667
recovery = chainer
density_floor = 1e-5
energy_floor = 4.641588833612779e-9
enable_B = true
enable_heating = true
riemann_solver = laxf
limiter = ppm4

<metric>
type = mks
h = 0.7
a = 0.9375

<fm_torus>
rin = 6.0
rmax = 12.0
kappa = 1e-3
perturbation = 4e-2
beta = 100.0
aphi_rho_cut = 0.2
magnetic_field = sane

<parthenon/output0>
file_type = hdf5
write_xdmf = true
dt = 1
variables = density, entropy, electron_entropy
```

### `<core>` Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `cfl_number` | Real | `0.3` | CFL number |
| `adiabatic_index` | Real | `1.666...` | Adiabatic index Γ |
| `recovery` | String | `chainer` | C2P method: `chainer` (fallback 2D->1Dvsq->1D) or `robuster` (Kastaun) |
| `density_floor` | Real | `1e-5` | Density floor base |
| `energy_floor` | Real | `1e-7` | Energy floor base |
| `density_floor_pow` | Real | `-1.5` | Density floor radial power-law |
| `energy_floor_pow` | Real | `-2.5` | Energy floor radial power-law |
| `enable_B` | Boolean | `false` | Enable magnetic field |
| `enable_heating` | Boolean | `false` | Enable electron heating |
| `riemann_solver` | String | `laxf` | `laxf` or `hll` |
| `limiter` | String | `ppm4` | `ppm4` or `mc` |

### `<fm_torus>` Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `rin` | Real | `6.0` | Inner edge radius |
| `rmax` | Real | `12.0` | Pressure maximum radius |
| `kappa` | Real | `1e-3` | Entropy normalization constant |
| `perturbation` | Real | `2e-2` | Random perturbation amplitude |
| `beta` | Real | `100.0` | Target plasma beta (magnetic normalization) |
| `aphi_rho_cut` | Real | `0.2` | Vector potential density cutoff |
| `magnetic_field` | String | `sane` | Field configuration: `sane` (standard) or `mad` (magnetically arrested disk) |

> `magnetic_field` only takes effect when `enable_B = true`.
> `mad` = `sane` x (r/rin x sinθ)3 x e^{-r/400}, concentrating the field near the torus inner edge.

---

## Analysis and Visualization

Entry point: [`analyze.sh`](https://github.com/adamdarx/Pangu/blob/master/scripts/shell/analyze.sh)

```bash
bash ./scripts/shell/analyze.sh -p <problem> [options]
```

| Mode | Flag | Description |
| --- | --- | --- |
| contour1d | default | Single 1D contour plot |
| movie2d | `--movie2d` | 2D frame sequence |
| xzplot | `--xzplot` | x-z plane mapping (MKS) |
| xzplot (CKS) | `--xzplot --cks` | x-z plane mapping (CKS) |
| 2T temperature | `--2t` | Ion/electron temperature dual panel |

```bash
# Examples
bash ./scripts/shell/analyze.sh -p fm_torus_mks -f density                     # 1D density
bash ./scripts/shell/analyze.sh -p fm_torus_mks -f density --xzplot            # x-z MKS
bash ./scripts/shell/analyze.sh -p fm_torus_cks -f density --xzplot --cks      # x-z CKS
bash ./scripts/shell/analyze.sh -p fm_torus_mks --2t -w 4                      # 2T temperature
```

---

## Example Problems

| Problem | Metric | Description |
| --- | --- | --- |
| [`shock_tube`](https://github.com/adamdarx/Pangu/blob/master/pangu/problem/shock_tube/inputfile) | Minkowski | MHD shock tube |
| [`orszag_tang_vortex`](https://github.com/adamdarx/Pangu/blob/master/pangu/problem/orszag_tang_vortex/inputfile) | Minkowski | Orszag-Tang vortex |
| [`kelvin_helmholtz`](https://github.com/adamdarx/Pangu/blob/master/pangu/problem/kelvin_helmholtz/inputfile) | Minkowski | Kelvin-Helmholtz instability |
| [`blast`](https://github.com/adamdarx/Pangu/blob/master/pangu/problem/blast/inputfile) | Minkowski | MHD blast wave |
| [`bondi_flow`](https://github.com/adamdarx/Pangu/blob/master/pangu/problem/bondi_flow/inputfile) | MKS | GR Bondi accretion |
| [`bondi_flow_cks`](https://github.com/adamdarx/Pangu/blob/master/pangu/problem/bondi_flow_cks/inputfile) | CKS | CKS GR Bondi accretion |
| [`magnetised_bondi`](https://github.com/adamdarx/Pangu/blob/master/pangu/problem/magnetised_bondi/inputfile) | MKS | Magnetized GR Bondi accretion |
| [`fm_torus_mks`](https://github.com/adamdarx/Pangu/blob/master/pangu/problem/fm_torus_mks/inputfile) | MKS | Kerr FM torus (MKS) |
| [`fm_torus_cks`](https://github.com/adamdarx/Pangu/blob/master/pangu/problem/fm_torus_cks/inputfile) | CKS | Kerr FM torus (CKS, 3D) |

To add a new problem, create `problem_generator.cpp` + `inputfile` under `pangu/problem/<name>/`, then build with `PROBLEM=<name> bash ./scripts/shell/make.sh`.
