# PANGU

[English Version](README.md)

**PANGU**（**P**arthenon-based **A**ccretion and **N**umerical **G**eneral-relativistic **U**nified framework）是一个基于 [Parthenon](https://github.com/parthenon-hpc-lab/parthenon) 与 [Kokkos](https://github.com/kokkos/kokkos) 的 **GRMHD 数值模拟代码**。

PANGU 以中国上古创世之神**盘古**命名。传说盘古开天辟地，阳清为天、阴浊为地，从混沌中建立秩序——这正呼应了本代码的愿景：从第一性原理出发，模拟黑洞吸积、相对论性喷流与湍流等最极端的 astrophysical 现象。代码支持按问题目录选择初始化器、CPU/CUDA 后端构建、多种 GR 度规（Minkowski/BL/CKS/MKS）、多个 Riemann solver、多级 C2P 恢复流程、约束传输磁场演化与双温电子加热。

---

## 目录

- [代码结构](#代码结构)
- [依赖环境](#依赖环境)
- [构建](#构建)
- [运行](#运行)
- [输入文件](#输入文件)
- [分析与绘图](#分析与绘图)
- [常用算例](#常用算例)

---

## 代码结构

| 路径 | 说明 |
| --- | --- |
| `pangu/src/` | 求解器核心 |
| `pangu/src/constrained_transport/` | 约束传输 (CT) 磁场演化 |
| `pangu/src/fixer/` | 原语修复器 |
| `pangu/src/initialization/` | 包注册、字段声明、变量助记符 |
| `pangu/src/interpolation/` | 空间重构插值器：MC、PPM4 |
| `pangu/src/metric/` | 度规：Minkowski、BL、CKS、MKS |
| `pangu/src/physics/` | 态计算、逆变通量、快磁声速、双温模型 |
| `pangu/src/recovery/` | C2P 恢复：1D、1Dvsq、2D、Kastaun |
| `pangu/src/riemann_solver/` | Riemann solver：LAXF、HLL |
| `pangu/src/task_list/` | GRMHD 任务图 |
| `pangu/problem/` | 各算例的 `problem_generator.cpp` 和 `inputfile` |
| `scripts/shell/` | 构建、运行、分析入口脚本 |
| `scripts/python/` | 绘图脚本 |
| `athenak/` | AthenaK 参考实现 |
| `parthenon/` | Parthenon 框架（子模块） |
| `data/` | 默认运行输出目录 |
| `pic/` | 默认图像输出目录 |

---

## 依赖环境

- **CMake** 3.10+，推荐 3.16+
- **C++17** 编译器
- **Python 3** + numpy、h5py、matplotlib
- **HDF5**（推荐 Parallel HDF5）
- 可选：**MPI**（多进程）、**CUDA Toolkit**（GPU）、**OpenMP**（CPU 多线程）

推荐先装 OpenMPI，再装 Parallel HDF5（后者依赖 `mpicc`）：

```bash
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.gz
tar -zxvf openmpi-5.0.10.tar.gz && cd openmpi-5.0.10
./configure --prefix=/openmpi --enable-shared
make all install
echo 'export PATH=$PATH:/openmpi/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/openmpi/lib' >> ~/.bashrc
source ~/.bashrc

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

> 如不允许写根目录，请将 `--prefix` 改为自己的安装路径。

---

## 构建

入口：[`make.sh`](https://github.com/adamdarx/PANGU/blob/master/scripts/shell/make.sh)，默认按 **CUDA** 后端构建。

```bash
# GPU 构建
PROBLEM=fm_torus_mks BUILD_DIR=build bash ./scripts/shell/make.sh

# CPU 构建
PROBLEM=orszag_tang_vortex BUILD_DIR=build bash ./scripts/shell/make.sh cpu
```

### 构建环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PROBLEM` | `shock_tube` | 选择 `pangu/problem/<PROBLEM>` |
| `BUILD_DIR` | `build` | 构建目录 |
| `BUILD_TYPE` | `Release` | CMake 构建类型 |
| `BUILD_JOBS` | `4` | 并行编译数 |
| `ENABLE_CUDA` | `ON` | CUDA 目标 |
| `ENABLE_OPENMP` | `OFF` | OpenMP |
| `HOST_ARCH` | `NATIVE` | CPU 架构 |
| `DEVICE_ARCH` | 空 | GPU 架构 (`AMPERE80`/`HOPPER90`/`AMD_GFX90A`/`INTEL_PVC` 等) |
| `CMAKE_EXTRA_ARGS` | 空 | 额外 CMake 参数 |

### 构建产物

| 后端 | 可执行文件 |
| --- | --- |
| CPU | `build/pangu/src/pangu.host` |
| CUDA | `build/pangu/src/pangu.cuda` |

---

## 运行

入口：[`run.sh`](https://github.com/adamdarx/PANGU/blob/master/scripts/shell/run.sh)

```bash
bash ./scripts/shell/run.sh
```

```bash
# CPU 单进程 FM torus
BUILD_DIR=build ENABLE_CUDA=OFF PROBLEM=fm_torus_mks bash ./scripts/shell/run.sh -n 1

# 自定义 input 文件
BUILD_DIR=build PROBLEM=fm_torus_mks bash ./scripts/shell/run.sh -i /path/to/custom.athinput
```

| 参数 | 说明 |
| --- | --- |
| `-i, --input <path>` | 指定 inputfile |
| `-b, --build-dir <path>` | 构建目录 |
| `-p, --problem <name>` | 算例名 |
| `-n, --np <N>` | MPI 进程数 |

默认输出：`data/<problem>`

---

## 输入文件

PANGU 使用 Parthenon 风格输入文件。典型结构：

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

### `<core>` 核心参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `cfl_number` | Real | `0.3` | CFL 数 |
| `adiabatic_index` | Real | `1.666...` | 绝热指数 Γ |
| `recovery` | String | `chainer` | C2P：`chainer`（链式回退 2D→1Dvsq→1D）或 `robuster`（Kastaun） |
| `density_floor` | Real | `1e-5` | 密度下限基值 |
| `energy_floor` | Real | `1e-7` | 能量下限基值 |
| `density_floor_pow` | Real | `-1.5` | 密度下限径向幂律 |
| `energy_floor_pow` | Real | `-2.5` | 能量下限径向幂律 |
| `enable_B` | Boolean | `false` | 启用磁场 |
| `enable_heating` | Boolean | `false` | 启用电子加热 |
| `riemann_solver` | String | `laxf` | `hll` |
| `limiter` | String | `ppm4` | `mc` |

### `<fm_torus>` 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `rin` | Real | `6.0` | 内缘半径 |
| `rmax` | Real | `12.0` | 压强峰值半径 |
| `kappa` | Real | `1e-3` | 熵归一化常数 |
| `perturbation` | Real | `2e-2` | 随机扰动幅度 |
| `beta` | Real | `100.0` | 目标等离子体 beta |
| `aphi_rho_cut` | Real | `0.2` | 矢量势密度截断 |
| `magnetic_field` | String | `sane` | 磁场构型：`sane`（标准）/ `mad`（磁制动盘） |

> `magnetic_field` 仅在 `enable_B = true` 时生效。
> `mad` 构型 = `sane` × (r/rin × sinθ)³ × e^{-r/400}，磁场集中于 torus 内缘。

---

## 分析与绘图

入口：[`analyze.sh`](https://github.com/adamdarx/PANGU/blob/master/scripts/shell/analyze.sh)

```bash
bash ./scripts/shell/analyze.sh -p <problem> [options]
```

| 模式 | 开关 | 说明 |
| --- | --- | --- |
| contour1d | 默认 | 单张 1D 曲线图 |
| movie2d | `--movie2d` | 2D 帧序列 |
| xzplot | `--xzplot` | x-z 平面映射（MKS） |
| xzplot (CKS) | `--xzplot --cks` | x-z 平面映射（CKS） |
| 2T 温度 | `--2t` | 离子/电子温度双面板 |

```bash
# 示例
bash ./scripts/shell/analyze.sh -p fm_torus_mks -f density                     # 1D 密度
bash ./scripts/shell/analyze.sh -p fm_torus_mks -f density --xzplot            # x-z MKS
bash ./scripts/shell/analyze.sh -p fm_torus_cks -f density --xzplot --cks      # x-z CKS
bash ./scripts/shell/analyze.sh -p fm_torus_mks --2t -w 4                      # 双温温度
```

---

## 常用算例

| 算例 | 度规 | 说明 |
| --- | --- | --- |
| [`shock_tube`](https://github.com/adamdarx/PANGU/blob/master/pangu/problem/shock_tube/inputfile) | Minkowski | MHD shock tube |
| [`orszag_tang_vortex`](https://github.com/adamdarx/PANGU/blob/master/pangu/problem/orszag_tang_vortex/inputfile) | Minkowski | Orszag-Tang vortex |
| [`kelvin_helmholtz`](https://github.com/adamdarx/PANGU/blob/master/pangu/problem/kelvin_helmholtz/inputfile) | Minkowski | Kelvin-Helmholtz 不稳定性 |
| [`blast`](https://github.com/adamdarx/PANGU/blob/master/pangu/problem/blast/inputfile) | Minkowski | MHD 爆炸波 |
| [`bondi_flow`](https://github.com/adamdarx/PANGU/blob/master/pangu/problem/bondi_flow/inputfile) | MKS | GR Bondi 吸积 |
| [`bondi_flow_cks`](https://github.com/adamdarx/PANGU/blob/master/pangu/problem/bondi_flow_cks/inputfile) | CKS | CKS GR Bondi 吸积 |
| [`magnetised_bondi`](https://github.com/adamdarx/PANGU/blob/master/pangu/problem/magnetised_bondi/inputfile) | MKS | 磁化 GR Bondi 吸积 |
| [`fm_torus_mks`](https://github.com/adamdarx/PANGU/blob/master/pangu/problem/fm_torus_mks/inputfile) | MKS | Kerr FM torus（MKS） |
| [`fm_torus_cks`](https://github.com/adamdarx/PANGU/blob/master/pangu/problem/fm_torus_cks/inputfile) | CKS | Kerr FM torus（CKS，3D） |

新增算例需 `problem_generator.cpp` + `inputfile`，然后 `PROBLEM=<name> bash ./scripts/shell/make.sh`。

---