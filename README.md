# Pangu

Pangu 是一个基于 Parthenon 与 Kokkos 的 GRMHD 数值模拟代码。当前代码支持按问题目录选择初始化器、CPU/CUDA 后端构建、Minkowski/BL/MKS/CKS 度规下的算例、HLL/HLLD/LAXF Riemann solver、新的 conservative-to-primitive (C2P) 恢复流程，以及单模型双温电子加热。

本文档以实际使用流程为主：如何构建、运行、设置输入文件、分析输出，以及如何理解双温模式下的温度单位。

## 目录

- [代码结构](#代码结构)
- [依赖环境](#依赖环境)
- [构建](#构建)
- [运行](#运行)
- [输入文件](#输入文件)
- [双温电子加热](#双温电子加热)
- [分析与绘图](#分析与绘图)
- [温度单位](#温度单位)
- [常用算例](#常用算例)
- [开发说明](#开发说明)
- [常见问题](#常见问题)

## 代码结构

主要目录如下：

| 路径 | 说明 |
| --- | --- |
| `pangu/src` | 求解器核心、任务图、物理模块、恢复、通量、fixer |
| `pangu/problem` | 每个算例的 `problem_generator.cpp` 和 `inputfile` |
| `scripts/shell` | 构建、运行、分析入口脚本 |
| `scripts/python` | 自定义绘图脚本 |
| `kharma` | 参考实现，用于对照物理模型和输入文件风格 |
| `data` | 默认运行输出目录，运行时生成 |
| `pic` | 默认图像输出目录，分析时生成 |

当前构建系统通过 `PROBLEM=<name>` 选择 `pangu/problem/<name>/problem_generator.cpp`。脚本默认启用 problem proxy 模式，把活动问题映射到 `pangu/problem/__active_problem__`，从而避免频繁修改 CMake 目标。

## 依赖环境

基础依赖：

- CMake 3.10 或更新版本，推荐 3.16+
- 支持 C++17 的编译器
- Python 3
- HDF5

可选依赖：

- MPI：用于多进程运行
- CUDA toolkit：用于 GPU 构建
- OpenMP：CPU 多线程后端

如果需要通过代理环境下载依赖，可以先配置 Clash 之类的本地代理，再执行下面的安装流程。

推荐安装顺序是先装 OpenMPI，再装 Parallel HDF5，因为后者需要 `mpicc`、MPI 头文件和 MPI 库支持。

```bash
# 可选：Clash 代理示例
wget https://github.com/doreamon-design/clash/releases/download/v2.0.24/clash_2.0.24_linux_amd64.tar.gz
tar zxvf clash_2.0.24_linux_amd64.tar.gz
chmod +x clash
mv clash /usr/local/bin/clash
mkdir /etc/clash
cat > /etc/clash/config.yaml << EOF
# Clash 配置示例
# 直接从客户端软件复制
EOF
echo "export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7891" >> ~/.bashrc
source ~/.bashrc

# OpenMPI
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.gz
tar -zxvf openmpi-5.0.10.tar.gz
cd openmpi-5.0.10
./configure --prefix=/openmpi --with-cuda=/usr/lib/cuda --enable-mpi-cxx --enable-shared
make all install
export PATH=$PATH:/openmpi/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/openmpi/lib
source ~/.bashrc

# Parallel HDF5
wget https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_3/downloads/hdf5-1.14.3.tar.gz
tar -zxvf hdf5-1.14.3.tar.gz
export CC=mpicc
export HDF5_MPI="ON"
./configure --enable-shared --enable-parallel --prefix=/hdf5
export HDF5_DIR="/hdf5"
make
make install
make check-install
export PATH=$PATH:"/hdf5/bin"
export HDF5_ROOT="/hdf5"
source ~/.bashrc
```

必要说明：

- `Clash` 只是可选示例，目的是在网络受限环境下下载源码包；如果你本机有其他代理工具，也可以直接替换。
- `--prefix=/openmpi` 和 `--prefix=/hdf5` 需要有写权限；如果系统不允许写根目录，请改成自己的安装前缀。
- `--with-cuda=/usr/lib/cuda` 仅在 CUDA 构建场景下有意义；如果没有 CUDA，可以去掉这项。
- `make check-install` 需要在 HDF5 源码目录执行，确认并行 HDF5 安装可用。

Python 分析依赖通常包括：

```bash
python3 -m pip install --user numpy h5py matplotlib
```

如果本地 Parthenon 附带 requirements 文件，也可以使用：

```bash
python3 -m pip install --user -r parthenon/requirements.txt
```

## 构建

构建入口：

```bash
bash ./scripts/shell/make.sh
```

不带后端参数时，`make.sh` 默认按 CUDA 后端构建

常用 GPU/GRMHD 构建示例：

```bash
PROBLEM=fm_torus BUILD_DIR=build bash ./scripts/shell/make.sh
```

Minkowski 度规测试问题构建示例：

```bash
PROBLEM=orszag_tang_vortex BUILD_DIR=build bash ./scripts/shell/make.sh cpu
```

如果需要固定 Kokkos 的 GPU 架构，请设置 `DEVICE_ARCH`。脚本会自动展开为 `-DKokkos_ARCH_<ARCH>=ON`；例如：

```bash
DEVICE_ARCH=AMPERE80 PROBLEM=fm_torus BUILD_DIR=build bash ./scripts/shell/make.sh
DEVICE_ARCH=AMD_GFX90A PROBLEM=fm_torus BUILD_DIR=build bash ./scripts/shell/make.sh
DEVICE_ARCH=INTEL_PVC PROBLEM=fm_torus BUILD_DIR=build bash ./scripts/shell/make.sh
```

`DEVICE_ARCH` 一次只建议填一个值；Kokkos 官方配置页也说明了单个 device backend 只能配一个 architecture。若不设置，CUDA 构建会尝试自动探测。

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ENABLE_OPENMP` | `OFF` | 是否启用 OpenMP |
| `ENABLE_CUDA` | `ON` | 是否构建 CUDA 目标 |
| `PROBLEM` | `shock_tube` | 选择 `pangu/problem/<PROBLEM>` |
| `BUILD_DIR` | `build` | 构建目录 |
| `BUILD_TYPE` | `Release` | CMake 构建类型 |
| `BUILD_JOBS` | `4` | 并行编译任务数 |
| `HOST_ARCH` | `NATIVE` | CPU 架构选项，对应 `Kokkos_ARCH_NATIVE` 或其他 CPU 架构 |
| `DEVICE_ARCH` | 空 | GPU 架构选项，对应 `Kokkos_ARCH_<ARCH>` |
| `CMAKE_EXTRA_ARGS` | 空 | 透传给 CMake 的额外参数 |

Kokkos GPU 架构速查：

| 平台 | `DEVICE_ARCH` 值 | 实际 CMake 选项 | 备注 |
| --- | --- | --- | --- |
| NVIDIA | `BLACKWELL120` | `Kokkos_ARCH_BLACKWELL120` | Blackwell, CC 12.0 |
| NVIDIA | `BLACKWELL103` | `Kokkos_ARCH_BLACKWELL103` | Blackwell, CC 10.3 |
| NVIDIA | `BLACKWELL100` | `Kokkos_ARCH_BLACKWELL100` | Blackwell, CC 10.0 |
| NVIDIA | `HOPPER90` | `Kokkos_ARCH_HOPPER90` | Hopper, CC 9.0 |
| NVIDIA | `ADA89` | `Kokkos_ARCH_ADA89` | Ada Lovelace, CC 8.9 |
| NVIDIA | `AMPERE87` | `Kokkos_ARCH_AMPERE87` | Ampere, CC 8.7 |
| NVIDIA | `AMPERE86` | `Kokkos_ARCH_AMPERE86` | Ampere, CC 8.6 |
| NVIDIA | `AMPERE80` | `Kokkos_ARCH_AMPERE80` | Ampere, CC 8.0 |
| NVIDIA | `TURING75` | `Kokkos_ARCH_TURING75` | Turing, CC 7.5 |
| NVIDIA | `VOLTA72` | `Kokkos_ARCH_VOLTA72` | Volta, CC 7.2 |
| NVIDIA | `VOLTA70` | `Kokkos_ARCH_VOLTA70` | Volta, CC 7.0 |
| NVIDIA | `PASCAL61` | `Kokkos_ARCH_PASCAL61` | Pascal, CC 6.1 |
| NVIDIA | `PASCAL60` | `Kokkos_ARCH_PASCAL60` | Pascal, CC 6.0 |
| NVIDIA | `MAXWELL53` | `Kokkos_ARCH_MAXWELL53` | Maxwell, CC 5.3 |
| NVIDIA | `MAXWELL52` | `Kokkos_ARCH_MAXWELL52` | Maxwell, CC 5.2 |
| NVIDIA | `MAXWELL50` | `Kokkos_ARCH_MAXWELL50` | Maxwell, CC 5.0 |
| AMD | `AMD_GFX950` | `Kokkos_ARCH_AMD_GFX950` | MI355X / MI350X |
| AMD | `AMD_GFX942_APU` | `Kokkos_ARCH_AMD_GFX942_APU` | MI300A |
| AMD | `AMD_GFX942` | `Kokkos_ARCH_AMD_GFX942` | MI300X；MI300A 新版本建议用 `_APU` |
| AMD | `AMD_GFX940` | `Kokkos_ARCH_AMD_GFX940` | MI300A pre-production |
| AMD | `AMD_GFX90A` | `Kokkos_ARCH_AMD_GFX90A` | MI200 系列 |
| AMD | `AMD_GFX908` | `Kokkos_ARCH_AMD_GFX908` | MI100 |
| AMD | `AMD_GFX906` | `Kokkos_ARCH_AMD_GFX906` | MI50 / MI60 |
| AMD | `AMD_GFX1201` | `Kokkos_ARCH_AMD_GFX1201` | Radeon AI PRO R9700 / RX 9070 XT |
| AMD | `AMD_GFX1103` | `Kokkos_ARCH_AMD_GFX1103` | Ryzen 8000G Phoenix APU |
| AMD | `AMD_GFX1100` | `Kokkos_ARCH_AMD_GFX1100` | 7900XT |
| AMD | `AMD_GFX1030` | `Kokkos_ARCH_AMD_GFX1030` | V620 / W6800 |
| Intel | `INTEL_PVC` | `Kokkos_ARCH_INTEL_PVC` | Ponte Vecchio / Data Center GPU Max 1550 |
| Intel | `INTEL_XEHP` | `Kokkos_ARCH_INTEL_XEHP` | Xe-HP |
| Intel | `INTEL_DG2` | `Kokkos_ARCH_INTEL_DG2` | Intel Flex / Arc |
| Intel | `INTEL_DG1` | `Kokkos_ARCH_INTEL_DG1` | Iris Xe MAX |
| Intel | `INTEL_GEN12LP` | `Kokkos_ARCH_INTEL_GEN12LP` | Gen12LP |
| Intel | `INTEL_GEN11` | `Kokkos_ARCH_INTEL_GEN11` | Gen11 |
| Intel | `INTEL_GEN9` | `Kokkos_ARCH_INTEL_GEN9` | Gen9 |
| Intel | `INTEL_GEN` | `Kokkos_ARCH_INTEL_GEN` | JIT 模式 |

旧版 Kepler 选项 `Kokkos_ARCH_KEPLER30/32/35/37` 已在 Kokkos 5.0 移除，不建议再用。若需要 CPU 架构，仍然使用 `HOST_ARCH` 对应的 `Kokkos_ARCH_NATIVE` 或其他 CPU 选项。

构建产物：

| 后端 | 可执行文件 |
| --- | --- |
| CPU | `build/pangu/src/pangu.host` |
| CUDA | `build/pangu/src/pangu.cuda` |

构建脚本会写出 `.pangu_build.env`，记录上次构建的关键配置，便于复现实验。

## 运行

运行入口：

```bash
bash ./scripts/shell/run.sh
```

示例：

```bash
BUILD_DIR=build ENABLE_CUDA=OFF PROBLEM=fm_torus bash ./scripts/shell/run.sh -n 1
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-i, --input <path>` | 指定 inputfile；默认使用 `pangu/problem/<problem>/inputfile` |
| `-b, --build-dir <path>` | 指定构建目录 |
| `-p, --problem <name>` | 指定算例名 |
| `-n, --np <N>` | MPI 进程数 |

默认输出目录为：

```text
data/<problem>
```

如果 `-n` 或 `MPI_NP` 大于 1，系统需要提供 `mpirun`。

## 输入文件

Pangu 使用 Parthenon 风格输入文件。典型结构如下：

```ini
<parthenon/job>
problem_id = fm_torus

<parthenon/mesh>
nx1 = 512
nx2 = 256
nx3 = 1

<parthenon/time>
integrator = rk2
tlim = 2000

<core>
adiabatic_index = 1.666666667
density_floor = 1e-5
energy_floor = 4.641588833612779e-9
riemann_solver = hlld

<electrons>
on = true
model = constant
gamma_e = 1.333333333
gamma_p = 1.666666667
fel_0 = 0.1
fel_constant = 0.1
ratio_min = 0.001
ratio_max = 1000.0

<metric>
h = 0.7
a = 0.9375

<fm_torus>
rin = 6.0
rmax = 12.0
```

`<core>` 控制单流体 GRMHD 主方程参数；`<electrons>` 控制双温电子加热；`<metric>` 与具体问题段控制 GR 度规和初始化。`<core>` 中的 `riemann_solver` 目前支持 `laxf`、`hll` 和 `hlld`，默认值是 `laxf`。

新的 C2P 恢复流程已经接入任务图：通量更新后会先做 conservative-to-primitive 恢复，再进入修复器和电子加热流程。当前采用的方法见 [arXiv:2005.01821](https://arxiv.org/abs/2005.01821)。

## 双温电子加热

Pangu 当前采用单模型双温模式：每次运行只演化一个 `electron_entropy` 字段。模型通过 `<electrons>` 段中的字符串参数选择：

```ini
<electrons>
on = true
model = howes
```

可选模型：

| `model` | 含义 |
| --- | --- |
| `constant` | 固定电子加热比例 `fel_constant` |
| `howes` | Howes 类湍流加热 prescription |
| `kawazura` | Kawazura et al. 电子/离子加热比例 |
| `werner` | Werner et al. 磁化率相关 prescription |
| `rowan` | Rowan et al. reconnection prescription |
| `sharma` | Sharma et al. prescription |

程序内部在 `pangu/src/physics/two_temperature.h` 中定义 `enum MODEL`，并通过 `StringToMODEL()` 与 `MODELToString()` 完成字符串和枚举之间的转换。运行时只保存规范化后的 `model_name` 参数，不再保存额外的整数模型参数。

电子加热的核心流程是：

1. 通量更新后，通过 primitive recovery 得到新的总内能与总熵。
2. 对比 advected entropy 与 energy-conserving entropy，估计本步耗散。
3. 根据所选模型计算电子获得的耗散比例 `fel`。
4. 更新 `electron_entropy`。
5. 按 `ratio_min` 与 `ratio_max` 限制温度比。

重要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | `constant` | 电子加热模型 |
| `gamma_e` | `4/3` | 电子绝热指数 |
| `gamma_p` | `5/3` | 离子绝热指数 |
| `fel_0` | `0.1` | 初始电子熵比例 |
| `fel_constant` | `0.1` | `constant` 模型的固定电子加热比例 |
| `limit_kel` | `true` | 是否对 `constant` 模型应用电子熵上下限 |
| `ratio_min` | `0.001` | 最小 `T_p/T_e` |
| `ratio_max` | `1000.0` | 最大 `T_p/T_e` |
| `suppress_highb_heat` | `false` | 高磁化区域是否抑制耗散加热 |
| `enforce_positive_dissipation` | `false` | 是否强制耗散非负 |

与 KHARMA 的关系：

- Pangu 已迁移 KHARMA 中上述加热模型的核心 heating fraction 公式。
- Pangu 当前只运行一个模型；KHARMA 可以在同一次模拟中同时演化多个 `Kel_*` 字段用于模型对比。
- 除时序、ghost 区、floor/fixup 顺序等数值实现细节外，单模型双温加热的物理 prescription 与 KHARMA 对齐。

## 分析与绘图

分析入口：

```bash
bash ./scripts/shell/analyze.sh
```

常用模式：

| 模式 | 开关 | 说明 |
| --- | --- | --- |
| contour1d | 默认 | 使用 Parthenon `contour1d.py` 绘制单张图 |
| movie2d | `--movie2d` | 生成 2D 帧序列 |
| xzplot | `--xzplot` | 将坐标映射到 x-z 平面并绘制任意字段 |
| 2T 温度 | `--2t` | 绘制离子/电子温度 x-z 双面板 |

密度或其它字段的 x-z 图：

```bash
bash ./scripts/shell/analyze.sh -p fm_torus -f density --xzplot
```

双温温度图：

```bash
bash ./scripts/shell/analyze.sh -p fm_torus --2t -w 4
```

对应 Python 脚本为：

```text
scripts/python/xz_temperature_plot.py
```

该脚本读取 `density`、`entropy` 和 `electron_entropy`，并生成 `log10(T [K])` 的离子/电子温度图。默认会使用 inputfile 中的 Kerr 参数 `a` 与坐标压缩参数 `h` 绘制视界和 x-z 映射。

## 温度单位

代码中的温度不是 Kelvin。Pangu/KHARMA 在电子加热模型中使用的温度变量是无量纲量：

```text
Theta = k_B T / (m_p c^2)
```

因此物理温度为：

```text
T[K] = Theta * m_p c^2 / k_B
```

使用 CGS 常数：

```text
m_p = 1.67262171e-24 g
c   = 2.99792458e10 cm s^-1
k_B = 1.380649e-16 erg K^-1
```

得到：

```text
m_p c^2 / k_B = 1.0888194058954387e13 K
```

`xz_temperature_plot.py` 已使用这个因子，而不是近似的 `1e13`。如果需要测试不同单位约定，可以显式覆盖：

```bash
python3 scripts/python/xz_temperature_plot.py \
  --temperature-unit-k 1.0888194058954387e13 \
  --output-directory pic/fm_torus/xztemp \
  data/fm_torus/*.phdf
```

脚本中的温度定义与加热模型一致：

```text
Theta_p = (gamma_p - 1) u / rho
Theta_e = K_e rho^(gamma_e - 1)
```

其中 `u` 由总熵反推：

```text
u = K_tot rho^gamma / (gamma - 1)
```

## 常用算例

| 算例 | 说明 | 推荐模式 |
| --- | --- | --- |
| `shock_tube` | Minkowski 度规下的 MHD shock tube | GRMHD |
| `orszag_tang_vortex` | Minkowski 度规下的 Orszag-Tang vortex | GRMHD |
| `kelvin_helmholtz` | Minkowski 度规下的 Kelvin-Helmholtz instability | GRMHD |
| `bondi_flow` | GR Bondi accretion | GRMHD |
| `fm_torus` | Kerr spacetime 下的 Fishbone-Moncrief torus | GRMHD |

新增算例时至少需要：

```text
pangu/problem/<name>/problem_generator.cpp
pangu/problem/<name>/inputfile
```

然后用：

```bash
PROBLEM=<name> bash ./scripts/shell/make.sh
```

重新构建。

## 开发说明

关键文件：

| 文件 | 说明 |
| --- | --- |
| `pangu/src/main.cc` | 程序入口 |
| `pangu/src/task_list/ideal_grmhd.cc` | GRMHD task graph |
| `pangu/src/initialization/package_registration.cc` | runtime package 与字段注册 |
| `pangu/src/initialization/variable_mnemonics.h` | primitive/conservative 索引 |
| `pangu/src/physics/two_temperature.h` | 双温模型、模型枚举与加热公式 |
| `pangu/src/riemann_solver/riemann_solver.cc` | HLL/HLLD/LAXF 通量分发 |
| `pangu/src/riemann_solver/hlld.cc` | HLLD Riemann solver |
| `pangu/src/recovery/invertor.cc` | 新的 conservative-to-primitive (C2P) 恢复 |
| `pangu/src/riemann_solver/electron_heating.cc` | 每步电子加热更新 |
| `pangu/src/fixer/primitive_fixer.cc` | primitive floor 与电子熵限制 |
| `pangu/problem/fm_torus/problem_generator.cpp` | FM torus 初始化 |

开发双温功能时应特别注意：

- `electron_entropy` 是随密度通量输运的被动标量。
- `entropy` 用于估计流体耗散，不是单独的热力学输出装饰量。
- 温比限制使用 `ratio_min` 和 `ratio_max`，不要改名为 KHARMA 的 `tp_over_te_*`。
- 目前 Pangu 每次只运行一个电子加热模型；如需 KHARMA 式多模型并行，必须扩展变量布局、通量、fixer、输出和初始化，不应只改 inputfile。
- `riemann_solver` 当前支持 `laxf`、`hll`、`hlld` 三种通量格式；如果要切换默认值或增加新 solver，需要同步更新 `package_registration.cc`、`riemann_solver/riemann_solver.cc` 和相关输入文件。

## 常见问题

| 问题 | 原因 | 处理 |
| --- | --- | --- |
| `Problem source not found` | `PROBLEM` 与 `pangu/problem` 下目录不匹配 | 检查目录名并重新构建 |
| 找不到 `pangu.host` 或 `pangu.cuda` | 构建失败或 `BUILD_DIR` 不一致 | 重新运行 `make.sh` |
| 没有 PHDF 输出 | 输出目录错误或 inputfile 中输出周期过大 | 检查 `data/<problem>` 和 `<parthenon/output*>` |
| `--2t` 报缺字段 | 输出文件中没有 `density`、`entropy` 或 `electron_entropy` | 在 inputfile 的 output 变量中加入这些字段 |
| 温度图数值看起来比代码温度大 `~1e13` | 图中单位是 Kelvin，代码温度是 `Theta` | 这是预期行为 |
| `model` 参数无效 | 不是支持的模型名 | 使用 `constant/howes/kawazura/werner/rowan/sharma` |

## 复现实验建议

保存以下信息：

- 当前代码版本或补丁
- `.pangu_build.env`
- 完整 inputfile
- 运行命令
- 输出目录
- 分析命令和脚本版本

这样可以明确区分物理设置变化、构建变化和后处理变化。
