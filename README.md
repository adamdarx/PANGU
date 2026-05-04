# Pangu

Pangu 是一个基于 Parthenon 与 Kokkos 的 GRMHD 数值模拟代码。当前代码支持按问题目录选择初始化器、CPU/CUDA 后端构建、Kerr 或 Minkowski 度规下的算例，以及单模型双温电子加热。

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
./scripts/shell/make.sh
```

常用 CPU/GRMHD 构建示例：

```bash
ENABLE_CUDA=OFF PROBLEM=fm_torus BUILD_DIR=build ./scripts/shell/make.sh
```

Minkowski 度规测试问题构建示例：

```bash
ENABLE_CUDA=OFF PROBLEM=orszag_tang_vortex BUILD_DIR=build ./scripts/shell/make.sh
```

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ENABLE_OPENMP` | `ON` | 是否启用 OpenMP |
| `ENABLE_CUDA` | `ON` | 是否构建 CUDA 目标 |
| `PROBLEM` | `shock_tube` | 选择 `pangu/problem/<PROBLEM>` |
| `BUILD_DIR` | `build` | 构建目录 |
| `BUILD_TYPE` | `Release` | CMake 构建类型 |
| `BUILD_JOBS` | `4` | 并行编译任务数 |
| `KOKKOS_ARCH` | 空 | Kokkos 架构选项 |
| `CMAKE_EXTRA_ARGS` | 空 | 透传给 CMake 的额外参数 |

构建产物：

| 后端 | 可执行文件 |
| --- | --- |
| CPU | `build/pangu/src/pangu.host` |
| CUDA | `build/pangu/src/pangu.cuda` |

构建脚本会写出 `.pangu_build.env`，记录上次构建的关键配置，便于复现实验。

## 运行

运行入口：

```bash
./scripts/shell/execute.sh
```

示例：

```bash
BUILD_DIR=build ENABLE_CUDA=OFF PROBLEM=fm_torus ./scripts/shell/execute.sh -n 1
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

`<core>` 控制单流体 GRMHD 主方程参数；`<electrons>` 控制双温电子加热；`<metric>` 与具体问题段控制 GR 度规和初始化。

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
./scripts/shell/analyze.sh
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
./scripts/shell/analyze.sh -p fm_torus -f density --xzplot
```

双温温度图：

```bash
./scripts/shell/analyze.sh -p fm_torus --2t -w 4
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
PROBLEM=<name> ./scripts/shell/make.sh
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
| `pangu/src/riemann_solver/electron_heating.cc` | 每步电子加热更新 |
| `pangu/src/fixer/primitive_fixer.cc` | primitive floor 与电子熵限制 |
| `pangu/problem/fm_torus/problem_generator.cpp` | FM torus 初始化 |

开发双温功能时应特别注意：

- `electron_entropy` 是随密度通量输运的被动标量。
- `entropy` 用于估计流体耗散，不是单独的热力学输出装饰量。
- 温比限制使用 `ratio_min` 和 `ratio_max`，不要改名为 KHARMA 的 `tp_over_te_*`。
- 目前 Pangu 每次只运行一个电子加热模型；如需 KHARMA 式多模型并行，必须扩展变量布局、通量、fixer、输出和初始化，不应只改 inputfile。

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
