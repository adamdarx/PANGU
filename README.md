# Pangu

Pangu 是一个基于 Parthenon + Kokkos 的高性能数值模拟代码框架，面向 SR/GRMHD 类问题，支持 CPU/GPU 多后端、任务图驱动时间推进、模块化问题初始化与标准化后处理流程。

本文档按“由浅入深”的顺序组织：先帮助你 5 分钟跑起来，再解释目录、脚本和核心执行流程，最后给出开发与常见问题。

## 目录

1. 项目定位
2. 快速开始
3. 环境与依赖
4. 仓库结构
5. 构建与运行（脚本化）
6. 输出与可视化
7. 输入文件说明
8. 核心执行流程
9. 新增问题指南
10. 常见问题
11. 许可证

## 1. 项目定位

Pangu 关注以下目标：

- 在统一代码框架下快速切换不同物理问题。
- 复用 Parthenon 的网格、并行、I/O 与驱动机制。
- 通过 Kokkos 提供跨 CPU/GPU 的可移植并行实现。
- 保持求解流程清晰：守恒更新、恢复、修正、边界、时间步估计。

核心能力概览：

- 任务图驱动的多阶段推进（MultiStageDriver）。
- 模块化物理组件（flux/recovery/fixer/metric/mesh）。
- 按问题目录选择初始化器（ProblemGenerator）。
- 标准化脚本入口：构建、运行、分析分离。

## 2. 快速开始

下面以 shock_tube 为例给出最短路径。

### 2.1 一次构建

```bash
cd /path/to/Pangu
PROBLEM=shock_tube ENABLE_CUDA=ON ENABLE_OPENMP=ON ./scripts/shell/make.sh
```

### 2.2 一次运行

```bash
./scripts/shell/run.sh -p shock_tube
```

运行输出会写入：

- [data/shock_tube](data/shock_tube)

### 2.3 一次出图

```bash
./scripts/shell/analyze.sh -p shock_tube -f Density --savename contour_density.png
```

图片会写入：

- [pic/shock_tube](pic/shock_tube)

## 3. 环境与依赖

建议环境：

- CMake >= 3.10
- 支持 C++17 的编译器
- Python3（用于后处理脚本）
- MPI（可选，用于多进程运行）
- CUDA + 对应工具链（可选，用于 GPU）

说明：

- Parthenon 以子模块形式包含在仓库中，通常无需额外安装。
- HDF5 相关能力通过 Parthenon 构建链路提供。

## 4. 仓库结构

### 4.1 顶层

- [CMakeLists.txt](CMakeLists.txt)：顶层构建入口。
- [pangu](pangu)：Pangu 主代码与问题目录。
- [parthenon](parthenon)：上游 Parthenon 子模块。
- [scripts/shell](scripts/shell)：脚本入口（构建/运行/分析）。
- [data](data)：数值输出数据目录。
- [pic](pic)：可视化图片输出目录。

### 4.2 关键源码子目录

- [pangu/src/main.cpp](pangu/src/main.cpp)：程序入口，注册包与驱动。
- [pangu/src/simulator](pangu/src/simulator)：时间推进与任务图调度。
- [pangu/src/initialize](pangu/src/initialize)：状态包与变量注册。
- [pangu/src/flux](pangu/src/flux)：通量与守恒更新。
- [pangu/src/recovery](pangu/src/recovery)：守恒到原始量恢复。
- [pangu/src/fixer](pangu/src/fixer)：恢复后修正与稳定化。
- [pangu/src/metric](pangu/src/metric)：度规相关计算。
- [pangu/problem](pangu/problem)：问题目录（每个问题独立初始化与输入）。

### 4.3 问题目录约定

每个问题目录通常包含：

- [ProblemGenerator.cpp](pangu/problem/shock_tube/ProblemGenerator.cpp)
- [inputfile](pangu/problem/shock_tube/inputfile)

示例问题（非完整列表）：

- SR/MHD：shock_tube、kh、kelvin_helmholtz、orszag_tang、blast。
- GR：bondi、gr_torus、fm_torus、gr_analytic。
- BNS/特殊初始化：elliptica_bns、lorene_bns。

## 5. 构建与运行（脚本化）

脚本都位于 [scripts/shell](scripts/shell)。

### 5.1 构建脚本 make.sh

文件： [scripts/shell/make.sh](scripts/shell/make.sh)

默认行为：

- `ENABLE_OPENMP=ON`
- `ENABLE_CUDA=ON`
- `BUILD_DIR=build`
- `BUILD_TYPE=Release`
- `PROBLEM=BrioWuShocktube`

常见用法：

```bash
# 默认构建
./scripts/shell/make.sh

# CPU-only
ENABLE_CUDA=OFF ./scripts/shell/make.sh

# 指定问题 + Debug
PROBLEM=shock_tube BUILD_TYPE=Debug ./scripts/shell/make.sh

# 指定构建目录
PROBLEM=shock_tube BUILD_DIR=build_shock ./scripts/shell/make.sh
```

构建成功后会生成：

- `.pangu_build.env`（供运行脚本读取）

可执行文件：

- CUDA 开启：`pangu.cuda`
- CUDA 关闭：`pangu.host`

路径示例：

- [build/pangu/src](build/pangu/src)

### 5.2 运行脚本 run.sh

文件： [scripts/shell/run.sh](scripts/shell/run.sh)

默认行为：

- 若存在 `.pangu_build.env`，自动复用上次构建配置。
- 默认输入：`pangu/problem/<PROBLEM>/inputfile`。
- 运行目录：`data/<问题名>`。

常见用法：

```bash
# 使用上次构建结果
./scripts/shell/run.sh

# 指定问题
./scripts/shell/run.sh -p shock_tube

# 指定输入文件
./scripts/shell/run.sh -i pangu/problem/shock_tube/inputfile

# MPI 运行
./scripts/shell/run.sh -p shock_tube -n 4
```

### 5.3 分析脚本 analyze.sh

文件： [scripts/shell/analyze.sh](scripts/shell/analyze.sh)

作用：

- 从 `data/<problem>` 读取 `.phdf`。
- 调用 Parthenon 的 contour1d.py 生成 1D 时空图。
- 输出到 `pic/<problem>`。

常见用法：

```bash
# 指定问题与字段
./scripts/shell/analyze.sh -p shock_tube -f Density

# 自定义输出文件名
./scripts/shell/analyze.sh -p shock_tube -f Density --savename contour_density.png

# 在 data/<problem> 目录内可省略 -p
cd data/shock_tube
/path/to/Pangu/scripts/shell/analyze.sh -f Density
```

## 6. 输出与可视化

### 6.1 运行输出

默认输出为 `.phdf` 与对应 `.xdmf`（视输入配置而定）。

推荐规则：

- 数据只放 [data](data)
- 图片只放 [pic](pic)

### 6.2 contour1d.py 参考

脚本位置：

- [parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/contour1d.py](parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/contour1d.py)

直接调用示例：

```bash
cd data/shock_tube
python3 /path/to/Pangu/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/contour1d.py \
  --workers 4 \
  --savename contour_density.png \
  --colorbar Density \
  Density *.phdf
```

## 7. 输入文件说明

输入文件采用 Parthenon 风格分段，常见段包括：

- `<parthenon/job>`：任务标识。
- `<parthenon/mesh>`：网格、边界与层级参数。
- `<parthenon/meshblock>`：块尺寸。
- `<parthenon/time>`：时间推进控制（`dt_init`、`tlim`、`nlim`）。
- `<PANGU>`：Pangu 相关参数（如 `AdiabaticIndex`）。
- `<parthenon/output*>`：输出格式、频率、变量列表。

建议：

- 新问题优先使用 `<PANGU>` 统一管理本代码参数。

## 8. 核心执行流程

每个推进阶段大致包括：

1. 边界通信与必要同步。
2. 通量与守恒量更新。
3. 约束输运（若启用）。
4. 恢复（Conservative -> Primitive）。
5. 修正与稳定化。
6. 边界处理与派生量刷新。
7. 时间步估计与 AMR 标记。

对应实现主干位于 [pangu/src/simulator](pangu/src/simulator)。

## 9. 新增问题指南

新增问题的最小步骤：

1. 在 [pangu/problem](pangu/problem) 下创建新目录。
2. 新增 ProblemGenerator.cpp 与 inputfile。
3. 构建时指定 `-DPROBLEM=<新目录名>`。
4. 使用 run.sh 运行并验证输出。

该模式下每个问题目录独立维护，无需额外注册分发表。

## 10. 常见问题

### 10.1 配置阶段报 PROBLEM 未设置

请显式指定：

```bash
cmake -S . -B build -DPROBLEM=shock_tube
```

### 10.2 运行时报找不到可执行文件

检查：

- 构建是否成功。
- `build` 目录是否一致。
- 当前后端应使用 `pangu.host` 还是 `pangu.cuda`。

### 10.3 数据输出过大

可通过输入文件减少输出频率或变量数量。

## 11. 许可证

本项目采用仓库内许可证约束，详见 [LICENSE](LICENSE)。