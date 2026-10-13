# AthenaK 与 PANGU：普通演化步 GPU 执行对比

日期：2026-10-13。目的：定位速度差距，不修改演化实现。

## 结论及范围

本地统一参数的固定网格测试表明，PANGU 的普通步约慢 1.42–1.55 倍。
主要可测差距是跨 MeshBlock 边界打包和解包，双边同步插桩显示约差
4.84–5.07 倍；RHS 两组测量基本持平，不能把主要差距归因于 RHS
的数据视图或“抄了算法但计算慢”。

本结论针对 RTX 3060 Laptop 上的固定网格，不是服务器 484-block
AMR 算例的完整归因。尚未定量拆分边界内的索引、坐标变换、所有权掩码、
内存布局及调度粒度各自的贡献，也不能由此判定 Parthenon 存在不可修复的硬伤。

## 对照条件

- 同一 GPU，单 MPI rank，交替串行运行，无并发 GPU benchmark。
- 两边 Release、`-O3 -DNDEBUG`、CUDA `sm_86`；不是 O2/O3 对比。
- 总网格 64³，MeshBlock 16³，64 个块，ghost 宽度 2，二阶差分。
- 四阶段 RK4，40 步，CFL 0.2，dt=0.0125。
- 静止单位质量 puncture，区域 [-2,2]³，outflow。
- dissipation=0.1、chi_div_floor=1e-5、kappa1=0.02、kappa2=0。
- slow-start lapse 开启，幅度 0.6、时间 20、指数 1；其他 gauge 参数一致。
- 无 AMR；关闭 tracker、horizon、waveform 与场文件输出。
- 对齐的是主演化输入、总网格和步数，不是所有派生任务：AthenaK 仍执行
  每步 ADM constraints（含初始化共 41 次，组1累计 0.233812 秒），PANGU
  不在普通步图中执行该诊断。AthenaK 在额外承担该工作的情况下仍更快。
- 未在本报告中证明两代码最终场逐元素一致。

测试 PANGU 是保存的既有优化基线，不包含最近失败的描述缓存、线程扫描、
索引解码实验，也不包含单 rank fence 跳过实验。

可执行文件 SHA256：

```
AthenaK c814319427d1cde064a70cf21269fb075a85d815973ae9d3c3b09791b5d15988
PANGU   f869cad5238afa3e4217f7761319d8ece7b47e386c205f3e5683093223eed76d
```

## 未插桩普通步

取 cycle 4–39，共 36 步，排除启动步，不使用两框架不同口径的最终
cpu/walltime 指标。AthenaK 使用相邻 elapsed 差；PANGU 使用 wsec_step。

| 交替测试轮次 | AthenaK，36 步秒数 | PANGU，36 步秒数 | PANGU/AthenaK |
|---|---:|---:|---:|
| 1 | 2.3894554 | 3.7069 | 1.551 |
| 2 | 2.7936560 | 4.0570 | 1.452 |
| 3 | 2.7712048 | 3.9410 | 1.422 |

笔记本 GPU 存在时钟、温度和调度波动，所以不把单轮小差异当成优化收益。

## 双边分项

Kokkos Tools 回调要求全局 fencing，使每个 parallel_for/reduce 在返回回调前
完成，以同步墙钟计时归属到 kernel。此方法扰动原执行，以下值仅用于定位，
不能与上表直接相加或算端到端收益。第二组反转两代码运行顺序。

| 项目，40 步累计秒数 | AthenaK 组1 | PANGU 组1 | AthenaK 组2 | PANGU 组2 |
|---|---:|---:|---:|---:|
| RHS，160 次 | 1.881174 | 1.979073 | 1.754784 | 1.707740 |
| K-O，160 次 | 0.458100 | 0.386217 | 0.416629 | 0.328206 |
| 边界打包 | 0.164630 | 0.851232 | 0.158149 | 0.860212 |
| 边界解包 | 0.161426 | 0.727182 | 0.156675 | 0.734770 |
| 边界合计 | 0.326056 | 1.578414 | 0.314824 | 1.594983 |

边界计数含启动交换：AthenaK SendBuff/RecvBuff 各 322 次，PANGU
Send/Set 各 162 次。这不是仅有“更多 kernel launch”的问题：PANGU
launch 数更少，合计反而更慢。RHS 组1慢 5.2%，组2快 2.7%，没有稳定大差距。

## 单 MeshBlock 对照

总物理网格和参数保持 64³，仅把 MeshBlock 从 16³ 改成 64³，消除跨块交换。
块尺寸变化也会改变 kernel 粒度，因此此对照不是单独隔离通信的完美实验，
需要与上面的分项证据一起解释。

| 轮次，36 普通步 | AthenaK 秒 | PANGU 秒 | PANGU/AthenaK |
|---|---:|---:|---:|
| 1 | 2.6294407 | 2.6754 | 1.017 |
| 2 | 2.8934515 | 2.9308 | 1.013 |
| 3 | 4.8486750 | 2.7932 | 0.576 |

第三轮出现明显运行波动，不隐去，也不据此宣称 PANGU 比 AthenaK 快。
前两轮与分项测量共同支持：多块测试中的主要额外耗时在边界路径。

## Nsight Compute 补充与源码差异

过滤双边打包函数，skip 2、count 2，采集 LaunchStats、SpeedOfLight、
InstructionStats。AthenaK 两个样本是同函数内不同 launch，不能把单个样本
当成完全对应的单次 PANGU 交换；以整段双边计时作为主要比较依据。

| 打包样本 | Duration ms | CUDA block 线程数 | grid blocks | 每线程寄存器 |
|---|---:|---:|---:|---:|
| AthenaK 0 | 0.856128 | 128 | 78848 | 45 |
| AthenaK 1 | 0.424928 | 128 | 78848 | 42 |
| PANGU 0 | 4.899840 | 128 | 936 | 40 |
| PANGU 1 | 6.107072 | 128 | 936 | 40 |

不能解释为 PANGU 打包寄存器更多，实测恰好更少。相同 block 线程数不代表
相同工作划分：AthenaK 以 block×neighbor×variable 展开 league；Parthenon
以 boundary buffer 展开，线程内部还处理通用拓扑与多维索引。

源码中已确认的区别：

- AthenaK `src/bvals/bvals_cc.cpp`：专用发送/接收范围，按变量展开团队；
  同 rank 写接收缓冲区，仍需解包，不是直接写目标 ghost 区。
- Parthenon `src/bvals/comms/boundary_communication.cpp`：通用 BndInfo、
  六维索引、接收坐标变换和 ownership mask；同 rank 缓冲区共享，不能
  声称这里额外发生了一次 host memcpy。
- AthenaK 的主变量是连续五维数组；PANGU 使用变量 pack。但 RHS 实测
  基本持平，所以不能仅凭这个源码差异解释本次主要性能差距。

以上说明 task graph 对齐没有对齐边界 kernel 内部的数据组织和线程粒度。
要进一步解释每一项成本，需要针对边界做双边硬件指标或控制变量实验；
目前不把“通用性”作为一个不可检验的总括原因。后续运行时检查发现了
更关键的内存位置差异，见文末；不能把最初的边界时间差简单归于通用索引。

## 原始证据

本机目录：`/tmp/pangu-athenak-compare-Gm626l/`。

- 输入：`athena.in`、`pangu.in`、`one-athena.in`、`one-pangu.in`。
- 日志：`athena/`、`pangu/`、`one-athena/`、`one-pangu/` 下 trial*.log。
- 分项：athena 与 pangu 的 `profile.log`、`profile2.log`。
- 硬件指标：`athena-ncu.csv`、`pangu-ncu.csv`。
- 工具源码：`/tmp/pangu-fence-verify-CdHA45/kernel_timing.cc`。

本轮未修改演化代码，未提交或推送。

## 后续实验：按分量增加边界团队数

在上述定位后，只改 Parthenon 边界工作划分：稠密字段的每缓冲区团队数
取现有最小团队规则与字段分量数的较大者，稀疏路径不变。沿用原生
SplitFlatIndexRangeAmongTeams，通信格式、所有权掩码和粗细网格插值不变。
这并非绕过 Parthenon；原生 minimum_number_of_teams_for_boundary_kernel
也提供提高团队数的能力，本改动让团队数随字段分量自动选择。

同一 GPU 的普通步中位数（秒/步）：

| 轮次 | 优化前 | 优化后 | 用时减少 |
|---|---:|---:|---:|
| 1 | 0.1590 | 0.1450 | 8.8% |
| 2 | 0.1560 | 0.1515 | 2.9% |
| 3 | 0.1640 | 0.1495 | 8.8% |

该轮 GPU 时钟降至约 900 MHz，绝对值不能与前面的双边测量直接比较。
普通步表的优化前是原既有基线，优化后还包含单 rank fence 跳过。
下面额外使用已跳过 fence 的旧二进制进行分项对照，隔离团队数改动：

| 插桩累计秒数 | 优化前 | 优化后 |
|---|---:|---:|
| RHS（未改） | 3.565402 | 3.531309 |
| 边界发送 | 0.957280 | 0.740624 |
| 边界接收设置 | 0.787329 | 0.654560 |
| 边界合计 | 1.744609 | 1.395184 |

边界合计用时降低 20.0%，RHS 相差不足 1%。
固定 64 块和启动混合粗细网格后删除 56 块的 AMR 测试中，最终
nr.z4c 均逐元素一致、最大差值 0。尚未验证多 rank 或服务器 484 块收益。
改动暂留本地，未提交。证据位于同一临时目录的 component-* 子目录。

## 关键补充：同 rank 边界缓冲区实际在主机内存

进一步模仿 AthenaK 的固定分量、按空间行复制后，普通步从约 0.154 秒
退化到约 0.369 秒。插桩定位新增时间在 Send（累计 9.270818 秒），
不是 RHS 或 Set。该实验已撤回，保留此前团队展开版本。

随后核查编译配置和运行时内存：

1. 当前基准的 PANGU 开启 `PARTHENON_ENABLE_HOST_COMM_BUFFERS=ON`。
2. `parthenon/src/kokkos_types.hpp` 将 BufMemSpace 全局定义为
   `Kokkos::CudaHostPinnedSpace`，不是仅为跨 rank 缓冲区定义。
3. `build_boundary_buffers.cpp` 使用同一 `buf_pool_t<Real>` 建立边界池，
   同 rank 通信也使用该池。
4. 在单 rank 的实际运行中，通过 Kokkos allocation 回调取得池指针，
   `cudaPointerGetAttributes` 返回 status=0、type=1；CUDA 头文件定义
   type=1 为 `cudaMemoryTypeHost`。

实际日志示例：

```
CudaHostPinned  pool buffer of size 1323 x 288  3048192 bytes  type=1 status=0
CudaHostPinned  pool buffer of size 189 x 432   653184 bytes   type=1 status=0
CudaHostPinned  pool buffer of size 2646 x 288  6096384 bytes  type=1 status=0
```

因此前面的双边对比并非仅有索引和线程粒度不同：PANGU 的边界 kernel
读写主机锁页内存，而 AthenaK 本地接收缓冲区在 GPU 上。这种 kernel 内
的直接主机内存访问不一定表现为单独 cudaMemcpy 调用。

此机制已经由源码和指针属性证实，但尚未量化它占边界差距的比例。
本地将该选项切到 OFF，开始重编译；保持其他参数和团队展开实现不变，
待 GPU 显存缓冲区版本完成后做配对测速和字段一致性验证。
只先验证单 rank。多 rank 直接向 MPI 提交 GPU 指针需要 GPU-aware MPI，
不能无条件建议服务器关闭 host buffers。

运行时原始证据：
`/tmp/pangu-fence-verify-CdHA45/memory-check/run.log`。

## 显存缓冲区对照已完成

本地关闭 HOST_COMM_BUFFERS 后重编译完成，其他 Release/CUDA 架构选项不变，
仍使用上一版团队展开实现，未加入已撤回的逐行复制路径。
实际边界池指针属性变为 `Cuda`、type=2（cudaMemoryTypeDevice）、status=0。

GPU-buffer 可执行文件 SHA256：
`2ee924f1018e5917fb5ae0681c83795252881358537b6027fa647334e0481d3e`。

与已保存的 host-buffer 团队展开版本对照：

| 同步插桩累计秒数，40 步 | Host buffers | Device buffers |
|---|---:|---:|
| RHS，160 次 | 3.531309 | 3.552188 |
| Send | 0.740624 | 0.105377 |
| Set | 0.654560 | 0.135539 |
| 边界合计 | 1.395184 | 0.240916 |

边界用时减少 82.7%，RHS 变化约 0.6%。这支持缓冲区内存位置是本次边界
瓶颈的实质原因，而不只是 task graph 或 Metadata 通用性。

三方未插桩测试改变执行顺序：轮1 Host→Device→AthenaK，轮2
AthenaK→Device→Host，轮3 Device→Host→AthenaK。普通步仍取 cycle 4–39。

| 轮次 | Host，步中位数秒 | Device，步中位数秒 | AthenaK，步中位数秒 |
|---|---:|---:|---:|
| 1 | 0.1450 | 0.1200 | 0.132125 |
| 2 | 0.1725 | 0.1200 | 0.1293225 |
| 3 | 0.1450 | 0.1215 | 0.1337025 |

| 轮次 | Host，36 步累计秒 | Device，36 步累计秒 | AthenaK，36 步累计秒 |
|---|---:|---:|---:|
| 1 | 5.245 | 4.348 | 4.7948832 |
| 2 | 6.156 | 4.305 | 4.6666820 |
| 3 | 6.915 | 4.652 | 4.9543189 |

该本地固定网格测试中，PANGU 的普通步不再慢于 AthenaK；不能扩展成整个
项目更快的结论。AthenaK 仍额外运行每步 constraints，两边派生任务不同。
各轮 GPU 状态记录在对应 trial*.gpu.txt，笔记本运行波动仍存在。

数值验证：固定 64 块及 AMR 初始混合网格删除 56 块的测试中，最终
nr.z4c 与 Host 基线逐元素完全一致，最大绝对差 0。

用户另反馈：服务器去掉同一选项后从约 193 秒降到 128 秒；若其他条件
不变，相当于 1.51 倍加速、用时减少 33.7%。这是用户报告，尚未读取
服务器本轮完整日志。与 AthenaK 78 秒相比，服务器仍有剩余差距；达到
其速度 90% 对应时间约 86.7 秒，不能宣称服务器目标已经完成。

原始证据：

- 内存和 GPU-buffer 分项：
  `/tmp/pangu-fence-verify-CdHA45/device-memory-check/run.log`。
- 三方测速：比较目录下 host-control、device-control、athena-control。
- 字段验证：比较目录下 device-verify、device-remesh。

本地编译缓存的 HOST_COMM_BUFFERS 已为 OFF；未无条件修改多 rank 部署策略。
本轮未提交或推送。

## 显存缓冲区重新选择工作粒度

为避免沿用 host-buffer 结论，使用同一测试二进制扫描每缓冲区 1、2、4、
8、16、22、32 个团队。正向与反向执行两轮。64 块单层网格的普通步差别
较小且受波动影响，未据此选择某个中间档位。

补充 232 块固定四级细化网格：总根网格 32³、块 16³、域 [-32,32]³、
中心 [-4,4]³ 细化到物理 level 4。AthenaK 打印层级块数 56/56/56/64。
双方 40 步、dt=0.025、关闭输出和 tracker，其余主演化参数沿用前述对照。

| 普通步 cycle 4–39 | 步中位数秒 | 36 步累计秒 |
|---|---:|---:|
| AthenaK | 0.484975 | 17.554611 |
| PANGU，1 团队 | 0.4600 | 18.906 |
| PANGU，22 团队 | 0.4675 | 19.181 |

累计值包含 PANGU 的明显单步尖峰，不能只展示较好的中位数。该对照是
固定多层网格，不包含动态 AMR 的创建、销毁和 tracker 成本。

| 同步插桩累计秒数 | 1 团队 | 22 团队 |
|---|---:|---:|
| RHS | 11.548714 | 11.513849 |
| Send | 0.238706 | 0.388462 |
| Set | 0.388293 | 0.482759 |
| Send + Set | 0.626999 | 0.871220 |
| 物理边界 GenericBC（31,104 次） | 1.532369 | 1.574619 |
| ProlongationRestrictionLoop（830 次） | 2.224071 | 2.245226 |

因此显存缓冲区恢复原生团队规则，不再自动按分量提高团队数；host-buffer
分支保留此前实测有效的分量展开。最终代码不保留扫描用的环境变量开关。
后续批量处理多层物理边界有明确的 kernel 数量依据，但本轮尚未实现，
也不能把同步插桩的每个小 kernel 耗时直接当成原运行可节省时间。

还有版本一致性问题：当前 GitHub main 的 package.cc 仍使用 32 块重型
kernel 批次，且 tasks.cc 保留普通步 ComputeConstraintsMeshTask。本地
测试版本使用 2048 块批次、per-block pack 缓存并移除普通步该诊断。
这些 root 仓库修改仍未提交，不能把本地测试当作已发布 main 的性能。
服务器当前是否包含这些修改尚待提交号和完整日志核实。

证据目录：device-team-sweep、multilevel-athena、multilevel-pangu-1、
multilevel-pangu-22；多层输入为 multilevel-athena.in 和 multilevel-pangu.in。

最终显存原生团队规则版本已重新编译，并通过固定网格及 AMR 删除 56 块
的字段对比：nr.z4c 均逐元素一致，最大差值 0。原始日志和结果位于
native-device-verify、native-device-remesh。未提交或推送。

### 多层物理边界 MeshData 批处理：正确性检查

fine/coarse 均改为分方向的 MeshData 批量边界 kernel，适用范围仅限三维
真空 NR、六面 nr_outflow、无 Fine FillGhost 字段；其他路径继续使用原实现。
第一版检查失败，不能算有效优化。定位到旧版 MeshData 的 variable-pack
缓存键未区分 coarse/fine；同一容器交替取两种 pack 会复用错误视图。
当前 coarse pack 使用独立浅层 MeshData 容器隔离缓存，共享原变量和
coarse 存储，不复制字段。AMR 的 RedistributeAndRefineMeshBlocks 会
mesh_data.clear()，因此新拓扑不会继续使用旧容器。

重新编译后与优化前程序比较最终 nr.z4c：

| 检查 | 最终块数 | 最大绝对差值 | 不同元素数 |
|---|---:|---:|---:|
| 固定四级加密，8 步 | 232 | 0 | 0 |
| 加密区接触物理边界，8 步 | 92 | 0 | 0 |
| AMR 重网格删除 56 块，20 步 | 64 | 0 | 0 |

所有结果有限。证据为 packed-multilevel-cache-fixed、packed-face-cache-fixed、
packed-remesh-cache-fixed；优化前对照为 packed-multilevel-before、
packed-face-before 和 remesh-baseline。普通步交替计时正在进行；尚不宣称
服务器加速比例，尚未提交或推送。

普通步 cycle 4–39，顺序为 before1、after1、after2、before2：

| 版本 | 步中位数秒 | 36 步累计秒 |
|---|---:|---:|
| before1 | 0.4925 | 17.394 |
| after1 | 0.5015 | 18.725 |
| after2 | 0.5035 | 18.025 |
| before2 | 0.5000 | 18.229 |

after1 包含 1.4 秒尖峰，均未剔除。未插桩运行没有确认加速。
同一时段插桩：旧版 GenericBC 31,104 次、1.575422 秒；新版剩余
GenericBC 384 次、0.021978 秒，加上六方向 packed kernel 共 1,920 次、
0.862740 秒。RHS 旧/新 12.471301/12.414313 秒，prolongation/restriction
2.382713/2.385965 秒，基本相同。同步插桩夸大启动成本，不能将这约
0.69 秒差值当作未插桩收益。因此已恢复 multilevel 路径使用原物理边界，
不默认启用本轮批处理。保留缓存隔离实现供后续改进，但没有上线收益声明。
证据目录 packed-timing。

### Z4c 粗细网格算子的编译期 stencil 特化

代码差异：AthenaK mesh/prolongation.hpp 的 ProlongInterpolation<NGHOST>
及 mesh/restriction.hpp 的 RestrictInterpolation<NGHOST> 使用编译期
NGHOST；PANGU amr_refinement.h 原先从 ib.s 运行时决定 points、radius
及权重。正在验证将常用 ghost=2/4 分派到常量大小 stencil 的版本，其他
宽度保留旧路径。公式、权重、累加次序、越界位置的平均限制回退不变。
这属于 PANGU 的 NR 算子优化，不修改 Parthenon 通用接口，也不降低精度。

另已检查普通 RK 图：vacuum 的 current_boundary/next_boundary 使用仅
nr.z4c 的浅层 MeshData。因此关闭未启用 Weyl 的 FillGhost 不能减少
这条普通步热路径的通信，未实施该无依据改动。

本轮检查通过：232 块固定多层网格、8 步，ghost=2/FD2 与 ghost=4/FD4
分别对照旧版；AMR 删除 56 块、20 步，最终 64 块。三者 nr.z4c 最大
绝对差值均为 0，不同元素数 0，所有元素有限。

232 块、ghost=2/FD2、无输出、无 AMR，40 步的交替计时：

| 顺序 | cycle 4–39 步中位数秒 | 36 步累计秒 |
|---|---:|---:|
| before1 | 0.487 | 18.039 |
| after1 | 0.467 | 16.905 |
| after2 | 0.466 | 16.808 |
| before2 | 0.498 | 17.909 |

旧版 before1 的 0.932 秒尖峰未剔除。两轮中位数平均时间下降 5.28%，
两轮累计时间合计下降 6.22%；这是局部同机固定多层网格结果，非服务器
端到端时间。GPU 采样为 900 MHz，有温度/功耗波动，仍需服务器复测。

同一时段同步插桩的粗细网格 ProlongationRestrictionLoop 均 830 次：
旧 2.420129 秒，新 1.471879 秒，下降 39.18%。RHS 旧/新为
12.469823/12.477412 秒，GenericBC 为 1.763848/1.766022 秒，基本不变。
因此收益定位于 stencil 算子，而非输出、RHS 或物理边界。保留此改动，
未提交或推送。输入仍为 multilevel-pangu.in；证据目录 stencil-timing、
stencil-g2-after、stencil-g4-before、stencil-g4-after、stencil-remesh-after。
