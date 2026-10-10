# 七任务电子加热运行包

| CASE | 初始磁场 | a* |
|---|---|---:|
| sane_a0 | SANE | 0 |
| sane_a05 | SANE | 0.5 |
| sane_a09 | SANE | 0.9 |
| sane_a09375 | SANE | 0.9375 |
| mad_am09375 | MAD seed | -0.9375 |
| mad_a0 | MAD seed | 0 |
| mad_ap09375 | MAD seed | 0.9375 |

每个任务同时演化 Constant/Howes/Kawazura/Werner/Rowan/Sharma 六组电子变量。
共同设置：384×192×192；两个径向 192×192×192 MeshBlock；FP64；MKS/static/wave；
2 GPU、2 MPI rank、每 rank 1 CPU；GPU80G、QOS=low、单段 120 小时；
终点 30000M，场、history、restart 每 100M 调度输出。
`electrons.prim` 中含 Ktot 和六模型电子熵；`electrons.diagnostics` 只针对首个模型 Constant，
不能当作六模型全部加热诊断。

## 本次统一物理初态

七组都采用 r_edge=20、r_peak=41、r_out=1000M，r_in=1.2M，gamma=4/3，hslope=0.3。
之前 r_edge=6、r_peak=12 在负自旋逆行算例不构成有效 FM torus；已由实际初始化测试确认。
因此七组统一换为较大 torus，不能接续旧 0.9375 的小 torus restart。本包从新目录起跑。
`prograde=true` 保持圆盘正向旋转，a*=-0.9375 是相对黑洞逆行，而非同时反转圆盘。

`magnetic_topology=mad` 使用扩展 poloidal-loop seed：
`Aφ=max[(ρ/ρmax)(r/r_edge)^3 sin³θ exp(-r/400)-0.2,0]`。
SANE 使用原密度等值线 loop。二者均经离散旋度产生面磁场，并按全局
`max(pgas)/max(b²/2)=100` 归一化。注意这不是逐点 beta 的最小值。
MAD 是否实际形成要在长期输出中检查视界磁通饱和与吸积受阻；本包不把初始化标签当成已形成 MAD 的证明。

## 更新及构建

先按交付消息提供的 Gitee 提交更新 PANGU；不改 Parthenon 子模块、不需要删除构建目录。
解压在 `/lustre/home/2601110206/`，与 `PANGU/` 并列：

```bash
tar -xzf electron_mks_7tasks.tar.gz
cd electron_mks_7tasks
bash scripts/build.sh
```

固定构建路径：`PANGU/build-electron-mks`。脚本加载 gcc/12.2.0、cuda/12.6.0、cmake/3.31.9，
使用 nvcc_wrapper、mpicc、MPI、HDF5_PREFER_PARALLEL、host communication buffers、AMPERE80、Release、-j3。
构建过程不执行 GPU 程序。已有匹配的 GPU 构建可以增量编译；如果此前构建缓存使用不同编译器，另行移走旧构建目录后配置。
`server.conf` 可修改仓库位置、站点模块和 MPI 参数；默认 mpirun 的 UCX 参数与原 wm2 作业一致。

## 提交（不会自动取消或删除旧任务）

```bash
# 一次提交七个独立实验，共请求最多 14 GPU / 14 CPU task
bash scripts/submit_all.sh
# 或只提交其中一个
bash scripts/submit.sh mad_am09375
```

提交七个实验与提前提交续跑段是两回事：每个实验同时最多一个活动/排队作业，
后继只在当前作业实际结束后提交，没有 afterany 预排队。
输入按 CASE 固定，运行目录为 `run/<CASE>`，禁止从环境变量继承其他任务的输入或运行目录。
相同目录重复调用 submit.sh 会保留现有作业、恢复监控，不重复提交。

## 自动续跑

登录节点 `nohup watch.sh CASE` 每 60 秒查询 squeue/sacct：

- 活跃作业：继续等待；终止状态 TIMEOUT/NODE_FAIL/PREEMPTED/COMPLETED：检查候选 restart 后提交下一段；
- FAILED/OOM/CANCELLED：停止，保留日志；查询失败：等待，不能把失败查询当作任务结束；
- 没有候选 restart、所有候选验证失败或连续三次没有 checkpoint 进展：停止，避免重复烧机时；
- 成功跑到 30000M 写 `run/<CASE>/DONE`，不再续跑。

不设置 `-t` 或墙钟 checkpoint 定时器。硬截止时从常规 100M restart 恢复，
末次写盘不完整时会尝试较旧快照，因此有回退重算成本。
checkpoint.py 读取 restart 配套的 `.rhdf.xdmf` 文本周期数/时间排序，检查非空和 HDF5 文件签名；
这不是完整性证明。计算节点用 `PANGU -r FILE -m 2` 实际读取 restart 状态，失败后回退到较旧候选。
保留归档 final restart 及其 XDMF，避免最终文件被覆盖后失去恢复点。
XDMF 时间精度有限，不用于判断任务完成；必须由 PANGU 按 restart 内精确时间正常完成后才标记 DONE。
恢复旧 checkpoint 可能覆盖较新时间的同名编号输出；出现损坏后需要取证时，先备份 run 目录。

生产脚本只需要 Python3 标准库，不需要 h5py/numpy/h5dump。PANGU 本身当然仍需 HDF5 动态库。
每个作业的版本检查、输入检查和 restart 预检均在 GPU allocation 内通过 2-rank MPI 运行。

SSH 断开不影响 Slurm 和 nohup。登录节点重启/进程清理会停止监控，重新运行 submit.sh CASE 恢复；
集群须允许登录节点轻量长期监控。下一段何时开跑仍受队列资源影响，不能保证无缝衔接。

## 日志与控制

```bash
case=mad_am09375
cat "run/$case/job.id"
tail -f "run/$case/logs/run-$(cat "run/$case/job.id").log"
tail -f "run/$case/logs/monitor.log"
# 停止后续自动提交（不会杀正在运行的计算）
touch "run/$case/STOP"
# 需要终止当前作业时，再明确 scancel 相应 job ID
# 解除暂停后，删除该 CASE 的 STOP 文件，再执行 submit.sh CASE
```

## 验证与数据量

`python3 tests/test_automation.py` 仅用标准库检查七任务、资源、恢复选择、坏 restart 回退、
不预排队和结束停链。代码回归另用真实 MKS 双 MPI 小网格检查 CT、beta、六模型、逆行与 restart。
本地测试不等于生产 GPU 30000M 验证，也不证明已达到 MAD 饱和。
七组 FP64 输出/restart 的总磁盘需求可能达 TB 量级；脚本不会自动删数据。
