# 本地验证记录

- 实际构建：MKS/static、wave、Release、MPI、HDF5，CPU 后端（本地），路径 build-electron-mks。
- 七组输入均在 32×24×8、两个径向 MeshBlock、双 MPI rank 上完成初始化和两步短演化。
- 六电子模型（加 Ktot 共 7 个电子组件）和流体场均有限；归一化比值均为 100（舍入误差内）。
- 七组最大 |divB| 的最大值为 6.86e-16。
- MAD 和 SANE 种子磁场确实不同；显式 SANE 与省略 selector 的旧行为逐位一致。
- MAD 省略权重参数与显式标准权重逐位一致；非法 selector 和误用 SANE 权重被拒绝。
- 负自旋模型密集 torus 保持正向旋转，代表相对黑洞逆行。
- 单/双 MPI 初始场一致；MAD 分段重启与连续短演化在 rtol=1e-12、atol=1e-13 内一致。
- 实际 PANGU `-r checkpoint -m 2` 读取 restart 成功且未生成输出；脚本能读取真实 XDMF 时间/周期数。
- 13 项标准库脚本测试通过（包括旧 final 不遮盖新编号快照、坏 restart 回退、XDMF 舍入不能误判完成）。

未执行服务器 GPU80G 生产运行、384×192×192 完整初始化或 30000M 长演化。
该验证确认初始化和续跑机制，不证明磁通已经达到 MAD 饱和。
