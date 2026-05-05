# GRMHD Intro

本文档介绍广义相对论磁流体力学（general relativistic magnetohydrodynamics, GRMHD）的物理背景、物理图像、基本公式、数值守恒形式和典型预期结果。写作参考了 HARM 论文和 Parthenon 框架论文：

- Gammie, McKinney & Tóth 2003, *HARM: A Numerical Scheme for General Relativistic Magnetohydrodynamics*, arXiv: <https://arxiv.org/abs/astro-ph/0301509>
- Grete et al. 2022, *Parthenon -- a performance portable block-structured adaptive mesh refinement framework*, arXiv: <https://arxiv.org/abs/2202.12309>

HARM 论文的核心贡献是把 GRMHD 写成适合有限体积方法的守恒型激波捕捉格式，并用 constrained transport 保持磁场无散。Parthenon 论文的核心贡献是提供面向现代异构硬件和 AMR 的高性能框架，使具体物理代码可以专注于方程、变量、通量和源项。

## 1. 为什么需要 GRMHD

### 1.1 适用物理系统

GRMHD 用来描述强引力、强磁场、高速流体共同作用的等离子体系统。典型应用包括：

- 黑洞吸积盘。
- 黑洞喷流。
- 中子星并合后的磁化残余盘。
- 相对论性恒星塌缩。
- 低辐射效率吸积流。
- 事件视界附近的磁通积累和喷流发射。

这些系统有三个共同特点：

1. 引力场强，必须使用广义相对论。
2. 等离子体速度可接近光速，必须使用相对论流体力学。
3. 磁场能量密度可与气体内能、动能甚至静质量能相比，必须使用磁流体力学。

牛顿 MHD 不能处理黑洞时空和近光速流。普通 GR hydrodynamics 又不能描述磁张力、磁压、Alfvén 波和 Poynting flux。因此需要 GRMHD。

### 1.2 物理图像

可以把 GRMHD 看成“在弯曲时空中演化带磁场的相对论流体”。流体携带质量、动量、能量和磁场；磁场又反过来提供磁压和磁张力，改变流体运动。

在黑洞吸积盘中，基本图像是：

1. 气体绕黑洞近 Keplerian 旋转。
2. 弱磁场通过 magnetorotational instability（MRI）触发湍流。
3. 湍流产生有效角动量输运，使气体向内吸积。
4. 磁场被差分旋转拉伸和放大。
5. 部分磁能通过湍流级联、磁重联和数值尺度耗散转化为热。
6. 黑洞附近强磁场可以把旋转能转化为 Poynting flux，形成喷流。

GRMHD 的核心物理不是单一力学过程，而是下面几种过程的耦合：

- 相对论惯性和 Lorentz 因子。
- 黑洞时空的红移、frame dragging 和 horizon causal structure。
- 磁压和磁张力。
- Alfvén、slow、fast magnetosonic 波。
- 激波和接触间断。
- 湍流和磁重联。

### 1.3 理想 MHD 近似

大多数 GRMHD 代码采用理想 MHD 条件：

\[
u_\mu F^{\mu\nu} = 0.
\]

这表示在流体共动系中电场为零：

\[
\mathbf{E}' = 0.
\]

在非相对论语言中，它对应：

\[
\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0.
\]

物理含义是磁场“冻结”在流体中。磁力线随流体运动，除非在网格尺度、显式电阻或 kinetic scale 上发生重联。

这个近似要求等离子体电导率足够高，电阻耗散尺度远小于宏观尺度。黑洞吸积流虽然弱碰撞，但通常仍可在大尺度上使用理想 MHD 描述磁场和流体的耦合。

## 2. 基本几何和记号

GRMHD 在给定时空背景上演化。常见黑洞吸积流模拟使用固定 Kerr metric，不演化 Einstein 方程。这称为 test-fluid 或 fixed-background approximation。

度规记为：

\[
ds^2 = g_{\mu\nu} dx^\mu dx^\nu.
\]

逆度规满足：

\[
g^{\mu\alpha}g_{\alpha\nu} = \delta^\mu_{\ \nu}.
\]

四维体积因子由度规行列式给出：

\[
\sqrt{-g} = \sqrt{-\det(g_{\mu\nu})}.
\]

四速度满足归一化条件：

\[
u^\mu u_\mu = -1.
\]

其中采用 \((- + + +)\) 符号约定。三维速度不是基本协变量；GRMHD 方程通常用四速度 \(u^\mu\)、磁场四矢量 \(b^\mu\) 和应力能量张量 \(T^{\mu\nu}\) 写出。

## 3. 守恒方程

GRMHD 的核心方程包括：

1. 质量守恒。
2. 能量动量守恒。
3. Maxwell 方程中的磁感应方程。
4. 磁场无散约束。
5. 状态方程。

### 3.1 质量守恒

协变形式：

\[
\nabla_\mu(\rho u^\mu) = 0.
\]

其中 \(\rho\) 是共动系质量密度，\(u^\mu\) 是四速度。

把协变散度写成坐标守恒形式：

\[
\partial_\mu(\sqrt{-g}\rho u^\mu) = 0.
\]

展开时间和空间部分：

\[
\partial_t(\sqrt{-g}\rho u^t) + \partial_i(\sqrt{-g}\rho u^i) = 0.
\]

这就是有限体积方法中的质量守恒方程。守恒变量通常包含：

\[
D = \rho u^t.
\]

有些代码还把 \(\sqrt{-g}\) 吸收到守恒变量中，有些则放在通量和源项外层。

### 3.2 能量动量守恒

协变形式：

\[
\nabla_\mu T^\mu_{\ \nu} = 0.
\]

展开成坐标形式：

\[
\partial_\mu(\sqrt{-g}T^\mu_{\ \nu}) = \sqrt{-g}T^\kappa_{\ \lambda}\Gamma^\lambda_{\nu\kappa}.
\]

右侧是几何源项，来自 Christoffel 符号：

\[
\Gamma^\lambda_{\nu\kappa} = \frac{1}{2}g^{\lambda\sigma}(\partial_\nu g_{\sigma\kappa} + \partial_\kappa g_{\sigma\nu} - \partial_\sigma g_{\nu\kappa}).
\]

平直时空中 \(\Gamma^\lambda_{\nu\kappa}=0\)，能量动量方程就是严格守恒律。弯曲坐标或弯曲时空中，坐标分量的能量动量会和几何交换，因此出现源项。

### 3.3 理想 GRMHD 应力能量张量

理想 GRMHD 的总应力能量张量为：

\[
T^{\mu\nu} = (\rho + u + p + b^2)u^\mu u^\nu + \left(p + \frac{b^2}{2}\right)g^{\mu\nu} - b^\mu b^\nu.
\]

这里：

- \(\rho\)：静质量密度。
- \(u\)：气体内能密度。
- \(p\)：气体压强。
- \(b^\mu\)：共动系磁场四矢量。
- \(b^2 = b^\mu b_\mu\)：磁场强度平方。

可定义焓密度：

\[
w = \rho + u + p.
\]

则：

\[
T^{\mu\nu} = (w + b^2)u^\mu u^\nu + \left(p + \frac{b^2}{2}\right)g^{\mu\nu} - b^\mu b^\nu.
\]

这个表达式有清楚物理分解：

- \((w+b^2)u^\mu u^\nu\)：流体惯性加磁场惯性。
- \(p g^{\mu\nu}\)：气体压强。
- \((b^2/2)g^{\mu\nu}\)：各向同性磁压。
- \(-b^\mu b^\nu\)：沿磁力线方向的磁张力。

磁张力项使磁场像有张力的弹性线，抵抗弯曲；磁压项使强磁场区向外膨胀。

### 3.4 状态方程

最常用的理想气体状态方程为：

\[
p = (\gamma - 1)u.
\]

其中 \(\gamma\) 是绝热指数。非相对论单原子气体常用 \(\gamma=5/3\)，相对论热气体常用 \(\gamma=4/3\)。许多 GRMHD 模拟为了简化使用常数 \(\gamma\)，但真实吸积流中电子和离子的有效绝热指数可能随温度变化。

## 4. 磁场和 Maxwell 方程

### 4.1 磁场四矢量

在理想 MHD 中，常用共动磁场四矢量 \(b^\mu\)。它与实验室系磁场和四速度有关。一个常见关系是：

\[
b^t = B^i u_i,
\]

\[
b^i = \frac{B^i + b^t u^i}{u^t}.
\]

这里 \(B^i\) 是 Eulerian 或坐标系下演化的三维磁场变量。\(b^\mu\) 满足：

\[
b^\mu u_\mu = 0.
\]

这表示共动磁场在流体本地静止系中是纯空间向量。

### 4.2 磁感应方程

理想 MHD 的磁感应方程可写成守恒形式：

\[
\partial_t(\sqrt{-g}B^i) + \partial_j\left[\sqrt{-g}(b^j u^i - b^i u^j)\right] = 0.
\]

这等价于弯曲时空中的 Faraday 方程加上理想 MHD 条件。

直观上，它表达磁通冻结：磁场随流体拖拽、拉伸和压缩。差分旋转会把径向磁场拉成环向磁场，湍流会折叠磁力线，重联会在数值或微观尺度改变磁拓扑。

### 4.3 无散约束

磁场还必须满足：

\[
\partial_i(\sqrt{-g}B^i) = 0.
\]

这是没有磁单极子的约束。如果数值误差产生非零散度，会导致非物理磁力，破坏激波、喷流和磁压平衡。

HARM 使用 constrained transport 思想维持磁场无散。Parthenon/Athena 系代码通常也把 constrained transport 或等价技术作为 MHD 模块的核心。

## 5. 守恒型数值形式

HARM 的关键思想是把 GRMHD 写成适合有限体积更新的形式：

\[
\partial_t \mathbf{U}(\mathbf{P}) = -\partial_i \mathbf{F}^i(\mathbf{P}) + \mathbf{S}(\mathbf{P}).
\]

其中：

- \(\mathbf{P}\)：primitive variables。
- \(\mathbf{U}\)：conserved variables。
- \(\mathbf{F}^i\)：方向 \(i\) 上的通量。
- \(\mathbf{S}\)：几何源项。

典型 primitive variables 是：

\[
\mathbf{P} = (\rho, u, \tilde{u}^1, \tilde{u}^2, \tilde{u}^3, B^1, B^2, B^3).
\]

典型 conserved variables 是：

\[
\mathbf{U} = (D, T^t_{\ t}, T^t_{\ 1}, T^t_{\ 2}, T^t_{\ 3}, B^1, B^2, B^3).
\]

不同代码会改变符号、能量变量定义或是否扣除静质量能，但核心结构相同：守恒变量来自质量流和应力能量张量的时间分量。

### 5.1 有限体积更新

对一个网格单元体积积分：

\[
\frac{d}{dt}\int_V \mathbf{U}\,dV = -\sum_{\rm faces}\int_{\partial V}\mathbf{F}^i n_i\,dA + \int_V \mathbf{S}\,dV.
\]

离散化后：

\[
\mathbf{U}^{n+1}_{ijk} = \mathbf{U}^n_{ijk} - \Delta t\sum_d\frac{\mathbf{F}^{d}_{i+1/2}-\mathbf{F}^{d}_{i-1/2}}{\Delta x^d} + \Delta t\,\mathbf{S}_{ijk}.
\]

有限体积方法的优势是激波捕捉能力强。即使解中出现不连续，守恒律仍然通过界面通量正确传递质量、动量和能量。

### 5.2 Riemann 通量

HARM 使用 Harten-Lax-van Leer 类型的近似 Riemann solver。其思想是：界面两侧有左右状态 \(\mathbf{P}_L,\mathbf{P}_R\)，通过波速估计构造一个稳定的数值通量。

Lax-Friedrichs 型通量可写为：

\[
\mathbf{F}_{\rm LF} = \frac{1}{2}\left[\mathbf{F}(\mathbf{P}_L) + \mathbf{F}(\mathbf{P}_R)\right] - \frac{a}{2}\left[\mathbf{U}(\mathbf{P}_R) - \mathbf{U}(\mathbf{P}_L)\right].
\]

其中 \(a\) 是最大信号速度估计。第一项是中心通量，第二项是数值耗散。GRMHD 中信号速度包括 fast magnetosonic waves，精确求解较复杂，因此许多代码使用保守估计。

### 5.3 primitive recovery

GRMHD 的困难之一是从守恒变量恢复 primitive variables：

\[
\mathbf{U} \rightarrow \mathbf{P}.
\]

在牛顿流体中，这通常是显式代数关系。但在 GRMHD 中，守恒变量包含 Lorentz 因子、磁场、焓和度规的非线性耦合。恢复过程通常需要数值求根。

典型未知量可以选为：

\[
W = \rho h \Gamma^2,
\]

或 \(W\) 与 \(v^2\) 的组合。恢复算法需要保证：

\[
\rho > 0,\qquad u > 0,\qquad v^2 < 1.
\]

如果恢复失败，代码通常需要 fixer 或 floor：

- 密度 floor。
- 内能 floor。
- Lorentz factor ceiling。
- 磁化强度 ceiling。
- 熵变量辅助恢复。

这不是纯数值细节，而是 GRMHD 模拟稳定性的核心。黑洞 funnel 区常有极低密度和高磁化，守恒变量很容易落到物理解空间外。

## 6. 源项的物理意义

能量动量方程的源项：

\[
S_\nu = \sqrt{-g}T^\kappa_{\ \lambda}\Gamma^\lambda_{\nu\kappa}.
\]

它来自时空几何。物理上，它描述坐标基底随空间变化导致的动量和能量交换。

在平直 Cartesian 坐标中：

\[
\Gamma^\lambda_{\nu\kappa}=0.
\]

所以源项消失。

在球坐标的平直时空中，虽然没有真实引力，Christoffel 符号也不为零；源项包含离心项和曲线坐标几何项。

在 Kerr 时空中，源项还包含真正的强引力效应、frame dragging 和黑洞势阱。

## 7. 黑洞吸积中的关键物理结果

### 7.1 MRI 湍流和角动量输运

弱磁化差分旋转盘满足 MRI 条件时会不稳定。简化 Newtonian 图像中，只要角速度随半径下降：

\[
\frac{d\Omega}{dR} < 0,
\]

弱磁场连接相邻流体元。内侧流体元转得更快，外侧转得更慢，磁张力把角动量从内侧传到外侧。内侧失去角动量下落，外侧获得角动量外移。

预期结果：

- 弱种子磁场被放大。
- 盘内形成 MHD 湍流。
- Maxwell stress 主导角动量输运。
- 吸积率进入准稳态波动。

角动量输运常由应力表示：

\[
T^r_{\ \phi} \sim \rho u^r u_\phi - b^r b_\phi.
\]

其中磁项 \(-b^r b_\phi\) 通常是关键贡献。

### 7.2 磁通积累和 MAD/SANE

黑洞吸积流按磁通量可分为两类常见状态：

- SANE：standard and normal evolution，黑洞附近磁通较弱或中等。
- MAD：magnetically arrested disk，黑洞附近积累大量磁通，磁压强足以阻碍吸积。

无量纲磁通量常写成：

\[
\phi_{\rm BH} \sim \frac{\Phi_{\rm BH}}{\sqrt{\dot{M}r_g^2 c}}.
\]

SANE 中磁通较低，盘体湍流主导。MAD 中强磁通压住内盘，吸积呈现间歇性磁通喷发和强喷流。

预期结果：

- MAD 的磁场更有序，喷流更强。
- MAD 中吸积率波动更剧烈。
- SANE 更接近弱磁湍流盘。

### 7.3 Blandford-Znajek 喷流

旋转黑洞可通过磁场释放旋转能，形成 Poynting-flux dominated jet。这一机制通常称为 Blandford-Znajek process。

黑洞视界角速度：

\[
\Omega_H = \frac{a}{2r_H}.
\]

喷流功率的标度关系可写成：

\[
P_{\rm BZ} \propto \Phi_{\rm BH}^2 \Omega_H^2.
\]

其中 \(\Phi_{\rm BH}\) 是穿过黑洞视界的磁通量，\(a\) 是无量纲自旋。

GRMHD 模拟预期看到：

- 高自旋黑洞喷流更强。
- MAD 状态喷流功率显著增强。
- funnel 区由低密度、高磁化、Poynting flux 主导。

### 7.4 激波、间断和耗散

GRMHD 解中常出现：

- 磁声波。
- Alfvén 波。
- 激波。
- 接触间断。
- current sheet。

有限体积激波捕捉方法通过数值通量处理这些间断。不可逆耗散通常表现为内能或熵增加。两温模型中，后续还要决定这些耗散有多少进入电子。

## 8. 坐标和黑洞时空

### 8.1 Kerr 时空

旋转黑洞通常用 Kerr metric 描述。黑洞由质量 \(M\) 和自旋 \(a\) 决定。事件视界半径：

\[
r_H = M + \sqrt{M^2 - a^2}.
\]

在几何单位 \(G=c=M=1\) 下：

\[
r_H = 1 + \sqrt{1-a^2}.
\]

GRMHD 代码常使用 horizon-penetrating 坐标，例如 Kerr-Schild 坐标，而不是 Boyer-Lindquist 坐标。原因是 Boyer-Lindquist 坐标在视界处有坐标奇性，不利于跨视界演化。

### 8.2 坐标与物理

GRMHD 方程是协变的，但数值代码必须选坐标。坐标选择会影响：

- 网格集中在哪里。
- 源项大小。
- CFL 时间步。
- 是否能穿过视界。
- 极轴附近的数值稳定性。

HARM 的一个重要设计是：只需要坐标基下的协变度规 \(g_{\mu\nu}\) 就能指定几何。这使代码可适配不同坐标和时空背景。

## 9. Parthenon 框架中的计算图像

Parthenon 不是 GRMHD 方程本身，而是用于实现这类方程的 block-structured AMR 框架。它来自 Athena++ 代码体系，并使用 Kokkos 实现性能可移植。

### 9.1 为什么物理代码需要 Parthenon 这类框架

现代 GRMHD 模拟通常需要：

- 三维网格。
- 高分辨率。
- AMR。
- MPI 多节点并行。
- GPU 加速。
- 大量变量和派生场。
- 多物理模块，例如辐射、电子热力学、核物理或自引力。

如果每个物理代码都从零实现这些基础设施，成本很高，也难以维护。Parthenon 提供：

- mesh block 管理。
- AMR 数据结构。
- package 机制。
- 多维变量抽象。
- Kokkos 并行 kernel。
- device memory 管理。
- variable packing。
- MPI 通信。

### 9.2 GRMHD 在框架中的模块化

一个 GRMHD 代码通常可拆成：

1. 初始化模块：设置 torus、磁场、扰动和边界。
2. metric 模块：计算 \(g_{\mu\nu}\)、\(g^{\mu\nu}\)、\(\sqrt{-g}\)、Christoffel 符号。
3. primitive-to-conserved 模块。
4. reconstruction 模块。
5. Riemann solver 模块。
6. constrained transport 模块。
7. source term 模块。
8. primitive recovery 模块。
9. fixer 模块。
10. diagnostics 模块。

Parthenon 的 package 机制适合把这些功能拆成独立组件，并在 task list 中组织执行顺序。

### 9.3 性能预期

Parthenon 论文强调性能可移植和 AMR 基础设施。对 GRMHD 来说，这意味着：

- 同一物理代码可运行在 CPU 和 GPU。
- 大型三维模拟可以扩展到多节点。
- 变量打包可减少 kernel launch overhead。
- mesh block 可作为并行和 AMR 的基本单位。
- 物理开发者可以更专注于方程和算法。

## 10. 典型 GRMHD 模拟流程

一个黑洞吸积 GRMHD 模拟通常如下：

1. 选择时空：Kerr 或 Minkowski。
2. 选择坐标：Kerr-Schild、modified Kerr-Schild、Boyer-Lindquist 或 Cartesian。
3. 初始化平衡 torus，例如 Fishbone-Moncrief torus。
4. 加入弱 poloidal magnetic field。
5. 加入小扰动触发 MRI。
6. 演化 GRMHD 方程。
7. 监测质量吸积率、磁通量、喷流功率和能量守恒。
8. 等待内盘进入准稳态。
9. 统计时间平均结构。
10. 可选地进行辐射转移或两温电子演化。

常见诊断量包括：

\[
\dot{M} = -\int \rho u^r \sqrt{-g}\,d\theta d\phi.
\]

\[
\dot{E} = \int T^r_{\ t}\sqrt{-g}\,d\theta d\phi.
\]

\[
\dot{J} = -\int T^r_{\ \phi}\sqrt{-g}\,d\theta d\phi.
\]

\[
\Phi_{\rm BH} = \frac{1}{2}\int |B^r|\sqrt{-g}\,d\theta d\phi.
\]

这些量分别描述质量、能量、角动量和磁通穿过某个半径球面的通量。

## 11. 预期物理结果

### 11.1 稳定测试

在平直时空或简单波动测试中，应看到：

- 光滑解二阶收敛，若采用二阶重构和二阶时间推进。
- 激波位置正确。
- 守恒误差随分辨率降低。
- 磁场散度保持在截断误差水平。

这对应 HARM 论文中强调的 conservative shock-capturing 和 constrained transport 性质。

### 11.2 磁化 torus 演化

在黑洞附近磁化 torus 中，预期阶段为：

1. 初始 torus 近似平衡。
2. MRI 增长。
3. 湍流发展。
4. 内盘开始吸积。
5. 黑洞附近形成 corona 和 funnel。
6. 若磁通足够强，形成喷流。

典型空间结构：

- 赤道附近：高密度、湍流盘体。
- 盘上方：低密度 corona。
- 极区：低密度、高磁化 funnel。
- funnel wall：盘风和喷流交界。

### 11.3 守恒和耗散

理想 GRMHD 方程本身守恒总能量动量，但数值模拟中的激波和湍流会通过数值耗散产生熵。合理结果应满足：

- 总守恒量误差受控。
- 内能在激波和湍流耗散区增加。
- 磁能可转化为热能、动能和 Poynting flux。
- 无散约束不积累成大尺度误差。

### 11.4 失败模式

常见失败模式包括：

- primitive recovery 失败。
- 负密度或负内能。
- 过高 Lorentz 因子。
- 过高磁化导致守恒变量病态。
- 极轴坐标奇性附近不稳定。
- 磁场散度误差积累。
- atmosphere/floor 注入过多质量，污染喷流。

因此 GRMHD 模拟不仅是方程求解问题，也是约束维护和物理解空间投影问题。

## 12. 从 GRMHD 到两温和辐射

标准 GRMHD 只演化总内能。若要预测辐射，通常还需要电子温度：

\[
T_e \neq T_i.
\]

两温模型加入电子熵或电子内能方程，并使用电子加热模型决定：

\[
Q_e = f_e Q.
\]

辐射 GRMHD 还会加入辐射能量动量张量 \(R^{\mu\nu}\)，总守恒变为：

\[
\nabla_\mu(T^{\mu}_{\ \nu} + R^{\mu}_{\ \nu}) = 0.
\]

物质和辐射之间通过四力密度 \(G_\nu\) 交换能量动量：

\[
\nabla_\mu T^\mu_{\ \nu} = G_\nu,\qquad \nabla_\mu R^\mu_{\ \nu} = -G_\nu.
\]

这说明 GRMHD 是更复杂模型的基础层。两温电子、辐射、非理想 MHD 和 kinetic closure 都建立在同一套质量、能量动量和磁场演化框架上。

## 13. 与代码实现的对应关系

在本项目中，GRMHD 相关模块大致对应：

- `metric/`：度规、逆度规、连接系数。
- `physics/energy_momentum_tensor.h`：应力能量张量。
- `physics/contravariant_flux.h`：通量。
- `riemann_solver/`：守恒变量、Riemann 通量和源项。
- `constrained_transport/`：磁场无散维护。
- `recovery/`：primitive recovery。
- `fixer/`：floor、Lorentz 限制和状态修复。
- `task_list/ideal_grmhd.cc`：GRMHD 时间推进任务组织。

这和 HARM 的基本算法图像一致：primitive variables 生成 flux 和 conserved variables，有限体积更新 conserved variables，再 recovery 回 primitive variables。Parthenon 则提供 mesh block、变量注册、并行 kernel 和 task list 基础设施。

## 14. 参考文献

- Gammie, C. F., McKinney, J. C., & Tóth, G. 2003, *HARM: A Numerical Scheme for General Relativistic Magnetohydrodynamics*, Astrophysical Journal, 589, 444-457, arXiv:astro-ph/0301509, <https://arxiv.org/abs/astro-ph/0301509>.
- Grete, P., Dolence, J. C., Miller, J. M., et al. 2022, *Parthenon -- a performance portable block-structured adaptive mesh refinement framework*, arXiv:2202.12309, <https://arxiv.org/abs/2202.12309>.
