# Electron Heating Models

本文档说明 `src/physics/heating_model.h` 中实现的电子加热模型。这里的电子加热模型不是直接求解 kinetic plasma physics，而是在 GRMHD 网格尺度耗散已经出现之后，用一个闭合关系给出耗散热量进入电子的比例 \(f_e\)。

这些模型的共同目标是回答一个问题：

**数值耗散、湍流或磁重联释放出的热量，应该有多少进入电子？**

在弱碰撞黑洞吸积流、喷流和热稀薄等离子体中，电子和离子之间的库仑耦合很慢。离子和电子不必共享同一个温度，因此单温 GRMHD 中的总内能 \(u\) 并不能唯一决定电子温度。两温模型需要额外闭合，把不可逆耗散拆分为电子加热和离子加热。

## 1. 总物理背景

### 1.1 为什么需要电子加热闭合

GRMHD 方程通常演化总质量、总动量、总能量和磁场。理想 MHD 本身没有显式粘性、电阻或碰撞项；真实物理中的湍流级联、Landau damping、cyclotron damping、磁重联和压力各向异性耗散，在网格尺度上只表现为数值耗散或恢复过程中的熵增。

如果只做单温流体，可以把这些耗散全部加入一个总内能：

\[
u = u_i + u_e .
\]

但辐射主要由电子产生，尤其在黑洞吸积流中，synchrotron、Compton 和 bremsstrahlung 都强烈依赖电子温度 \(T_e\)。因此即使总内能正确，如果电子获得的热量比例错了，辐射图像和光变也会错。

两温模型把耗散拆成：

\[
Q = Q_i + Q_e, \qquad f_e = \frac{Q_e}{Q_i + Q_e}.
\]

代码中的各模型就是不同的 \(f_e\) 处方。

### 1.2 弱碰撞等离子体图像

在稀薄吸积流中，粒子平均自由程可能远大于微观回旋半径，也可能接近甚至大于宏观梯度尺度。此时“温度”仍可作为粗粒化变量使用，但加热不再由简单的 collisional equipartition 决定。

关键无量纲量包括：

\[
\beta = \frac{p_{\rm gas}}{p_{\rm mag}} \simeq \frac{2p}{b^2},
\]

\[
\sigma = \frac{b^2}{\rho} \quad \text{或} \quad \sigma = \frac{b^2}{\rho + u + p_g},
\]

\[
\theta = \frac{T_i}{T_e}.
\]

其中：

- 低 \(\beta\)：磁压主导，磁场强，电子通常更容易被小尺度电磁涨落加热。
- 高 \(\beta\)：热压主导，离子通道通常更强。
- 高 \(\sigma\)：单位质量可释放磁能高，重联可变得相对论性。
- 大 \(\theta\)：离子比电子热，后续加热比例可能被温度比反馈修正。

### 1.3 三类主要物理闭合

本文档中的模型大致分为三类：

1. 湍流加热模型：`howes`、`kawazura`
2. 磁重联加热模型：`werner`、`rowan`
3. 简化参数或温度比模型：`constant`、`sharma`

湍流模型的图像是：大尺度 MHD 湍流把能量级联到 kinetic scale，然后通过不同 damping channel 加热离子或电子。

重联模型的图像是：磁场拓扑改变，磁能在 current sheet 中释放，粒子通过 reconnection electric field、contracting islands 和 Fermi-like process 被加热或加速。

温度比模型的图像是：不显式判断耗散机制，只用当前 \(T_e/T_i\) 或固定参数决定电子份额。

## 2. 代码中的共同框架

代码演化两个熵变量：

- 总气体熵：\(K_{\rm tot}\)
- 电子熵：\(K_e\)

恢复得到的总熵记为 \(K_{\rm tot}^{\rm rec}\)，随流输运得到的总熵记为 \(K_{\rm tot}^{\rm adv}\)。二者之差表示这一时间步中由不可逆过程产生的有效熵增：

\[
\Delta K_{\rm tot} = K_{\rm tot}^{\rm rec} - K_{\rm tot}^{\rm adv}.
\]

为了把总熵增映射到电子熵增，代码使用：

\[
\Delta K_e = \frac{\gamma_e - 1}{\gamma - 1} \rho^{\gamma - \gamma_e} \Delta K_{\rm tot}.
\]

然后由模型给出的电子加热比例 \(f_e\) 更新电子熵：

\[
K_e = K_e^{\rm adv} + f_e \Delta K_e .
\]

这里有两个层次：

1. `apply` 决定“这一步有多少可分配给电子的熵增”。
2. `computeHeatingFraction` 决定“电子拿走多少比例”。

代码中使用的温度尺度为：

\[
T_p = \frac{(\gamma_p - 1)u}{\rho}, \qquad T_e = K_e^{\rm adv}\rho^{\gamma_e - 1}.
\]

其中 \(u\) 对应 `energy`，\(\rho\) 对应 `rho`，\(b^2\) 对应 `bsq`。这里的 \(T_p\) 和 \(T_e\) 是代码单位下的温度尺度，不必理解为带物理常数 \(k_B\) 的绝对温度。

如果启用 `limit_kel`，代码还会用 `ratio_min` 和 `ratio_max` 限制电子熵，使离子/电子温度比不越界。这是工程上的稳定性和物理可接受性约束。

### 2.1 从总熵增到电子熵增

这里的推导从理想气体熵变量出发。对总流体，代码使用类似

\[
K_{\rm tot}=\frac{p}{\rho^\gamma}=\frac{(\gamma - 1)u}{\rho^\gamma}.
\]

对电子分量，

\[
K_e=\frac{p_e}{\rho^{\gamma_e}}.
\]

如果某一步恢复过程给出更大的总熵，则

\[
\Delta K_{\rm tot}=K_{\rm tot}^{\rm rec}-K_{\rm tot}^{\rm adv}
\]

表示这一时间步中由不可逆过程产生的总熵增。总压强增量对应：

\[
\Delta p=\rho^\gamma \Delta K_{\rm tot}.
\]

总内能增量为：

\[
\Delta u=\frac{\Delta p}{\gamma - 1}=\frac{\rho^\gamma}{\gamma - 1}\Delta K_{\rm tot}.
\]

如果电子获得其中比例 \(f_e\)，则电子内能增量为：

\[
\Delta u_e=f_e \Delta u.
\]

电子压强增量为：

\[
\Delta p_e=(\gamma_e - 1)\Delta u_e=f_e\frac{\gamma_e - 1}{\gamma - 1}\rho^\gamma\Delta K_{\rm tot}.
\]

电子熵增量满足：

\[
\Delta K_e=\frac{\Delta p_e}{\rho^{\gamma_e}}=f_e\frac{\gamma_e - 1}{\gamma - 1}\rho^{\gamma - \gamma_e}\Delta K_{\rm tot}.
\]

所以代码先计算不含 \(f_e\) 的部分：

\[
\Delta K_{e,0} = \frac{\gamma_e - 1}{\gamma - 1} \rho^{\gamma - \gamma_e} \Delta K_{\rm tot},
\]

再更新：

\[
K_e = K_e^{\rm adv} + f_e \Delta K_{e,0}.
\]

这说明 `computeHeatingFraction` 中的每个模型只负责 \(f_e\)，而不负责总耗散大小。

### 2.2 温度比限制的含义

电子熵限制来自温度比约束。若近似把总压强分为质子压强和电子压强：

\[
p = p_p + p_e,
\]

并定义温度比

\[
R = \frac{T_p}{T_e},
\]

那么在单位常数吸收到代码单位后，可以把压强比理解为温度比的代理：

\[
\frac{p_p}{p_e} \sim R.
\]

限制 `ratio_min` 和 `ratio_max` 等价于要求：

\[
R_{\min} \le \frac{T_p}{T_e} \le R_{\max}.
\]

在数值上，这避免电子熵过小导致电子温度塌缩，也避免电子熵过大导致电子温度吞掉几乎全部热压。它不是微观加热模型的一部分，而是两温流体演化中的物理/数值边界条件。

### 2.3 符号与代码变量

| 符号 | 代码变量 | 含义 |
| --- | --- | --- |
| \(\rho\) | `rho` | 质量密度 |
| \(u\) | `energy` | 内能密度 |
| \(b^2\) | `bsq` | 磁场四矢量平方 |
| \(\gamma\) | `gamma` | 总气体绝热指数 |
| \(\gamma_p\) | `gamma_p` | 质子绝热指数 |
| \(\gamma_e\) | `gamma_e` | 电子绝热指数 |
| \(K_{\rm tot}^{\rm adv}\) | `advected_total_entropy` | 随流输运总熵 |
| \(K_{\rm tot}^{\rm rec}\) | `recovered_total_entropy` | 恢复得到总熵 |
| \(K_e^{\rm adv}\) | `advected_electron_entropy` | 随流输运电子熵 |
| \(f_e\) | `fel` | 电子加热比例 |
| \(T_p\) | `tpr` | 质子温度尺度 |
| \(T_e\) | `tel` | 电子温度尺度 |

## 3. `constant`

### 3.1 物理动机

`constant` 模型用于建立最小闭合：

\[
f_e = f_{e,0}.
\]

它的动机不是还原某一种微观耗散机制，而是提供一个可控基准。很多两温计算首先需要回答：如果电子总是获得固定比例的耗散热，系统会给出什么电子温度和辐射结果？

### 3.2 物理假设

该模型隐含假设：

\[
\frac{Q_e}{Q_i + Q_e}
\]

在空间、时间和等离子体状态上都近似常数。

这等价于认为局部 \(\beta\)、\(\sigma\)、\(T_i/T_e\)、湍流形态和重联几何都不会显著改变电子加热比例。

### 3.3 物理图像

可以把它想成“黑箱耗散器”：每个网格单元发生的不可逆耗散都按同一比例分给电子。这个黑箱不看局部状态，只执行固定分账。

### 3.4 推导过程

从定义出发：

\[
Q = Q_i + Q_e, \qquad Q_e = f_{e,0}Q.
\]

因此：

\[
f_e = \frac{Q_e}{Q} = f_{e,0}.
\]

### 3.5 适用与局限

适合：

- 调试两温模块。
- 做参数扫描。
- 建立与复杂模型的对照。

不适合：

- 研究电子加热如何随 \(\beta\) 或 \(\sigma\) 改变。
- 区分盘体湍流、喷流、current sheet 和重联区。

## 4. `howes`

参考：

- Howes, G. G. 2010, *A prescription for the turbulent heating of astrophysical plasmas*, MNRAS Letters.
- arXiv: <https://arxiv.org/abs/1009.4212>
- DOI: <https://doi.org/10.1111/j.1745-3933.2010.00958.x>

### 4.1 物理背景

Howes 模型针对弱碰撞磁化等离子体中的 Alfvénic turbulence。arXiv 摘要明确说明，它计算的是 Alfvénic turbulence dissipation 导致的 ion/electron heating ratio，并基于 weakly collisional plasma 的 turbulence cascade model。

基本图像是：

\[
\text{大尺度 Alfvénic fluctuation} \rightarrow \text{各向异性湍流级联} \rightarrow \text{kinetic scale damping} \rightarrow Q_i, Q_e.
\]

在磁化弱碰撞等离子体中，湍流不是简单地在 Kolmogorov 尺度被普通粘性耗散，而是级联到离子 Larmor 半径、电子 Larmor 半径等 kinetic scale。不同尺度和不同 damping channel 对离子和电子的耦合不同，因此 \(Q_i/Q_e\) 取决于 \(\beta\) 和温度比。

### 4.2 物理动机

黑洞吸积流中，MRI 湍流可能是主要耗散源。如果耗散来自 Alfvénic turbulence，那么电子加热比例应该反映湍流级联和 kinetic damping，而不是磁重联 PIC 的能量分配。因此 `howes` 是“湍流耗散解释”的代表模型。

### 4.3 物理假设

主要假设包括：

- 耗散来源主要是 Alfvénic turbulence。
- 等离子体弱碰撞，离子和电子可以有不同温度。
- 局部加热比例可由局部 \(\beta\) 和 \(T_i/T_e\) 近似决定。
- 湍流级联模型可以被压缩成一个 \(Q_i/Q_e\) 拟合公式。

代码使用：

\[
\theta = \frac{T_p}{T_e}, \qquad \beta = \frac{2\rho T_p}{b^2}.
\]

这里的 \(\beta\) 实际上是以质子压强尺度构造的离子 beta。

### 4.4 物理图像

低 \(\beta\) 时，磁场相对强，小尺度电磁涨落更容易把能量传给电子；高 \(\beta\) 时，离子 Landau damping 等通道增强，离子获得更多热量。

温度比 \(\theta\) 的作用是调节离子和电子响应频率。如果 \(T_p/T_e\) 改变，离子声速、热速度和 kinetic damping 条件都会变，因此 \(Q_i/Q_e\) 也会变。

### 4.5 公式和推导脉络

Howes 处方先构造离子/电子加热比：

\[
\frac{Q_i}{Q_e} = 0.92 \frac{c_2^2 + \beta^m}{c_3^2 + \beta^m} \exp\left(-\frac{1}{\beta}\right) \sqrt{\frac{m_p}{m_e}\theta}.
\]

其中

\[
m = 2 - 0.2\log_{10}\theta.
\]

分段经验系数为：

\[
c_2 = \begin{cases} 1.6/\theta, & \theta \le 1, \\ 1.2/\theta, & \theta > 1, \end{cases}
\]

\[
c_3 = \begin{cases} 18 + 5\log_{10}\theta, & \theta \le 1, \\ 18, & \theta > 1. \end{cases}
\]

最后把 heating ratio 转换成电子分数：

\[
f_e = \frac{Q_e}{Q_i + Q_e} = \frac{1}{1 + Q_i/Q_e}.
\]

这不是从 MHD 方程直接代数推导出来的闭式公式，而是 turbulence cascade model 与 kinetic damping 结果压缩后的 prescription。代码实现的是这个 prescription 的局部代数版本。

### 4.6 适用与局限

适合：

- MRI 湍流主导的吸积盘区域。
- 需要 \(\beta\) 和 \(T_i/T_e\) 双重反馈的两温计算。

局限：

- 不描述 current sheet 中的重联加热。
- 不包含湍流驱动几何、压缩性湍流比例等更复杂信息。
- 对极端参数区可能需要额外 floor 或 limiter；当前代码选择相信 fixer。

## 5. `kawazura`

参考：

- Kawazura, Y., Barnes, M., & Schekochihin, A. A. 2019, *Thermal disequilibration of ions and electrons by collisionless plasma turbulence*, PNAS.
- arXiv: <https://arxiv.org/abs/1807.07702>
- DOI: <https://doi.org/10.1073/pnas.1812491116>

### 5.1 物理背景

Kawazura 等人的工作同样讨论 collisionless plasma turbulence 中的离子/电子热分配。arXiv 摘要指出，在 Alfvénic turbulence 注入能量的情形下，collisionless turbulent heating 通常会让离子和电子温度偏离热平衡；\(Q_i/Q_e\) 随离子 beta 增大而增大，低 \(\beta_i\) 时电子更热，高 \(\beta_i\) 时离子更热。

### 5.2 物理动机

这个模型的核心动机是给出一个更简洁、更直接的 turbulence heating fit。相比 Howes，Kawazura 的表达式更短，并且强调主要控制变量是 \(\beta_i\)，对 \(T_i/T_e\) 相对不敏感。

### 5.3 物理假设

主要假设包括：

- 能量注入是 Alfvénic turbulence。
- 等离子体是弱碰撞、磁化的。
- 加热分配可用局部 \(\beta_i\) 和较弱的温度比修正表示。
- 公式来自 hybrid fluid-gyrokinetic simulation 的拟合，不是解析第一性原理公式。

代码中：

\[
\theta = \frac{T_p}{T_e}, \qquad \beta = \frac{2\rho T_p}{b^2}.
\]

### 5.4 物理图像

低 \(\beta\) 时，磁能密度高，小尺度 Alfvénic/Kinetic-Alfvénic fluctuation 更容易把能量传给电子，因此 \(Q_i/Q_e\) 小，\(f_e\) 大。

高 \(\beta\) 时，离子热运动和离子尺度 phase mixing 更强，离子加热占优，因此 \(Q_i/Q_e\) 大，\(f_e\) 小。

这给出一个很清楚的物理图像：

\[
\beta \ll 1 \Rightarrow Q_e > Q_i, \qquad \beta \gg 1 \Rightarrow Q_i > Q_e.
\]

### 5.5 公式和推导脉络

代码实现：

\[
\frac{Q_i}{Q_e} = \frac{35} { 1 + \left(\frac{\beta}{15}\right)^{-1.4} \exp\left(-\frac{0.1}{\theta}\right) },
\]

\[
f_e = \frac{1}{1 + Q_i/Q_e}.
\]

这个表达式可以理解为：

- 常数 \(35\) 给出高 \(\beta\) 极限附近的离子优势加热尺度。
- \((\beta/15)^{-1.4}\) 控制 beta 从低到高的过渡。
- \(\exp(-0.1/\theta)\) 给出温度比修正。

在低 \(\beta\) 时，分母中的 beta 项变大，\(Q_i/Q_e\) 变小，因此电子加热增强。在高 \(\beta\) 时，beta 项变小，\(Q_i/Q_e\) 接近较大值，因此离子加热增强。

### 5.6 适用与局限

适合：

- 需要简洁 turbulence heating closure。
- 重点关心 \(\beta\) 对电子温度的控制。

局限：

- 原始结果来自特定模拟设置，映射到任意 GRMHD 耗散都属于 sub-grid 假设。
- 不描述重联几何。
- 当前实现没有包含更一般的湍流注入类型或压缩模式比例。

## 6. `werner`

参考：

- Werner, G. R., Uzdensky, D. A., Begelman, M. C., Cerutti, B., & Nalewajko, K. 2018, *Non-thermal particle acceleration in collisionless relativistic electron-proton reconnection*, MNRAS.
- arXiv: <https://arxiv.org/abs/1612.04493>
- DOI: <https://doi.org/10.1093/mnras/stx2530>

### 6.1 物理背景

Werner 等人的工作研究 relativistic collisionless electron-proton reconnection。arXiv 摘要指出，relativistic collisionless magnetic reconnection 可以加速粒子并驱动高能辐射；电子-离子重联对黑洞吸积流和相对论喷流特别相关。

### 6.2 物理动机

如果局部 GRMHD 耗散主要发生在 current sheet 或磁重联层，那么 turbulence heating prescription 可能不是最佳图像。重联中释放的是磁能，控制参数自然是磁化强度：

\[
\sigma \sim \frac{\text{magnetic energy density}}{\text{matter rest-mass energy density}}.
\]

`werner` 模型用一个只依赖 \(\sigma\) 的平滑函数描述电子加热份额。

### 6.3 物理假设

主要假设包括：

- 耗散来源主要是 collisionless magnetic reconnection。
- 电子和质子在重联层中通过 kinetic process 分享磁能。
- 电子加热比例主要由磁化强度控制。
- \(\beta\) 和温度比的影响被忽略或吸收到拟合中。

代码采用：

\[
\sigma = \frac{b^2}{\rho}.
\]

### 6.4 物理图像

在弱磁化条件下，质子由于质量大、惯性大，通常拿走更多能量，电子份额约为 \(1/4\)。

在强磁化条件下，磁能足够大，电子也能被有效加热和加速。电子和离子的能量分配逐渐接近均分，电子份额趋向 \(1/2\)。

因此模型构造为：

\[
\sigma \to 0 \Rightarrow f_e \to 0.25, \qquad \sigma \to \infty \Rightarrow f_e \to 0.5.
\]

### 6.5 公式和推导脉络

代码实现：

\[
f_e = \frac{1}{4} \left[ 1 + \sqrt{ \frac{\sigma/5}{2+\sigma/5} } \right].
\]

注意这里的括号按代码等价于：

\[
f_e = \frac{1}{4} \left( 1 + \sqrt{ \frac{\sigma/5}{2+\sigma/5} } \right).
\]

极限行为：

低磁化：

\[
\sigma \ll 1 \Rightarrow \frac{\sigma/5}{2+\sigma/5} \approx \frac{\sigma}{10}, \qquad f_e \approx \frac{1}{4} \left(1 + \sqrt{\frac{\sigma}{10}}\right).
\]

高磁化：

\[
\sigma \gg 1 \Rightarrow \frac{\sigma/5}{2+\sigma/5} \to 1, \qquad f_e \to \frac{1}{2}.
\]

这不是从守恒律唯一推出的，而是把 PIC 结果中“磁化越强，电子份额越高”的趋势压缩成局部代数闭合。

### 6.6 适用与局限

适合：

- 磁化喷流。
- 磁重联层。
- MAD 或 corona 中强磁化耗散。

局限：

- 不响应 \(\beta\)。
- 不响应 \(T_i/T_e\)。
- 对热压环境不敏感，因此比 `rowan` 更粗。

## 7. `rowan`

参考：

- Rowan, M. E., Sironi, L., & Narayan, R. 2017, *Electron and proton heating in trans-relativistic magnetic reconnection*, ApJ.
- arXiv: <https://arxiv.org/abs/1708.04627>
- DOI: <https://doi.org/10.3847/1538-4357/aa9380>

### 7.1 物理背景

Rowan、Sironi 和 Narayan 研究 trans-relativistic magnetic reconnection 中电子和质子的加热分配。与 Werner 模型相比，它更明确关注热的、电子-质子等离子体中电子与质子获得不可逆热量的比例，并把结果拟合成 \(\beta\) 和 \(\sigma\) 的函数。

### 7.2 物理动机

黑洞吸积流中很多耗散可能发生在 current sheet，但这些区域未必是极端 relativistic，也未必可以只由 \(b^2/\rho\) 描述。热压也会影响重联层粒子加热。

因此 `rowan` 同时使用：

- \(\sigma\)：磁能相对物质焓的大小。
- \(\beta\)：热压相对磁压的大小。

这比 `werner` 更细，因为它区分了“强磁化但热压也高”和“强磁化且磁压主导”的情况。

### 7.3 物理假设

主要假设包括：

- 耗散是 collisionless magnetic reconnection。
- 电子/质子加热比例主要由 upstream \(\beta\) 和 \(\sigma\) 控制。
- 局部 GRMHD cell 的 \(\beta\)、\(\sigma\) 可以近似代表 unresolved reconnection layer 的 upstream 状态。
- PIC 拟合可以作为 sub-grid closure。

### 7.4 物理图像

磁重联把磁能释放给粒子。磁能越强，电子越容易获得可观能量；但如果等离子体热压很高，粒子已有热运动会改变重联层中的能量分配。

代码中：

\[
p_p = (\gamma_p - 1)u, \qquad p_g = (\gamma - 1)u,
\]

\[
\beta = \frac{2p_p}{b^2}, \qquad \sigma = \frac{b^2}{\rho + u + p_g}.
\]

这里的 \(\sigma\) 用的是相对论焓密度风格的分母，而不是简单 \(b^2/\rho\)。这表示磁能相对总惯性和热焓的大小。

### 7.5 公式和推导脉络

代码定义：

\[
\beta_{\max} = \frac{0.25}{\sigma}.
\]

电子加热比例：

\[
f_e = \frac{1}{2} \exp\left[ - \frac{ \left(1 - \beta/\beta_{\max}\right)^{3.3} } { 1 + 1.2\sigma^{0.7} } \right].
\]

该表达式的结构可以这样理解：

- 前因子 \(1/2\)：强重联加热时电子和质子趋向更接近均分。
- \(\beta/\beta_{\max}\)：衡量热压环境相对当前磁化强度允许范围的位置。
- 指数抑制：当局部状态不利于电子加热时，电子份额被压低。
- \(1 + 1.2\sigma^{0.7}\)：磁化增强会减弱指数抑制，使电子份额升高。

### 7.6 适用与局限

适合：

- current sheet。
- trans-relativistic reconnection。
- MAD 中磁通管和盘体交界附近的强磁耗散。

局限：

- 依赖把 GRMHD cell 状态解释为重联 upstream 状态，这是 sub-grid 假设。
- 不直接描述湍流级联。
- 对 \(\beta\) 和 \(\sigma\) 的定义很敏感；改变定义会改变物理含义。

## 8. `sharma`

参考：

- Sharma, P., Quataert, E., Hammett, G. W., & Stone, J. M. 2007, *Electron Heating in Hot Accretion Flows*, ApJ.
- arXiv: <https://arxiv.org/abs/astro-ph/0703572>
- DOI: <https://doi.org/10.1086/520800>

### 8.1 物理背景

Sharma 等人的工作讨论 hot accretion flows 中电子加热问题。热吸积流是弱碰撞、低辐射效率环境，电子和离子可能长期维持不同温度。压力各向异性、微观不稳定性和有效粘性加热都会影响 \(Q_e/Q_i\)。

### 8.2 物理动机

该模型想保留一个简单事实：电子温度本身会影响后续电子加热。如果电子已经很冷，它们获得的相对加热比例不同；如果电子较热，电子 heating channel 的效率也会改变。

### 8.3 物理假设

当前代码采用非常简化的温度比闭合：

- 不显式使用 \(\beta\)。
- 不显式使用 \(\sigma\)。
- 不区分湍流、重联或冲击。
- 只用 \(T_e/T_p\) 决定 \(Q_e/Q_i\)。

### 8.4 物理图像

这是一个“温度比反馈”模型。电子越热，电子通道越容易继续获得热量；电子越冷，离子通道相对更强。

### 8.5 公式和推导脉络

代码使用：

\[
\frac{Q_e}{Q_i} = 0.33 \sqrt{\frac{T_e}{T_p}}.
\]

因此：

\[
f_e = \frac{Q_e}{Q_i + Q_e} = \frac{Q_e/Q_i}{1 + Q_e/Q_i}.
\]

把上式代入：

\[
f_e = \frac{ 0.33\sqrt{T_e/T_p} } { 1 + 0.33\sqrt{T_e/T_p} }.
\]

如果 \(T_e/T_p\) 很小：

\[
f_e \propto \sqrt{\frac{T_e}{T_p}},
\]

电子获得的比例较小。如果 \(T_e/T_p\) 增大，\(f_e\) 单调升高。

### 8.6 适用与局限

适合：

- 需要简单温度比反馈。
- 不希望模型显式依赖磁场强度。
- 做热吸积流两温演化的简化闭合。

局限：

- 不能识别磁重联区。
- 不能表达低 \(\beta\) 和高 \(\beta\) 湍流加热差异。
- 当前实现只保留一个简化关系，没有完整实现原论文中更丰富的压力各向异性物理。

## 9. 模型之间的物理分工

| 模型 | 类型 | 主要物理机制 | 控制量 | 物理图像 |
| --- | --- | --- | --- | --- |
| `constant` | 参数模型 | 无指定机制 | \(f_{e,0}\) | 固定比例分账 |
| `howes` | 湍流模型 | Alfvénic cascade + kinetic damping | \(\beta, T_i/T_e\) | 湍流级联到 kinetic scale 后按 damping channel 分配 |
| `kawazura` | 湍流模型 | Collisionless Alfvénic turbulence | \(\beta, T_i/T_e\) | 低 \(\beta\) 电子更强，高 \(\beta\) 离子更强 |
| `werner` | 重联模型 | Relativistic electron-proton reconnection | \(\sigma\) | 磁化越强，电子份额越接近均分 |
| `rowan` | 重联模型 | Trans-relativistic reconnection | \(\beta, \sigma\) | 热压和磁化共同决定重联加热分配 |
| `sharma` | 温度比模型 | Hot accretion flow heating closure | \(T_e/T_i\) | 电子温度反馈决定后续电子加热 |

## 10. 极限行为对比

从使用者角度，最重要的是知道模型在典型极限下会把电子加热推向哪里。

### 10.1 低 \(\beta\) 与高 \(\beta\)

`howes` 和 `kawazura` 都显式依赖 \(\beta\)。它们通常表达类似趋势：

\[
\beta \ll 1 \Rightarrow f_e \text{ 增大},
\]

\[
\beta \gg 1 \Rightarrow f_e \text{ 减小}.
\]

物理解释是：低 \(\beta\) 磁场强，湍流到 kinetic scale 后更容易通过电子相关通道耗散；高 \(\beta\) 热压强，离子 damping 更强。

`werner` 和 `sharma` 不显式依赖 \(\beta\)。`rowan` 依赖 \(\beta\)，但它是重联图像，不应直接和湍流模型的 beta 趋势等同。

### 10.2 低 \(\sigma\) 与高 \(\sigma\)

`werner` 和 `rowan` 对 \(\sigma\) 敏感。

对 `werner`：

\[
\sigma \to 0 \Rightarrow f_e \to 0.25,
\]

\[
\sigma \to \infty \Rightarrow f_e \to 0.5.
\]

物理图像是：磁化增强时，重联释放的磁能更容易让电子获得接近离子的能量份额。

`rowan` 中，高 \(\sigma\) 会减弱指数抑制，使 \(f_e\) 更接近 \(0.5\)。但它还同时受 \(\beta\) 控制，因此不能只用单一 \(\sigma\) 极限判断。

### 10.3 冷电子与热电子

温度比相关模型包括 `howes`、`kawazura` 和 `sharma`。

`sharma` 最直接：

\[
f_e = \frac{ 0.33\sqrt{T_e/T_p} } { 1 + 0.33\sqrt{T_e/T_p} }.
\]

因此：

\[
T_e/T_p \to 0 \Rightarrow f_e \to 0,
\]

\[
T_e/T_p \text{ 增大} \Rightarrow f_e \text{ 单调增大}.
\]

`howes` 和 `kawazura` 的温度比依赖嵌入在 \(Q_i/Q_e\) 的经验拟合中，不能简单归纳为单调关系；使用时应同时看 \(\beta\)。

### 10.4 固定比例模型

`constant` 不关心任何极限：

\[
\frac{\partial f_e}{\partial \beta} = \frac{\partial f_e}{\partial \sigma} = \frac{\partial f_e}{\partial (T_i/T_e)} = 0.
\]

这既是优点，也是缺点。它适合控制实验，不适合解释电子温度结构。

## 11. 模型选择流程

如果目标是数值验证，使用 `constant`。它最容易判断是否是两温更新、熵限制或守恒变量转换出了问题。

如果目标是盘体 MRI 湍流中的电子温度，使用 `howes` 或 `kawazura`。`howes` 更细但更复杂，`kawazura` 更简洁且更强调 \(\beta\) 控制。

如果目标是 current sheet、磁喷流或 MAD 中的磁耗散，使用 `rowan` 或 `werner`。`werner` 是磁化强度一参数模型，`rowan` 同时考虑热压和相对论磁化。

如果目标是保留温度比反馈，同时避免强依赖磁场处方，使用 `sharma`。

一个实用选择流程是：

1. 先确定耗散物理图像。
2. 如果是湍流，选 `howes` 或 `kawazura`。
3. 如果是重联，选 `werner` 或 `rowan`。
4. 如果不确定，先跑 `constant` 做基准，再比较其他模型。
5. 如果结果主要由温度比反馈控制，单独加入 `sharma` 对照。

不要把所有模型看成“哪个更高级”。它们代表不同物理假设。一个更复杂的模型如果用在错误耗散机制上，未必比简单模型更好。

## 12. 实现层面的注意事项

当前 `heating_model.h` 选择相信 fixer 已经处理 floor，因此模型内部没有再对 \(\rho\)、\(b^2\)、\(u\)、\(\beta\)、\(\sigma\) 做额外保护。这让公式更接近物理表达式，但也意味着：

- 调用 `apply` 前必须保证 \(\rho > 0\)。
- 对需要 \(b^2\) 的模型，必须保证 \(b^2\) 不为零。
- 对需要 \(T_e\) 或 \(T_p\) 的模型，必须保证温度尺度为正。
- 如果后续把这些函数用于 fixer 之前的状态，需要重新加入保护或在调用前检查状态。

此外，`rowan` 中的公式来自当前代码实现。文档按代码语义解释，而不是宣称完全等同于论文中每个符号的原始定义。若未来改变 \(\beta\) 或 \(\sigma\) 的定义，应同步修改本文档。

## 13. 参考文献

- Howes, G. G. 2010, *A prescription for the turbulent heating of astrophysical plasmas*, MNRAS Letters, arXiv:1009.4212, <https://arxiv.org/abs/1009.4212>.
- Kawazura, Y., Barnes, M., & Schekochihin, A. A. 2019, *Thermal disequilibration of ions and electrons by collisionless plasma turbulence*, PNAS, arXiv:1807.07702, <https://arxiv.org/abs/1807.07702>.
- Werner, G. R., Uzdensky, D. A., Begelman, M. C., Cerutti, B., & Nalewajko, K. 2018, *Non-thermal particle acceleration in collisionless relativistic electron-proton reconnection*, MNRAS, arXiv:1612.04493, <https://arxiv.org/abs/1612.04493>.
- Rowan, M. E., Sironi, L., & Narayan, R. 2017, *Electron and proton heating in trans-relativistic magnetic reconnection*, ApJ, arXiv:1708.04627, <https://arxiv.org/abs/1708.04627>.
- Sharma, P., Quataert, E., Hammett, G. W., & Stone, J. M. 2007, *Electron Heating in Hot Accretion Flows*, ApJ, arXiv:astro-ph/0703572, <https://arxiv.org/abs/astro-ph/0703572>.
