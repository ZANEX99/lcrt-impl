# LCRT 实现说明

本目录中的 Python 实验使用 LCT 域 Riesz directional multiplier 构造二维方向响应，并将响应接入 MNIST 分类网络。

| 文件 | 作用 | LCT 参数 |
|---|---|---|
| `mnist_lct_riesz.py` | 固定 LCRT 插件的分类模型 | 固定，且两个空间轴共享同一矩阵 |
| `mnist_learnableLCRT.py` | 可学习 LCRT 插件的分类模型 | 每个空间轴分别学习一个合法 LCT 矩阵 |
| `ker_lcrt.m` | MATLAB 参考函数，仅生成两个 multiplier/kernel | 从输入矩阵 `mx,my` 取得 `b_1,b_2` |

## 1. 二维 LCRT 定义

一般的可分离二维 LCT 使用一对轴向矩阵：

$$
A_x=\begin{bmatrix}a_x&b_x\\c_x&d_x\end{bmatrix},
\qquad
A_y=\begin{bmatrix}a_y&b_y\\c_y&d_y\end{bmatrix},
$$

且各自满足：

$$
a_xd_x-b_xc_x=1,
\qquad
a_yd_y-b_yc_y=1.
$$

两个 Riesz 响应建立在同一个二维 LCT 域上：

$$
Z=\mathscr{L}_{A_x,A_y}(X),
$$

$$
R_x=\mathscr{L}^{-1}_{A_x,A_y}\!\left(m_x Z\right),
\qquad
R_y=\mathscr{L}^{-1}_{A_x,A_y}\!\left(m_y Z\right).
$$

代码中的 directional multipliers 使用稳定化尺度 \(\tilde b_x,\tilde b_y\)：

$$
m_x(\omega_x,\omega_y)
=-i\frac{\omega_x/\tilde b_x}
{\sqrt{(\omega_x/\tilde b_x)^2+(\omega_y/\tilde b_y)^2+\varepsilon}},
$$

$$
m_y(\omega_x,\omega_y)
=-i\frac{\omega_y/\tilde b_y}
{\sqrt{(\omega_x/\tilde b_x)^2+(\omega_y/\tilde b_y)^2+\varepsilon}}.
$$

固定版使用 \(\tilde b=|b|+\varepsilon\)，可学习版使用可微形式
\(\tilde b=\sqrt{b^2+\varepsilon^2}\)。两者均使用 `torch.fft.fftfreq`
构造频率网格，并令零频点的 multiplier 为零。

参数共享原则是：

- \(A_x\) 与 \(A_y\) 可以不同；它们分别控制水平轴和垂直轴的 LCT。
- \(R_x\) 与 \(R_y\) 共享同一对 \(A_x,A_y\)，只在 multiplier \(m_x,m_y\) 上不同。
- 逆变换使用解析得到的 \(A_x^{-1},A_y^{-1}\)，不引入新的可学习参数。

## 2. 固定版本：`mnist_lct_riesz.py`

固定版本中的 `FixedLCT2D` 由配置值 `lct_alpha,lct_m,lct_q` 生成一个矩阵 \(A\)，并将矩阵元素保存为普通 `float`。该版本对两个空间轴使用同一个矩阵：

$$
A_x=A_y=A.
$$

插件中的计算流程为：

```text
xr -> LCT -> multiply by kx/ky -> inverse LCT -> rx, ry
   -> concat[xr, rx, ry, magnitude] -> learnable fusion
```

其中：

$$
\mathrm{magnitude}=\sqrt{r_x^2+r_y^2+\varepsilon}.
$$

固定内容：

- LCT 矩阵 \(A\)；
- 由 \(b\) 构造的 `kx, ky` multiplier。

可学习内容：

- `pre`、`spec_fuse` 和 `spa_branch` 中的卷积与 BatchNorm；
- 残差门控参数 `alpha, beta`；
- 分类网络其余参数。

当前模型使用 `magnitude`，未使用 local phase descriptor。

## 3. 可学习版本：`mnist_learnableLCRT.py`

可学习版本为每个 LCRT 插件学习两个轴向矩阵：

$$
A_x=A(\alpha_x,m_x,q_x),
\qquad
A_y=A(\alpha_y,m_y,q_y).
$$

每个矩阵由三个标量参数生成，其中 \(m=\exp(\log m)>0\)：

$$
A(\alpha,m,q)=
\begin{bmatrix}
m\cos\theta & m\sin\theta\\
-qm\cos\theta-\dfrac{\sin\theta}{m}
&
-qm\sin\theta+\dfrac{\cos\theta}{m}
\end{bmatrix},
\qquad
\theta=\frac{\alpha\pi}{2}.
$$

该参数化天然保证：

$$
\det A=ad-bc=1.
$$

因此，这不是直接独立学习 `a,b,c,d` 后再强制修正行列式，而是在合法 LCT 矩阵族内学习三个自由参数。

### 参数数量

每个 LCRT 插件的 LCT 可学习标量参数为：

| 矩阵 | 参数 |
|---|---|
| \(A_x\) | `alpha_x, log_m_x, q_x` |
| \(A_y\) | `alpha_y, log_m_y, q_y` |

所以每个插件新增：

$$
3+3=6
$$

个 LCT 标量参数。模型包含 `plugin1` 和 `plugin2`，因此整网新增 12 个 LCT 标量参数。

`rx` 和 `ry` 不各自学习矩阵；二者共享当前插件中的 \(A_x,A_y\)。逆矩阵由：

$$
A^{-1}=\begin{bmatrix}d&-b\\-c&a\end{bmatrix}
$$

直接生成，也不增加参数。

### Multiplier 是否可学习

`kx, ky` 不是独立的 `nn.Parameter`。每次前向传播中，代码从当前矩阵取得：

```python
bx, by = matrix_x[1], matrix_y[1]
```

再据此重新构造 multipliers。因此：

```text
学习 Ax, Ay -> bx, by 随训练变化 -> kx, ky 解析地随训练变化
```

它是参数依赖的自适应解析算子，而不是自由学习的频域 mask。

## 4. 与 MATLAB `ker_lcrt.m` 的关系

`ker_lcrt.m` 仅完成 multiplier/kernel 构造：

```matlab
b1 = mx(1,2);
b2 = my(1,2);
[dxker, dyker] = ker_lcrt(image, mx, my);
```

完整 MATLAB 流程需要在外部脚本中额外执行：

```text
LCT -> kernel multiplication -> inverse LCT
```

MATLAB 代码允许输入两个不同矩阵 `mx,my`，这与可学习 Python 版本使用 \(A_x,A_y\) 的一般二维形式相对应。

需要注意，两者不是数值等价实现：

| 项目 | Python 版本 | `ker_lcrt.m` |
|---|---|---|
| 频率网格 | `torch.fft.fftfreq`，包含正负频率与零频 | 直接使用数组下标 `1:m, 1:n` |
| 零频处理 | multiplier 显式置零 | 没有零频点处理 |
| 输出范围 | Python 插件中逆变换后取 `.real` | MATLAB 调用脚本中逆变换后取 `abs(...)` |

因此，MATLAB 文件可作为 multiplier 公式来源参考，但不能作为当前 Python 输出的逐点对照基准。

## 5. 复杂度

两个 Python 版本的 LCRT 插件均包含：

```text
1 次二维 forward LCT
2 次二维 inverse LCT
2 次 multiplier 逐点乘法
1 次 magnitude 构造和后续融合
```

若一次二维 LCT 的复杂度记为：

$$
\mathcal{T}=\mathcal{O}\!\left(BCHW(\log H+\log W)\right),
$$

则 LCRT 变换部分约为：

$$
3\mathcal{T}+\mathcal{O}(BCHW).
$$

可学习版本相对固定版本不改变 FFT/LCT 的渐近复杂度。它额外增加的是：

- 每个插件 6 个标量 LCT 参数；
- 从参数计算 \(A_x,A_y\) 和 multipliers 的少量逐点运算；
- 训练阶段通过 LCT 参数与 multiplier 路径进行反向传播的开销。

主要运行开销仍由三次二维 LCT/逆 LCT 主导。

## 6. 运行与输出

```bash
python mnist_lct_riesz.py
python mnist_learnableLCRT.py
```

当前代码保存的最佳 checkpoint 文件名为：

```text
best_mnist_lct_riesz.pth
best_minist_learnableLCRT.pth
```

第二个名称中的 `minist` 是当前代码中使用的文件名拼写。

## 总结

| 版本 | 轴向矩阵 | Multiplier | 关键含义 |
|---|---|---|---|
| 固定 Python 版 | \(A_x=A_y=A\)，固定 | 固定 | 学习如何融合固定 LCRT 响应 |
| 可学习 Python 版 | \(A_x,A_y\) 分别由三参数学习 | 由当前 \(b_x,b_y\) 解析更新 | 在 LCT 结构约束内自适应方向响应 |
| MATLAB `ker_lcrt.m` | 由外部提供 `mx,my` | 由输入矩阵生成 | 仅为 kernel 构造函数，不是完整 LCRT 模块 |
