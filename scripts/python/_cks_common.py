#!/usr/bin/env python3
"""CKS 绘图公共模块 —— 仿照 AthenaK plot_slice.py 风格抽取的共享工具。

为 scripts/python/ 下的 CKS 绘图脚本提供统一的 CLI 参数、matplotlib 配置、
视界/能层/网格绘制、色标处理等功能。
"""

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# LaTeX 物理标签映射（与 plot_slice.py 的 set_labels 保持一致）
# ---------------------------------------------------------------------------

PHYS_LABELS = {
    "density": r"$\rho$",
    "dens": r"$\rho$",
    "eint": r"$u_\mathrm{gas}$",
    "energy": r"$u_\mathrm{gas}$",
    "entropy": r"$s$",
    "electron_entropy": r"$s_e$",
    "velx": r"$u^{x^\prime}$",
    "vely": r"$u^{y^\prime}$",
    "velz": r"$u^{z^\prime}$",
    "weighted_velocity_0": r"$u^{x^\prime}$",
    "weighted_velocity_1": r"$u^{y^\prime}$",
    "weighted_velocity_2": r"$u^{z^\prime}$",
    "bcc1": r"$B^x$",
    "bcc2": r"$B^y$",
    "bcc3": r"$B^z$",
    "pgas": r"$p_\mathrm{gas}$",
    "pmag": r"$p_\mathrm{mag}$",
    "T": r"$T$ (K)",
    "temperature": r"$T$ (K)",
    "sigma": r"$\sigma$",
    "beta_inv": r"$\beta^{-1}$",
    "beta": r"$\beta$",
}

# Temp files that could be either ion or electron
PHYS_LABELS["ion_temperature"] = r"$T_i$ (K)"
PHYS_LABELS["electron_temperature"] = r"$T_e$ (K)"


def label_for(field_name, notex=False):
    """返回字段对应的 LaTeX 标签；若未注册则返回原名。"""
    if notex or field_name not in PHYS_LABELS:
        return field_name
    return PHYS_LABELS[field_name]


# ---------------------------------------------------------------------------
# matplotlib 环境配置
# ---------------------------------------------------------------------------

def setup_plot_env(notex=False, output_file=None):
    """配置 matplotlib 后端及 LaTeX。

    Args:
        notex: 禁用 LaTeX 排版。
        output_file: 若非 'show' 则使用 'agg' 后端。
    """
    import matplotlib
    if output_file is not None and output_file != "show":
        matplotlib.use("agg")
    if not notex:
        matplotlib.rc("text", usetex=True)


# ---------------------------------------------------------------------------
# 统一 CLI 参数注册
# ---------------------------------------------------------------------------

def add_common_args(parser):
    """向 argparse.ArgumentParser 注册统一的绘图参数。"""
    import argparse as _ap  # noqa: F811

    # 坐标范围
    group_bounds = parser.add_argument_group("plot bounds")
    group_bounds.add_argument(
        "--r_max", type=float, default=None,
        help="half-width of plot in both coordinates, centered at origin",
    )
    group_bounds.add_argument(
        "--x1_min", type=float, default=None,
        help="horizontal left edge of plot",
    )
    group_bounds.add_argument(
        "--x1_max", type=float, default=None,
        help="horizontal right edge of plot",
    )
    group_bounds.add_argument(
        "--x2_min", type=float, default=None,
        help="vertical bottom edge of plot",
    )
    group_bounds.add_argument(
        "--x2_max", type=float, default=None,
        help="vertical top edge of plot",
    )

    # 色标
    group_color = parser.add_argument_group("color scale")
    group_color.add_argument(
        "--norm", choices=("linear", "log"), default="log",
        help="colormap normalization (default: log)",
    )
    group_color.add_argument(
        "--vmin", type=float, default=None,
        help="colormap minimum",
    )
    group_color.add_argument(
        "--vmax", type=float, default=None,
        help="colormap maximum",
    )
    group_color.add_argument(
        "-c", "--cmap", default="jet",
        help="Matplotlib colormap name (default: jet)",
    )

    # GR 特性
    group_gr = parser.add_argument_group("GR features")
    group_gr.add_argument(
        "--horizon", action="store_true",
        help="outline the black hole event horizon",
    )
    group_gr.add_argument(
        "--horizon_mask", action="store_true", default=True,
        help="mask (cover) the black hole event horizon (default: on when kerr-a != 0)",
    )
    group_gr.add_argument(
        "--no-horizon-mask", action="store_false", dest="horizon_mask",
        help="do not mask the black hole event horizon",
    )
    group_gr.add_argument(
        "--horizon_color", default="k",
        help="color string for horizon outline (default: k)",
    )
    group_gr.add_argument(
        "--horizon_mask_color", default="k",
        help="color string for horizon mask (default: k)",
    )
    group_gr.add_argument(
        "--ergosphere", action="store_true",
        help="outline the black hole ergosphere",
    )
    group_gr.add_argument(
        "--ergosphere_color", default="gray",
        help="color string for ergosphere marker (default: gray)",
    )

    # 网格
    group_grid = parser.add_argument_group("grid / block overlay")
    group_grid.add_argument(
        "--grid", action="store_true",
        help="overlay domain-decomposition block boundaries",
    )
    group_grid.add_argument(
        "--grid_color", default="gray",
        help="color for grid overlay (default: gray)",
    )
    group_grid.add_argument(
        "--grid_alpha", type=float, default=0.5,
        help="opacity for grid overlay (default: 0.5)",
    )

    # 输出
    group_out = parser.add_argument_group("output")
    group_out.add_argument(
        "--dpi", type=float, default=150,
        help="output image resolution (default: 150)",
    )
    group_out.add_argument(
        "--notex", action="store_true",
        help="disable LaTeX typesetting of labels",
    )


# ---------------------------------------------------------------------------
# 坐标范围解析
# ---------------------------------------------------------------------------

def get_plot_bounds(args_dict, default_x1_min, default_x1_max,
                    default_x2_min, default_x2_max):
    """根据 r_max / x1_* / x2_* 参数解析最终绘图范围。

    Args:
        args_dict: argparse 参数字典。
        default_*: 从数据中自动检测的默认范围。

    Returns:
        (x1_min, x1_max, x2_min, x2_max)
    """
    if args_dict.get("r_max") is not None:
        r = args_dict["r_max"]
        return -r, r, -r, r

    x1_min = args_dict.get("x1_min", None)
    x1_max = args_dict.get("x1_max", None)
    x2_min = args_dict.get("x2_min", None)
    x2_max = args_dict.get("x2_max", None)

    if x1_min is None:
        x1_min = default_x1_min
    if x1_max is None:
        x1_max = default_x1_max
    if x2_min is None:
        x2_min = default_x2_min
    if x2_max is None:
        x2_max = default_x2_max

    return x1_min, x1_max, x2_min, x2_max


# ---------------------------------------------------------------------------
# 色标归一化
# ---------------------------------------------------------------------------

def compute_color_norm(args_dict, data):
    """根据 --norm / --vmin / --vmax 参数创建 matplotlib Normalize 对象。

    Args:
        args_dict: argparse 参数字典。
        data: 用于自动推断范围的 ndarray。

    Returns:
        (norm, vmin, vmax) — norm 是 Normalize 实例，
        vmin/vmax 是实际使用的范围（可能为 None，当由 norm 内部处理时）。
    """
    import matplotlib.colors as mcolors

    norm_name = args_dict.get("norm", "log")
    vmin = args_dict.get("vmin", None)
    vmax = args_dict.get("vmax", None)

    finite = data[np.isfinite(data)]

    if norm_name == "log":
        positive = finite[finite > 0.0] if finite.size > 0 else finite
        if vmin is None:
            vmin = float(np.nanmin(positive)) if positive.size > 0 else 1e-30
        if vmax is None:
            vmax = float(np.nanmax(positive)) if positive.size > 0 else 1.0
        norm = mcolors.LogNorm(vmin, vmax)
        return norm, None, None
    else:
        if vmin is None:
            vmin = float(np.nanmin(finite)) if finite.size > 0 else 0.0
        if vmax is None:
            vmax = float(np.nanmax(finite)) if finite.size > 0 else 1.0
        norm = mcolors.Normalize(vmin, vmax)
        return norm, None, None


# ---------------------------------------------------------------------------
# Kerr 黑洞视界 / 能层
# ---------------------------------------------------------------------------

def kerr_r_horizon(kerr_a):
    """返回 Kerr 黑洞事件视界半径 r_h (Boyer-Lindquist)。"""
    return 1.0 + np.sqrt(max(0.0, 1.0 - kerr_a ** 2))


def draw_horizon(ax, kerr_a, mask=False, outline=True,
                 mask_color="k", outline_color="k"):
    """在 xz 子午面上绘制 Kerr 事件视界（椭圆）。

    在 CKS 笛卡尔坐标的 xz 平面 (y=0) 上，视界表现为椭圆：
      width  = 2 * sqrt(r_h^2 + a^2)
      height = 2 * sqrt(r_h^2 + a^2) / sqrt(1 + a^2 / r_h^2)

    Args:
        ax: matplotlib Axes。
        kerr_a: 黑洞无量纲自旋参数。
        mask: 是否用实心椭圆遮罩视界内部。
        outline: 是否绘制视界轮廓椭圆。
        mask_color: 遮罩颜色。
        outline_color: 轮廓颜色。
    """
    import matplotlib.patches as mpatches

    if kerr_a == 0.0:
        # Schwarzschild: 圆形
        r_h = 2.0
        if mask:
            ax.add_patch(mpatches.Circle(
                (0.0, 0.0), r_h,
                facecolor=mask_color, edgecolor="none", zorder=5,
            ))
        if outline:
            ax.add_patch(mpatches.Circle(
                (0.0, 0.0), r_h,
                linestyle="-", linewidth=1.0,
                facecolor="none", edgecolor=outline_color, zorder=6,
            ))
        return

    a2 = kerr_a ** 2
    r_h = kerr_r_horizon(kerr_a)

    # xz 平面 (y=0, 即 location=0) 的视界椭圆半轴
    # 参照 plot_slice.py dimension='y' 分支公式
    r_sq_plus_a2 = r_h ** 2 + a2
    full_width = 2.0 * np.sqrt(r_sq_plus_a2)
    full_height = 2.0 * np.sqrt(r_sq_plus_a2 / (1.0 + a2 / r_h ** 2))

    if mask:
        ax.add_patch(mpatches.Ellipse(
            (0.0, 0.0), full_width, full_height,
            facecolor=mask_color, edgecolor="none", zorder=5,
        ))
    if outline:
        ax.add_patch(mpatches.Ellipse(
            (0.0, 0.0), full_width, full_height,
            linestyle="-", linewidth=1.0,
            facecolor="none", edgecolor=outline_color, zorder=6,
        ))


def draw_ergosphere_xz(ax, kerr_a, color="gray", num_points=129):
    """在 xz 子午面上绘制 Kerr 能层边界。

    在 CKS 坐标的 xz 平面 (y=0) 上，能层满足:
      r^2 - 2r + a^2 z^2 / (r^2 + a^2 z^2 / r^2) = 0
    简化为 x-z 空间的参数曲线。

    Args:
        ax: matplotlib Axes。
        kerr_a: 黑洞无量纲自旋参数。
        color: 线条颜色。
        num_points: 曲线采样点数。
    """
    from scipy.optimize import brentq
    import matplotlib.pyplot as plt

    a2 = kerr_a ** 2
    r_h = kerr_r_horizon(kerr_a)

    # ergosphere 在赤道面最远可达 2 r_g (a→1 极限)
    max_xy = np.sqrt(4.0 + a2) if 4.0 + a2 > 0 else 2.0
    w = np.linspace(0.0, max_xy, num_points)

    z_upper = np.empty_like(w)
    for i, w_val in enumerate(w):
        def residual_h(z_val):
            rr2 = w_val ** 2 + z_val ** 2
            r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2
                        + 4.0 * a2 * z_val ** 2))
            return r2 - r_h ** 2

        # 找到 horizon 以上的 z 起点
        if residual_h(0.0) < 0.0:
            try:
                z_min = brentq(residual_h, 0.0, 2.0)
            except ValueError:
                z_min = 0.0
        else:
            z_min = 0.0

        def residual_ergo(z_val):
            rr2 = w_val ** 2 + z_val ** 2
            r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2
                        + 4.0 * a2 * z_val ** 2))
            return r2 ** 2 - 2.0 * r2 ** 1.5 + a2 * z_val ** 2

        if residual_ergo(z_min) <= 0.0:
            try:
                z_upper[i] = brentq(residual_ergo, z_min, 2.0)
            except ValueError:
                z_upper[i] = 0.0
        else:
            z_upper[i] = 0.0

    xy = w
    x_plot = np.concatenate((-xy[::-1], xy))
    z_plot = np.concatenate((z_upper[::-1], z_upper))
    x_plot = np.concatenate((x_plot, x_plot[::-1]))
    z_plot = np.concatenate((z_plot, -z_plot[::-1]))

    ax.plot(x_plot, z_plot, linestyle="-", linewidth=1.0, color=color, zorder=0)


# ---------------------------------------------------------------------------
# CKS 对数坐标 → 物理坐标变换
# ---------------------------------------------------------------------------

def cks_to_physical(x1, x2, r0=0.0, kerr_h=0.0):
    """将 CKS 逻辑坐标 (x1, x2) 转换为物理 (x, z) 坐标。

    r = exp(x1) + r0
    theta = pi/2 * (x2 + 1) + h/2 * sin(pi * (x2 + 1))
    x = r * sin(theta)
    z = r * cos(theta)

    Args:
        x1: 1D 或 2D ndarray of x1 逻辑坐标。
        x2: 1D 或 2D ndarray of x2 逻辑坐标。
        r0: 径向偏移量。
        kerr_h: 修正极角映射参数。

    Returns:
        (x_physical, z_physical): 与输入同形的 ndarray。
    """
    x1g, x2g = np.broadcast_arrays(np.asarray(x1), np.asarray(x2))
    r = np.exp(x1g) + r0
    theta = 0.5 * np.pi * (x2g + 1.0) + 0.5 * kerr_h * np.sin(np.pi * (x2g + 1.0))
    x_phys = r * np.sin(theta)
    z_phys = r * np.cos(theta)
    return x_phys, z_phys


# ---------------------------------------------------------------------------
# 面坐标外推（pcolormesh 需要）
# ---------------------------------------------------------------------------

def cell_centers_to_faces(centers):
    """从 cell-center 坐标外推出 face 坐标。

    内部面: f_i = 0.5 * (c_{i-1} + c_i)
    边界面: 由最近两个中心的线性外推得到。

    Args:
        centers: 1D ndarray of length N

    Returns:
        1D ndarray of length N+1
    """
    if len(centers) < 2:
        raise ValueError("Need at least 2 cell centers")
    faces = np.empty(len(centers) + 1, dtype=centers.dtype)
    faces[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    # 外推边界
    dx0 = centers[1] - centers[0]
    faces[0] = centers[0] - 0.5 * dx0
    dx1 = centers[-1] - centers[-2]
    faces[-1] = centers[-1] + 0.5 * dx1
    return faces


# ---------------------------------------------------------------------------
# 网格块边界绘制
# ---------------------------------------------------------------------------

def draw_block_grid(ax, xf_list, zf_list, color="gray", alpha=0.5):
    """为每个网格块绘制边界矩形。

    Args:
        ax: matplotlib Axes。
        xf_list: 每块的 x 面坐标列表。
        zf_list: 每块的 z 面坐标列表。
        color: 线条颜色。
        alpha: 透明度。
    """
    import matplotlib.patches as mpatches

    for xf, zf in zip(xf_list, zf_list):
        rect = mpatches.Rectangle(
            (xf[0], zf[0]),
            xf[-1] - xf[0],
            zf[-1] - zf[0],
            linewidth=0.5,
            linestyle="-",
            edgecolor=color,
            facecolor="none",
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(rect)
