# physics/thermal.py  ── 溫度場與熱梯度
# 對應 PDF §5 熱力模型（Eq.4: T(y)=T0-ΔT*y^2, Eq.5: F_th=-α∇T）
#
import taichi as ti
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg


# ── 溫度場（GPU 常駐，由 forcing.py 讀取）──
T_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))

# ── 溫度梯度場（供分析用）──
dTdy_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))


@ti.kernel
def init_temperature_kernel(T0: float, delta_T: float):
    """
    初始化拋物線溫度剖面（對應 PDF §5 Eq.4）：
      T(y) = T0 - ΔT * ((y - NY/2) / (NY/2))^2
    赤道（y=NY/2）最熱，極地（y=0 or NY-1）最冷。
    """
    ny_half = float(cfg.NY // 2)
    for y, x in T_field:
        y_norm    = (float(y) - ny_half) / ny_half   # ∈ [-1, 1]
        T_field[y, x] = T0 - delta_T * y_norm * y_norm


@ti.kernel
def advect_temperature_kernel(dt: float):
    """
    溫度平流（簡化版：用宏觀速度場平流，一階 upwind）。
    需要 ux_field / uy_field 已更新（從 collision.py 匯入）。
    
    ∂T/∂t + u·∇T = κ∇²T  （此處略去擴散項，由黏滯隱式耗散代替）
    """
    from core.collision import ux_field, uy_field
    for y, x in T_field:
        ux = ux_field[y, x]
        uy = uy_field[y, x]

        # Upwind 差分（一階精度，穩定但有數值耗散）
        if ux > 0.0:
            dTdx = T_field[y, x] - T_field[y, (x-1+cfg.NX) % cfg.NX]
        else:
            dTdx = T_field[y, (x+1) % cfg.NX] - T_field[y, x]

        if uy > 0.0:
            dTdy = T_field[y, x] - T_field[(y-1+cfg.NY) % cfg.NY, x]
        else:
            dTdy = T_field[(y+1) % cfg.NY, x] - T_field[y, x]

        T_field[y, x] -= dt * (ux * dTdx + uy * dTdy)


@ti.kernel
def relax_temperature_kernel(T0: float, delta_T: float, relax_rate: float):
    """
    溫度場弱鬆弛（Newtonian cooling）：讓溫度場緩慢恢復初始剖面，
    防止長時間模擬溫差消失。
      T_new = T + relax_rate * (T_eq - T)
    relax_rate ≈ 1e-4（每步）
    """
    ny_half = float(cfg.NY // 2)
    for y, x in T_field:
        y_norm = (float(y) - ny_half) / ny_half
        T_eq   = T0 - delta_T * y_norm * y_norm
        T_field[y, x] += relax_rate * (T_eq - T_field[y, x])


@ti.kernel
def compute_dTdy_kernel():
    """計算 ∂T/∂y（中央差分），存入 dTdy_field 供分析用"""
    for y, x in dTdy_field:
        yp = (y + 1) % cfg.NY
        ym = (y - 1 + cfg.NY) % cfg.NY
        dTdy_field[y, x] = 0.5 * (T_field[yp, x] - T_field[ym, x])


def init_temperature():
    """Python 層初始化入口"""
    init_temperature_kernel(cfg.T0, cfg.DELTA_T)
