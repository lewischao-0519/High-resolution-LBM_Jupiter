# core/forcing.py  ── Coriolis 力 + 熱力 + 噪音外力
# 對應 PDF §3 控制方程（Eq.1-2）及 §5 熱力模型（Eq.4-5）
#
# 外力分量寫入 collision.py 的 Fx_field / Fy_field，
# 由 bgk_collision_kernel 在碰撞步中透過 Guo's scheme 施加。
#
import taichi as ti
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.collision import Fx_field, Fy_field, ux_field, uy_field, rho_field

@ti.kernel
def update_forcing_kernel(f0: float, beta: float):
    """
    計算每個格點的合力並寫入 Fx_field / Fy_field。

    力的組成（對應 PDF §3, Eq.1）：
      F = F_Coriolis + F_thermal
    
    Coriolis（β-plane）：
      f(y) = f0 + β * (y - NY/2)          ← 緯度偏移，赤道在中央
      F_cor_x = +f(y) * uy                ← z×u 的 x 分量（2D投影）
      F_cor_y = -f(y) * ux                ← z×u 的 y 分量
    """
    ny_half = cfg.NY // 2
    for y, x in rho_field:
        ux = ux_field[y, x]
        uy = uy_field[y, x]

        # β-plane Coriolis
        dy      = float(y - ny_half)
        f_cor   = f0 + beta * dy
        fx_cor  =  f_cor * uy
        fy_cor  = -f_cor * ux
        Fx_field[y, x] = fx_cor  # 科氏力 x 分量
        Fy_field[y, x] = fy_cor # 科氏力 y 分量