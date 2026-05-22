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
from physics.thermal import T_field   # 溫度場由 thermal.py 維護


@ti.kernel
def update_forcing_kernel(f0: float, beta: float, alpha_t: float):
    """
    計算每個格點的合力並寫入 Fx_field / Fy_field。

    力的組成（對應 PDF §3, Eq.1）：
      F = F_Coriolis + F_thermal
    
    Coriolis（β-plane，對應 Eq.2）：
      f(y) = f0 + β * (y - NY/2)          ← 緯度偏移，赤道在中央
      F_cor_x = +f(y) * uy                ← z×u 的 x 分量（2D投影）
      F_cor_y = -f(y) * ux                ← z×u 的 y 分量
    
    熱力（對應 §5 Eq.5）：
      F_th = -α * ∇T  ← 此處簡化為 y 方向浮力
      F_th_y = alpha_t * (T - T0)        ← 浮力（暖→上升）
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

        # 熱力浮力（簡化：僅 y 方向）
        T_loc   = T_field[y, x]
        fy_th   = alpha_t * (T_loc - cfg.T0)

        Fx_field[y, x] = fx_cor
        Fy_field[y, x] = fy_cor + fy_th


def apply_noise_perturbation(step: int):
    """
    在初始階段（step < 500）注入小隨機噪音到外力場，
    加速帶狀流不穩定性觸發（對應 PDF §7 ε 參數）。
    """
    if step >= 500:
        return
    from core.collision import Fx_field, Fy_field
    ny, nx = cfg.NY, cfg.NX
    rng = np.random.default_rng(step)
    amp = cfg.NOISE_AMP * (1.0 - step / 500.0)   # 逐漸衰減
    noise = rng.uniform(-amp, amp, size=(ny, nx)).astype(np.float32)
    # 以 NumPy 加噪後重新上傳（只在初期，不影響效能）
    fx_np = Fx_field.to_numpy() + noise
    fy_np = Fy_field.to_numpy() + noise
    Fx_field.from_numpy(fx_np)
    Fy_field.from_numpy(fy_np)
