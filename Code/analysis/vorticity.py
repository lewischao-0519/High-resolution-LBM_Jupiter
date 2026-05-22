# analysis/vorticity.py  ── 渦度計算
# ω_z = ∂uy/∂x - ∂ux/∂y（對應 PDF §9 預期結果：大尺度渦流）
#
import taichi as ti
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.collision import ux_field, uy_field

# ── 渦度場（GPU 常駐）──
vorticity_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))


@ti.kernel
def compute_vorticity_kernel():
    """
    中央差分計算渦度：
      ω_z = (uy[x+1] - uy[x-1]) / 2 - (ux[y+1] - ux[y-1]) / 2
    週期邊界（x, y 兩方向均週期）。
    """
    for y, x in vorticity_field:
        xp = (x + 1) % cfg.NX
        xm = (x - 1 + cfg.NX) % cfg.NX
        yp = (y + 1) % cfg.NY
        ym = (y - 1 + cfg.NY) % cfg.NY

        duy_dx = 0.5 * (uy_field[y, xp] - uy_field[y, xm])
        dux_dy = 0.5 * (ux_field[yp, x] - ux_field[ym, x])
        vorticity_field[y, x] = duy_dx - dux_dy


def get_vorticity_numpy() -> np.ndarray:
    """計算渦度並以 NumPy 陣列回傳"""
    compute_vorticity_kernel()
    return vorticity_field.to_numpy()
