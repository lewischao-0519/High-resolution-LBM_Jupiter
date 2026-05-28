#外掛工具
import taichi as ti
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.collision import Fx_field, Fy_field, ux_field, uy_field, rho_field, f,noise_zonal
from core.lattice import feq_single

#AR噪音場初始化（於main中調用）
def init_noise_fields():
    noise_zonal.fill(0.0)
    Fx_field.fill(0.0)
    Fy_field.fill(0.0)

#柯氏力+阻尼+非破壞性field更新（於main中調用）
@ti.kernel
def apply_coriolis_drag_update_f(f0: float, beta: float, epsilon: float):
    ny_half = cfg.NY // 2
    for y, x in ux_field:
        rho    = rho_field[y, x]
        ux_old = ux_field[y, x]
        uy_old = uy_field[y, x]

        #柯氏力
        dy    = float(y - ny_half)
        f_cor = f0 + beta * dy
        a     = f_cor * 0.5          # a = f*dt/2，dt=1
        det   = 1.0 + a * a
        ux_rot = ((1.0 - a*a) * ux_old + 2.0 * a * uy_old) / det
        uy_rot = ((1.0 - a*a) * uy_old - 2.0 * a * ux_old) / det

        #阻尼
        ux_new = ux_rot * (1.0 - epsilon)
        uy_new = uy_rot * (1.0 - epsilon)

        ux_field[y, x] = ux_new
        uy_field[y, x] = uy_new

        #非破壞性field更新
        for i in ti.static(range(9)):
            delta_feq = feq_single(i, rho, ux_new, uy_new) \
                      - feq_single(i, rho, ux_old, uy_old)
            f[i, y, x] = ti.max(f[i, y, x] + delta_feq, 1e-12)

#AR緯向噪音field更新（只注入 k_x=0，大尺度噴流）（於main中調用）
@ti.kernel
def update_zonal_noise(alpha: float, sigma: float):
    for y in noise_zonal:
        noise_zonal[y] = alpha * noise_zonal[y] + sigma * ti.randn(ti.f32)

#AR緯向噪音field注入（只注入 k_x=0，大尺度噴流）（於main中調用）
@ti.kernel
def apply_zonal_ar1_forcing():
    for y, x in ux_field:
        rho    = rho_field[y, x]
        ux_old = ux_field[y, x]
        uy_old = uy_field[y, x]

        #速度修正（只注入緯向分量）
        ux_new = ux_old + 0.5 * noise_zonal[y] / rho
        uy_new = uy_old

        #安全限幅（避免超過 LBM 穩定極限 |u| < 0.3）
        u2 = ux_new * ux_new + uy_old * uy_old
        if u2 > 0.09:
            scale  = 0.3 / ti.sqrt(u2)
            ux_new = ux_new * scale
            uy_new = uy_new * scale

        ux_field[y, x] = ux_new
        uy_field[y, x] = uy_new  

        #非破壞性field更新
        for i in ti.static(range(9)):
            delta_feq = feq_single(i, rho, ux_new, uy_old) \
                      - feq_single(i, rho, ux_old, uy_old)
            f[i, y, x] = ti.max(f[i, y, x] + delta_feq, 1e-12)