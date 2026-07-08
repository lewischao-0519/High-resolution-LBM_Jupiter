#外掛工具
import taichi as ti
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.collision import Fx_field, Fy_field, ux_field, uy_field, rho_field, f, noise_x, noise_y, noise_zonal
from core.lattice import feq_single

# ── 強迫的空間相關結構 ──
# 對每格獨立的白噪音做分離式高斯低通濾波（x 方向週期性、y 方向 clamp 邊界），
# 把功率集中到 config.py 定義的 K_F 附近，避免白噪音殼層模態數∝k 導致能量
# 集中在 Nyquist（耗散尺度）附近。詳見 config.py 的 K_F / FORCE_SIGMA 等定義。
_R = cfg.FORCE_RADIUS
_kernel_w = ti.field(ti.f32, shape=2 * _R + 1)
_kernel_w.from_numpy(cfg.FORCE_KERNEL)

#濾波暫存場：raw=未濾波白噪音，mid=x方向（緯向，週期性）濾波後
_raw_x = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
_raw_y = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
_mid_x = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
_mid_y = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))

#AR噪音場初始化（於main中調用）
def init_noise_fields():
    noise_x.fill(0.0)
    noise_y.fill(0.0)
    noise_zonal.fill(0.0)
    _raw_x.fill(0.0)
    _raw_y.fill(0.0)
    _mid_x.fill(0.0)
    _mid_y.fill(0.0)
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

@ti.kernel
def _draw_raw_noise():
    """每格獨立抽樣（delta-correlated 白噪音，單位方差）。"""
    for y, x in _raw_x:
        _raw_x[y, x] = ti.randn(ti.f32)
        _raw_y[y, x] = ti.randn(ti.f32)


@ti.kernel
def _blur_x_pass():
    """高斯低通濾波第一段：緯向（x），週期性邊界。"""
    for y, x in _mid_x:
        acc_x = 0.0
        acc_y = 0.0
        for k in ti.static(range(-_R, _R + 1)):
            xx = (x + k + cfg.NX) % cfg.NX
            w  = _kernel_w[k + _R]
            acc_x += w * _raw_x[y, xx]
            acc_y += w * _raw_y[y, xx]
        _mid_x[y, x] = acc_x
        _mid_y[y, x] = acc_y


@ti.kernel
def _blur_y_pass_and_update(alpha: float, sigma: float):
    """高斯低通濾波第二段：徑向（y，clamp 邊界，非週期）＋ AR(1) 遞迴更新（融合以省一個暫存場）。"""
    for y, x in noise_x:
        acc_x = 0.0
        acc_y = 0.0
        for k in ti.static(range(-_R, _R + 1)):
            yy = ti.min(ti.max(y + k, 0), cfg.NY - 1)
            w  = _kernel_w[k + _R]
            acc_x += w * _mid_x[yy, x]
            acc_y += w * _mid_y[yy, x]
        filt_x = acc_x * cfg.FORCE_NORM
        filt_y = acc_y * cfg.FORCE_NORM
        noise_x[y, x] = alpha * noise_x[y, x] + sigma * filt_x
        noise_y[y, x] = alpha * noise_y[y, x] + sigma * filt_y


#AR噪音field更新（小尺度、各向同性、空間相關集中於 K_F 附近）（於main中調用）
def update_2d_noise(alpha: float, sigma_2d: float):
    _draw_raw_noise()
    _blur_x_pass()
    _blur_y_pass_and_update(alpha, sigma_2d)


@ti.kernel
def _update_zonal_ar1(alpha: float, sigma_zonal: float):
    """1D 緯向 AR(1) 噪音：每列（y）獨立演化，同一緯度各格共用同一值。"""
    for y in noise_zonal:
        noise_zonal[y] = alpha * noise_zonal[y] + sigma_zonal * ti.randn(ti.f32)


#1D 緯向 AR(1) 噪音更新（大尺度帶狀強迫）（於main中調用）
def update_zonal_noise(alpha: float, sigma_zonal: float):
    _update_zonal_ar1(alpha, sigma_zonal)