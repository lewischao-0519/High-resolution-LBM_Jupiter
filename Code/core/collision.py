#外掛程式
import taichi as ti
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.lattice import feq_single, M, M_inv

#全域分佈函數
f     = ti.field(ti.f32, shape=(9, cfg.NY, cfg.NX))
f_new = ti.field(ti.f32, shape=(9, cfg.NY, cfg.NX))

#外力貢獻
Fx_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
Fy_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))

#AR噪音場（1D，每個緯度行一個值，對所有 x 相同 → 只注入 k_x=0）
noise_zonal = ti.field(ti.f32, shape=cfg.NY)

#宏觀量
rho_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
ux_field  = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
uy_field  = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))

#MRT碰撞核函数（於mrt_collision_kernel中調用）
@ti.func
def meq(rho: float, ux: float, uy: float):   
    jx = rho * ux
    jy = rho * uy
    j2 = jx*jx + jy*jy
    m0 = rho
    m1 = -2.0*rho + 3.0 * j2 / rho
    m2 = rho - 3.0 * j2 / rho
    m3 = jx
    m4 = -jx
    m5 = jy
    m6 = -jy
    m7 = (jx*jx - jy*jy) / rho
    m8 = (jx*jy) / rho
    return ti.Vector([m0, m1, m2, m3, m4, m5, m6, m7, m8])

#MRT碰撞核函数(於mrt_collision_kernel中調用）)
@ti.func
def forcing_moments(ux: float, uy: float, Fx: float, Fy: float):
    uF = ux*Fx + uy*Fy
    return ti.Vector([
        0.0,
        6.0 * uF,
        -6.0 * uF,
        Fx,
        -Fx,
        Fy,
        -Fy,
        2.0 * (ux*Fx - uy*Fy),
        ux*Fy + uy*Fx
    ])

#MRT碰撞模型(於main中調用）
@ti.kernel
def mrt_collision_kernel(
    omega_shear: float,
    s1: float, s2: float, s4: float, s6: float,
    f0: float, beta: float, epsilon: float
):
    ny_half = cfg.NY // 2
    for y, x in rho_field:
        #Pull streaming
        f_local = ti.Vector([0.0] * 9)
        for i in ti.static(range(9)):
            px = (x - cfg.CX[i] + cfg.NX) % cfg.NX
            py = y - cfg.CY[i]

            if py < 0:   # 下邊界自由滑移（鏡面反射）
                if i == 2:          f_local[i] = f[4, y, x]
                elif i == 5:        f_local[i] = f[8, y, x]
                elif i == 6:        f_local[i] = f[7, y, x]
                else:               f_local[i] = f[i, y, x]
            elif py >= cfg.NY:  # 上邊界自由滑移
                if i == 4:          f_local[i] = f[2, y, x]
                elif i == 7:        f_local[i] = f[6, y, x]
                elif i == 8:        f_local[i] = f[5, y, x]
                else:               f_local[i] = f[i, y, x]
            else:
                f_local[i] = f[i, py, px]

        #計算宏觀參數
        rho = 0.0
        vx = 0.0
        vy = 0.0
        for i in ti.static(range(9)):
            rho += f_local[i]
            vx  += f_local[i] * float(cfg.CX[i])
            vy  += f_local[i] * float(cfg.CY[i])
        rho = ti.max(rho, 1e-6)
        vx /= rho
        vy /= rho

        #半隱式科氏力旋轉
        dy    = float(y - ny_half)
        f_cor = f0 + beta * dy
        a     = f_cor * 0.5          # a = f*dt/2, dt=1
        det   = 1.0 + a * a
        
        #旋轉後的速度（解析精確，無穩定性問題）
        vx_rot = ((1.0 - a*a) * vx + 2.0 * a * vy) / det
        vy_rot = ((1.0 - a*a) * vy - 2.0 * a * vx) / det

         # 用 δf_eq 將旋轉注入 f_local（這是原本穩定的作法）
        for i in ti.static(range(9)):
            delta_feq = feq_single(i, rho, vx_rot, vy_rot) \
                      - feq_single(i, rho, vx, vy)
            f_local[i] += delta_feq
            
        #計算外力（阻尼 + 噪音）
        # 阻尼基於旋轉後的速度
        Fx_damp = -epsilon * vx
        Fy_damp = -epsilon * vy

        #噪音（限幅）
        noise_val = noise_zonal[y]
        noise_max = 0.05
        Fx_noise = 0.5 * noise_val
        Fy_noise = 0.0

        Fx = Fx_damp + Fx_noise
        Fy = Fy_damp + Fy_noise

        #MRT碰撞（使用 Guo 格式）
        #Guo 格式
        vx_half = vx_rot + 0.5 * Fx / rho
        vy_half = vy_rot + 0.5 * Fy / rho

        #計算 m = M * f_local
        m = ti.Vector.zero(ti.f32, 9)
        for i in ti.static(range(9)):
            total = 0.0
            for j in ti.static(range(9)):
                total += M[i, j] * f_local[j]
            m[i] = total

        # 用 v_half 計算平衡態和力矩
        meq_vec = meq(rho, vx_half, vy_half)
        Fm_vec  = forcing_moments(vx_half, vy_half, Fx, Fy)

        S = ti.Vector([
            1.0, s1, s2,
            1.0, s4,
            1.0, s6,
            omega_shear, omega_shear
        ])

        m_star = ti.Vector.zero(ti.f32, 9)
        for i in ti.static(range(9)):
            m_star[i] = m[i] - S[i]*(m[i] - meq_vec[i]) + (1.0 - 0.5*S[i]) * Fm_vec[i]

        # 反變換
        for i in ti.static(range(9)):
            val = 0.0
            for j in ti.static(range(9)):
                val += M_inv[i, j] * m_star[j]
            f_new[i, y, x] = val   

        # Guo 格式：v_half 再補半次 = v + F/rho（完整外力效應）
        rho_field[y, x] = rho
        ux_field[y, x] = vx + Fx / rho      # vx + 0.5*Fx/rho + 0.5*Fx/rho
        uy_field[y, x] = vy + Fy / rho      # vy + 0.5*Fy/rho + 0.5*Fy/rho

#裁減負值（安全措施）（於main中調用）
@ti.kernel
def swap_fields():
    for i, y, x in f:
        f[i, y, x], f_new[i, y, x] = f_new[i, y, x], f[i, y, x]

#初始化平衡分佈（於main中調用）
@ti.kernel
def init_fields_kernel(U0: float):
    kx = 2.0 * ti.math.pi / cfg.NX
    ky = 2.0 * ti.math.pi / cfg.NY
    for y, x in ti.ndrange(cfg.NY, cfg.NX):
        fx = ti.cast(x, ti.f32)
        fy = ti.cast(y, ti.f32)
        ux_macro = U0 * ti.sin(2.0 * kx * fx) * ti.cos(2.0 * ky * fy)
        uy_macro = -U0 * ti.cos(2.0 * kx * fx) * ti.sin(2.0 * ky * fy)
        rho = 1.0

        #初始化分佈函數為平衡態
        for i in ti.static(range(9)):
            feq = feq_single(i, rho, ux_macro, uy_macro)
            f[i, y, x] = feq
            f_new[i, y, x] = feq

        #外力場初始化為0
        Fx_field[y, x] = 0.0
        Fy_field[y, x] = 0.0

@ti.kernel
def clamp_fields_kernel():
    for i, y, x in ti.ndrange(9, cfg.NY, cfg.NX):
        f[i, y, x] = ti.max(f[i, y, x], 1e-8)
        f_new[i, y, x] = ti.max(f_new[i, y, x], 1e-8)

#field初始化（於main中調用）
def init_fields():
    U0 = cfg.U_MAX * 0.5
    init_fields_kernel(U0)
    clamp_fields_kernel()
    Fx_field.fill(0.0)
    Fy_field.fill(0.0)