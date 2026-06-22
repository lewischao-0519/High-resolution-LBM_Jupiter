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
    omega_shear: ti.f32, 
    s1: ti.f32, 
    s2: ti.f32, 
    s4: ti.f32, 
    s6: ti.f32, 
    f0: ti.f32, 
    beta: ti.f32, 
    epsilon: ti.f32
):
    ny_half = cfg.NY // 2
    for y, x in rho_field:
        # 1. Pull streaming + half-way bounce-back (南北牆)
        f_local = ti.Vector([0.0]*9)
        for i in ti.static(range(9)):
            px = (x - cfg.CX[i] + cfg.NX) % cfg.NX
            py = y - cfg.CY[i]
            if py < 0 or py >= cfg.NY:
                f_local[i] = f[cfg.OPP[i], y, x]   # 在當前格反射
            else:
                f_local[i] = f[i, py, px]

        # 2. 巨觀量
        rho = 0.0; vx = 0.0; vy = 0.0
        for i in ti.static(range(9)):
            rho += f_local[i]
            vx  += f_local[i]*float(cfg.CX[i])
            vy  += f_local[i]*float(cfg.CY[i])
        rho = ti.max(rho, 1e-6); vx /= rho; vy /= rho

        # 3. 外力：科氏力(cross product) + 阻尼 + 噪音，皆為 rho*加速度
        dy    = float(y - ny_half)
        f_cor = f0 + beta*dy
        Fx_cor =  rho * f_cor * vy
        Fy_cor = -rho * f_cor * vx
        Fx_damp = -epsilon * rho * vx
        Fy_damp = -epsilon * rho * vy

        noise_val = noise_zonal[y]
        noise_max = 0.005
        if ti.abs(noise_val) > noise_max:
            noise_val = noise_max * ti.math.sign(noise_val)
        Fx = Fx_cor + Fx_damp + 0.5*noise_val
        Fy = Fy_cor + Fy_damp

        # 牆面無穿透：垂直外力歸零（在施力前）
        if y == 0 or y == cfg.NY-1:
            Fy = 0.0

        # 4. Guo 半步速度
        vx_half = vx + 0.5*Fx/rho
        vy_half = vy + 0.5*Fy/rho

        # 5. MRT
        m = ti.Vector.zero(ti.f32, 9)
        for i in ti.static(range(9)):
            s = 0.0
            for j in ti.static(range(9)): s += M[i,j]*f_local[j]
            m[i] = s
        meq_vec = meq(rho, vx_half, vy_half)
        Fm_vec  = forcing_moments(vx_half, vy_half, Fx, Fy)
        S = ti.Vector([1.0, s1, s2, 1.0, s4, s4, s6, omega_shear, omega_shear])
        m_star = ti.Vector.zero(ti.f32, 9)
        for i in ti.static(range(9)):
            m_star[i] = m[i] - S[i]*(m[i]-meq_vec[i]) + (1.0-0.5*S[i])*Fm_vec[i]
        for i in ti.static(range(9)):
            v = 0.0
            for j in ti.static(range(9)): v += M_inv[i,j]*m_star[j]
            f_new[i, y, x] = ti.max(v, 1e-8)

        rho_field[y, x] = rho
        ux_field[y, x] = vx_half
        uy_field[y, x] = vy_half if (y != 0 and y != cfg.NY-1) else 0.0
        
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