#core/collision.py  ── BGK 碰撞算子
# 對應 PDF §4 數值方法（Eq.3）：fi(x+ci*dt, t+dt) = fi - (1/τ)(fi - f^eq_i) + Fi
#
import taichi as ti
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.lattice import feq_single, M, M_inv
# ── 全域分佈函數 fields（GPU 常駐）──
#    shape=(9, NY, NX)：第 0 維為速度方向，第 1 維為 y，第 2 維為 x
f     = ti.field(ti.f32, shape=(9, cfg.NY, cfg.NX))
f_new = ti.field(ti.f32, shape=(9, cfg.NY, cfg.NX))

# ── 宏觀量 fields（供外部模組讀取）──
rho_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
ux_field  = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
uy_field  = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))

# ── 外力貢獻（由 forcing.py 計算後寫入，碰撞時使用）──
Fx_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
Fy_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))


@ti.func
def meq(rho: float, ux: float, uy: float):   # 无返回类型注解
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

@ti.kernel
def mrt_collision_kernel(
    omega_shear: float,      # 用于 s7 = s8 = omega_shear
    s1: float, s2: float, s4: float, s6: float):
    """
    MRT 碰撞 + Pull Streaming + 边界反射（y方向自由滑移）
    同时计算宏观量并存储到 rho_field, ux_field, uy_field
    外力从 Fx_field, Fy_field 读取（由 forcing 模块预填）
    """
    for y, x in rho_field:
        # ========== 1. Pull streaming + 自由滑移边界反射 ==========
        f_local = ti.Vector([0.0] * 9)
        for i in ti.static(range(9)):
            px = (x - cfg.CX[i] + cfg.NX) % cfg.NX
            py = y - cfg.CY[i]

            if py < 0:   # 下边界自由滑移（鏡面反射：只翻轉 y 分量）
                if i == 2:          # (0,+1) → (0,-1) = i=4
                    f_local[i] = f[4, y, x]
                elif i == 5:        # (+1,+1) → (+1,-1) = i=8  ← 原本錯寫成 i=7
                    f_local[i] = f[8, y, x]
                elif i == 6:        # (-1,+1) → (-1,-1) = i=7  ← 原本錯寫成 i=8
                    f_local[i] = f[7, y, x]
                else:
                    f_local[i] = f[i, y, x]
            elif py >= cfg.NY:  # 上边界自由滑移（鏡面反射：只翻轉 y 分量）
                if i == 4:          # (0,-1) → (0,+1) = i=2
                    f_local[i] = f[2, y, x]
                elif i == 7:        # (-1,-1) → (-1,+1) = i=6  ← 原本錯寫成 i=5
                    f_local[i] = f[6, y, x]
                elif i == 8:        # (+1,-1) → (+1,+1) = i=5  ← 原本錯寫成 i=6
                    f_local[i] = f[5, y, x]
                else:
                    f_local[i] = f[i, y, x]
            else:
                # 正常内部点
                f_local[i] = f[i, py, px]

        # ========== 2. 计算宏观密度与速度 ==========
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

        # ========== 3. 读取外力（由 forcing 模块预计算） ==========
        Fx = Fx_field[y, x]
        Fy = Fy_field[y, x]

        # ========== 4. MRT 碰撞 ==========
        # ===== 4.1 计算矩 m = M * f_local =====
        m = ti.Vector.zero(ti.f32, 9)
        for i in ti.static(range(9)):       # i 是矩索引 (0..8)
            total = 0.0
            for j in ti.static(range(9)):   # j 是分布函数索引 (0..8)
                total += M[i, j] * f_local[j]
            m[i] = total

        # 4.2 平衡态矩 meq
        meq_vec = meq(rho, vx, vy)

        # 4.3 外力矩 Fm
        Fm_vec = forcing_moments(vx, vy, Fx, Fy)

        # 4.4 松弛矩阵 S（对角元）
        S = ti.Vector([
            1.0, s1, s2,        # m0, m1, m2
            1.0, s4,            # m3, m4
            1.0, s6,            # m5, m6
            omega_shear, omega_shear  # m7, m8
        ])

        # 4.5 矩松弛: m* = m - S*(m - meq) + (I - S/2)*Fm
        m_star = ti.Vector.zero(ti.f32, 9)
        for i in ti.static(range(9)):
            m_star[i] = m[i] - S[i]*(m[i] - meq_vec[i]) + (1.0 - 0.5*S[i]) * Fm_vec[i]

        # 4.6 逆变换
        for i in ti.static(range(9)):
            val = 0.0
            for j in ti.static(range(9)):
                val += M_inv[i, j] * m_star[j]
            f_new[i, y, x] = ti.max(val, 1e-12)
        
        # ========== 5. 存储宏观量（供 forcing 和诊断使用） ==========
        rho_field[y, x] = rho
        # Guo 格式修正速度：u = (j/rho) + F/(2rho)
        ux_field[y, x] = vx + 0.5 * Fx / rho
        uy_field[y, x] = vy + 0.5 * Fy / rho

@ti.kernel
def swap_fields():
    for i, y, x in f:
        f[i, y, x], f_new[i, y, x] = f_new[i, y, x], f[i, y, x]
    """裁剪分布函数中的极小负值（LBM 要求 f > 0）"""
    for i, y, x in ti.ndrange(9, cfg.NY, cfg.NX):
        f[i, y, x] = ti.max(f[i, y, x], 1e-8)
        f_new[i, y, x] = ti.max(f_new[i, y, x], 1e-8)

@ti.kernel
def init_fields_kernel(U0: float):
    """在 GPU 上初始化速度场（大尺度涡旋）和平衡分布函数"""
    kx = 2.0 * ti.math.pi / cfg.NX
    ky = 2.0 * ti.math.pi / cfg.NY
    for y, x in ti.ndrange(cfg.NY, cfg.NX):
        fx = ti.cast(x, ti.f32)
        fy = ti.cast(y, ti.f32)
        ux_macro = U0 * ti.sin(2.0 * kx * fx) * ti.cos(2.0 * ky * fy)
        uy_macro = -U0 * ti.cos(2.0 * kx * fx) * ti.sin(2.0 * ky * fy)
        rho = 1.0

        for i in ti.static(range(9)):
            feq = feq_single(i, rho, ux_macro, uy_macro)
            f[i, y, x] = feq
            f_new[i, y, x] = feq

        # 外力场初始化为零
        Fx_field[y, x] = 0.0
        Fy_field[y, x] = 0.0

@ti.kernel
def clamp_fields_kernel():
    """裁剪极小负值（安全）"""
    for i, y, x in ti.ndrange(9, cfg.NY, cfg.NX):
        f[i, y, x] = ti.max(f[i, y, x], 1e-8)
        f_new[i, y, x] = ti.max(f_new[i, y, x], 1e-8)

def init_fields():
    """初始化入口：设置宏观涡旋 + 裁剪"""
    U0 = cfg.U_MAX * 0.5
    init_fields_kernel(U0)
    # add_noise_kernel(U0 * 0.01)   # 可选噪声
    clamp_fields_kernel()
    # 确保外力场清零（虽然 kernel 里已经做了，但再显式一下）
    Fx_field.fill(0.0)
    Fy_field.fill(0.0)