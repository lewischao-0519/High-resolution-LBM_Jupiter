# core/collision.py  ── BGK 碰撞算子
# 對應 PDF §4 數值方法（Eq.3）：fi(x+ci*dt, t+dt) = fi - (1/τ)(fi - f^eq_i) + Fi
#
import taichi as ti
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.lattice import feq_single

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


@ti.kernel
def bgk_collision_kernel(omega: float):
    """
    Pull-scheme 串流 + BGK 碰撞 + 外力修正（Guo's forcing scheme）
    """
    for y, x in rho_field:
        # ── Pull-scheme 串流 ──
        f_local = ti.Vector([0.0] * 9)
        for i in ti.static(range(9)):           # ← 這個迴圈不能刪！
            px = (x - cfg.CX[i] + cfg.NX) % cfg.NX   # x 方向週期
            py = y - cfg.CY[i]                       # y 方向不做模運算
            
            # 檢查 y 是否越界
            if 0 <= py < cfg.NY:
                f_local[i] = f[i, py, px]
            else:
                f_local[i] = 0.0  # 越界暫時設 0，等邊界 kernel 處理

        # ── 計算宏觀量 ──
        rho = 0.0
        vx  = 0.0
        vy  = 0.0
        for i in ti.static(range(9)):
            rho += f_local[i]
            vx  += f_local[i] * float(cfg.CX[i])
            vy  += f_local[i] * float(cfg.CY[i])
        rho = ti.max(rho, 1e-6)
        vx /= rho
        vy /= rho

        # ── 讀取外力場 ──
        fx = Fx_field[y, x]
        fy = Fy_field[y, x]

        # ── BGK 碰撞 + Guo forcing ──
        u2 = vx*vx + vy*vy
        for i in ti.static(range(9)):
            ci_x = float(cfg.CX[i])
            ci_y = float(cfg.CY[i])
            cu   = ci_x*vx + ci_y*vy
            feq  = rho * cfg.W[i] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2)

            # Guo's forcing term
            guo = cfg.W[i] * (
                3.0 * ((ci_x - vx)*fx + (ci_y - vy)*fy)
                + 9.0 * cu * (ci_x*fx + ci_y*fy)
            )

            f_new[i, y, x] = (
                f_local[i] * (1.0 - omega)
                + feq * omega
                + (1.0 - 0.5*omega) * guo
            )

        # ── 只儲存密度（速度等邊界處理完再算）──
        rho_field[y, x] = rho

@ti.kernel
def apply_boundary_y_free_slip():
    """
    y 方向自由滑移邊界（南北極牆面）
    使用 half-way bounce-back 規則，保持切向動量，反轉法向動量
    """
    for x in range(cfg.NX):
        # ── 下邊界 y = 0 ──
        # 方向 2 (往上，cy=1) 從邊界反彈到方向 4 (往下，cy=-1)
        f_new[4, 0, x] = f_new[2, 0, x]
        # 方向 5 (右上，cx=1,cy=1) → 方向 7 (左下，cx=-1,cy=-1)
        f_new[7, 0, x] = f_new[5, 0, x]
        # 方向 6 (左上，cx=-1,cy=1) → 方向 8 (右下，cx=1,cy=-1)
        f_new[8, 0, x] = f_new[6, 0, x]
        
        # ── 上邊界 y = NY-1 ──
        # 方向 4 (往下，cy=-1) → 方向 2 (往上，cy=1)
        f_new[2, cfg.NY-1, x] = f_new[4, cfg.NY-1, x]
        # 方向 7 (左下，cx=-1,cy=-1) → 方向 5 (右上，cx=1,cy=1)
        f_new[5, cfg.NY-1, x] = f_new[7, cfg.NY-1, x]
        # 方向 8 (右下，cx=1,cy=-1) → 方向 6 (左上，cx=-1,cy=1)
        f_new[6, cfg.NY-1, x] = f_new[8, cfg.NY-1, x]

@ti.kernel
def swap_fields():
    """交換 f 與 f_new"""
    for i, y, x in f:
        f[i, y, x], f_new[i, y, x] = f_new[i, y, x], f[i, y, x]

@ti.kernel
def compute_macro():
    for y, x in rho_field:
        rho = 0.0
        vx  = 0.0
        vy  = 0.0
        for i in ti.static(range(9)):
            rho += f_new[i, y, x]
            vx  += f_new[i, y, x] * float(cfg.CX[i])
            vy  += f_new[i, y, x] * float(cfg.CY[i])
        rho = ti.max(rho, 1e-6)
        vx /= rho
        vy /= rho
        rho_field[y, x] = rho
        ux_field[y, x] = vx + 0.5 * Fx_field[y, x] / rho
        uy_field[y, x] = vy + 0.5 * Fy_field[y, x] / rho

def init_fields():
    """使用 GPU kernel 初始化分布函数（大尺度涡旋 + 噪声）"""
    U0 = cfg.U_MAX * 0.5
    init_fields_kernel(U0)          # 设置宏观涡旋
    #add_noise_kernel(U0 * 0.01)     # 加入小噪声
    clamp_fields_kernel()           # 裁剪负值（安全）
    Fx_field.fill(0.0)              # 外力场清零
    Fy_field.fill(0.0)