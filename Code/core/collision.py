# core/collision.py  ── BGK 碰撞算子
# 對應 PDF §4 數值方法（Eq.3）：fi(x+ci*dt, t+dt) = fi - (1/τ)(fi - f^eq_i) + Fi
#
import taichi as ti
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
    
    Guo's forcing:
      f_i^new = f_i*(1-ω) + f^eq_i*ω + (1 - ω/2) * F_i * dt
    其中 F_i = w_i * [3(c_i - u) + 9(c_i·u)c_i] · F_body
    """
    for y, x in rho_field:
        # ── Pull-scheme 串流 ──
        f_local = ti.Vector([0.0] * 9)
        for i in ti.static(range(9)):
            px = (x - cfg.CX[i] + cfg.NX) % cfg.NX   # 週期邊界（x 方向）
            py = (y - cfg.CY[i] + cfg.NY) % cfg.NY   # 週期邊界（y 方向）
            f_local[i] = f[i, py, px]

        # ── 宏觀量 ──
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

        # 外力場（由 forcing.py 事先算好）
        fx = Fx_field[y, x]
        fy = Fy_field[y, x]

        # ── BGK + Guo forcing ──
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

        # 存宏觀量（避免重複計算）
        rho_field[y, x] = rho
        ux_field[y, x]  = vx + 0.5 * fx / rho   # 修正速度（Guo）
        uy_field[y, x]  = vy + 0.5 * fy / rho


@ti.kernel
def swap_fields():
    """交換 f 與 f_new"""
    for i, y, x in f:
        f[i, y, x], f_new[i, y, x] = f_new[i, y, x], f[i, y, x]


@ti.kernel
def apply_periodic_bc_y():
    """
    y 方向週期邊界條件（若不用週期，可改為固壁或 open BC）
    bgk_collision_kernel 中的 py = (y-cy+NY)%NY 已隱含週期，
    此 kernel 保留供顯式修正使用。
    """
    for x in range(cfg.NX):
        for i in ti.static(range(9)):
            f[i, 0,        x] = f[i, cfg.NY-2, x]
            f[i, cfg.NY-1, x] = f[i, 1,        x]


def init_fields():
    """初始化 f / f_new 為均勻靜止流場 + 小噪音"""
    rng = np.random.default_rng(42)
    f_np = np.zeros((9, cfg.NY, cfg.NX), dtype=np.float32)
    for i in range(9):
        f_np[i] = cfg.W_NP[i]
    # 加入小噪音以觸發不穩定性（對應 PDF §7 噪音強度 ε）
    noise = rng.uniform(-cfg.NOISE_AMP, cfg.NOISE_AMP,
                        size=(9, cfg.NY, cfg.NX)).astype(np.float32)
    f_np += noise * cfg.W_NP[:, None, None]
    f_np = np.clip(f_np, 1e-6, None)
    f.from_numpy(f_np)
    f_new.from_numpy(f_np)
    # 外力場初始為零
    Fx_field.fill(0.0)
    Fy_field.fill(0.0)
