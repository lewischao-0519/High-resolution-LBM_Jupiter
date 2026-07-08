"""
physical_units.py
─────────────────
把 LBM 的「格點 / 步」單位換算成木星實際物理量。

換算採兩個獨立錨點（皆用 ref 方法，可用環境變數覆寫，方便日後調整；
本模組刻意不 import config.py，避免觸發 ti.init(GPU)，可離線分析）：

  1) 長度 dx：由木星赤道幾何決定
        x 為週期方向（緯向），整個週期域對應木星赤道整圈 × JUP_LON_SPAN，
        均分到 NX 格：
            dx = (2π · R_eq · JUP_LON_SPAN) / NX      [m / 格]
        （lattice 各向同性，dy = dx。）

  2) 時間 dt：由自轉角速度 Ω 決定（透過 β 匹配）
        模擬的 β 是每步、每格的旋轉率梯度 [rad·step⁻¹·grid⁻¹]，
        木星赤道的 β_phys = 2Ω / R_eq [rad·s⁻¹·m⁻¹]。要兩者一致：
            β_sim = β_phys · dt · dx
          → dt = β_sim / (β_phys · dx)                [s / 步]
        代入後 R_eq 相消，dt 只由 β_sim、NX、Ω、JUP_LON_SPAN 決定。

⚠️ 重要：dx、dt 一旦由上面兩個錨點固定，其餘物理量（尤其速度）就被
「決定」了、無法再獨立匹配。u_phys = u_sim · dx/dt 若與木星實測噴流
（尖峰 ~100–150 m/s）差幾倍，這是有意義的診斷（代表模擬相對自轉被
過度/不足驅動），不是換算錯誤。若你想改成「以速度為錨點」，把 dt 改成
由目標速度反推即可（見 dt_from_velocity）。
"""
import os
import numpy as np

# ── 木星參考量（ref 方法，環境變數可覆寫）────────────────────────────
JUP_R_EQ_M       = float(os.environ.get('LBM_JUP_R_EQ_M', 7.1492e7))        # 赤道半徑 (m)
JUP_ROT_PERIOD_S = float(os.environ.get('LBM_JUP_ROT_PERIOD_S', 9.9250 * 3600.0))  # System III 自轉週期 (s)
JUP_LON_SPAN     = float(os.environ.get('LBM_JUP_LON_SPAN', 1.0))           # x 週期域佔赤道整圈的比例（1.0＝整圈 360°）

JUP_OMEGA   = 2.0 * np.pi / JUP_ROT_PERIOD_S     # 自轉角速度 Ω (rad/s)
JUP_BETA_EQ = 2.0 * JUP_OMEGA / JUP_R_EQ_M       # 赤道 β = 2Ω/R (rad·s⁻¹·m⁻¹)

# ── 網格常數（對齊 config.py；此處不 import config 以免 ti.init(GPU)）──
NX_DEFAULT = 1024
NY_DEFAULT = 512
NY_REF     = 256          # config.py：BETA = BETA_REF · NY_REF / NY


def grid_dx(nx: int = NX_DEFAULT) -> float:
    """空間解析度 dx [m/格]：x 週期域 = 木星赤道整圈 × JUP_LON_SPAN。"""
    return JUP_R_EQ_M * 2.0 * np.pi * JUP_LON_SPAN / nx


def beta_sim(beta_ref: float, ny: int = NY_DEFAULT) -> float:
    """由 BETA_REF 還原實際 β_sim [rad·step⁻¹·grid⁻¹]（config.py 的換算）。"""
    return beta_ref * NY_REF / ny


def grid_dt(beta_ref: float, nx: int = NX_DEFAULT, ny: int = NY_DEFAULT) -> float:
    """時間解析度 dt [s/步]：由自轉角速度換算（β 匹配，見檔頭）。"""
    return beta_sim(beta_ref, ny) / (JUP_BETA_EQ * grid_dx(nx))


def dt_from_velocity(u_sim_ref: float, u_phys_ref_ms: float,
                     nx: int = NX_DEFAULT) -> float:
    """（替代錨點）若改用速度匹配：給定某模擬速度 u_sim_ref 對應的目標實際
    速度 u_phys_ref_ms [m/s]，反推 dt = u_sim_ref · dx / u_phys_ref。"""
    return u_sim_ref * grid_dx(nx) / u_phys_ref_ms


# ── 便捷換算 ─────────────────────────────────────────────────────────
def velocity_ms(u_sim, beta_ref, nx=NX_DEFAULT, ny=NY_DEFAULT):
    """速度：u_sim [格/步] → [m/s]。"""
    return u_sim * grid_dx(nx) / grid_dt(beta_ref, nx, ny)


def length_km(L_grid, nx=NX_DEFAULT):
    """長度：L [格] → [km]。"""
    return L_grid * grid_dx(nx) / 1000.0


def time_days(steps, beta_ref, nx=NX_DEFAULT, ny=NY_DEFAULT):
    """時間：steps [步] → [天]。"""
    return steps * grid_dt(beta_ref, nx, ny) / 86400.0


def conversion_factors(beta_ref, nx=NX_DEFAULT, ny=NY_DEFAULT) -> dict:
    """回傳該組參數的換算因子與參考量，供寫入 CSV 記錄。"""
    dx = grid_dx(nx)
    dt = grid_dt(beta_ref, nx, ny)
    return {
        'dx_km'          : dx / 1000.0,
        'dt_s'           : dt,
        'beta_sim'       : beta_sim(beta_ref, ny),
        'channel_width_km': ny * dx / 1000.0,   # 徑向通道寬（NY × dx）
        'jup_R_eq_m'     : JUP_R_EQ_M,
        'jup_rot_period_h': JUP_ROT_PERIOD_S / 3600.0,
        'jup_lon_span'   : JUP_LON_SPAN,
    }


if __name__ == '__main__':
    # 快速自檢：用 config.py 預設 BETA_REF=7e-5 印出換算因子
    beta_ref = 7e-5
    f = conversion_factors(beta_ref)
    print("木星換算因子（BETA_REF=7e-5, NX=1024, NY=512）：")
    for k, v in f.items():
        print(f"  {k:18s} = {v:.6g}")
    print(f"  u_sim=0.03 → {velocity_ms(0.03, beta_ref):.1f} m/s")
    print(f"  3e6 步     → {time_days(3e6, beta_ref):.1f} 天")
