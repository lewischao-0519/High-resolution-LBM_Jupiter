# analysis/diagnostics.py  ── 延伸診斷量（對比木星觀測數據）
#
# 新增量：
#   1. Rhines 尺度  L_β = sqrt(U_rms / β)
#   2. 噴流位置時序（y index of each jet peak）
#   3. 帶狀動能 KE_zonal vs 渦流動能 KE_eddy
#   4. Reynolds stress  <u'v'>_x  (y profile + domain mean)
#   5. 渦度 RMS 與偏態 (skewness)  — 木星應為負偏（反氣旋主導）
#   6. β-Rossby 數  Ro_β = U_rms / (β * L_β²)
#
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.collision import ux_field, uy_field


# ──────────────────────────────────────────────
# 1. Rhines 尺度
# ──────────────────────────────────────────────
def rhines_scale(u_rms: float) -> float:
    """
    L_β = sqrt(U_rms / β)  (格點單位)
    理論上帶狀流間距 ≈ 2π L_β。
    對應木星觀測：赤道附近帶寬 ~10° ≈ 12000 km。
    """
    if cfg.BETA <= 0 or u_rms <= 0:
        return 0.0
    return float(np.sqrt(u_rms / cfg.BETA))


# ──────────────────────────────────────────────
# 2. β-Rossby 數
# ──────────────────────────────────────────────
def rossby_beta(u_rms: float) -> float:
    """
    Ro_β = U_rms / (β * L_β²) = U_rms / (β * (U_rms/β)) = 1  (理論上 ~1 時噴流形成)
    實用版本改用格點寬度 L = NY/2 作為特徵長度：
    Ro_β = U_rms / (β * (NY/2)²)
    """
    L = cfg.NY / 2.0
    denom = cfg.BETA * L * L
    if denom <= 0:
        return 0.0
    return float(u_rms / denom)


# ──────────────────────────────────────────────
# 3. 噴流位置（y index 列表）
# ──────────────────────────────────────────────
def jet_positions(u_bar: np.ndarray, threshold: float = 0.8) -> list[int]:
    """
    回傳所有噴流的 y-index 列表（東風峰與西風谷皆計入）。
    可隨時間追蹤噴流合併 / 分裂事件。
    """
    u_std = np.std(u_bar)
    if u_std < 1e-9:
        return []
    u_norm = u_bar / u_std
    positions = []
    for i in range(1, len(u_norm) - 1):
        if abs(u_norm[i]) > threshold:
            if (u_norm[i] > u_norm[i-1] and u_norm[i] > u_norm[i+1]) or \
               (u_norm[i] < u_norm[i-1] and u_norm[i] < u_norm[i+1]):
                positions.append(int(i))
    return positions


# ──────────────────────────────────────────────
# 4. 帶狀動能 vs 渦流動能
# ──────────────────────────────────────────────
def kinetic_energy_decomp(ux_np: np.ndarray = None,
                           uy_np: np.ndarray = None) -> dict:
    """
    分解總動能為帶狀分量（mean flow）與渦流分量（eddy）。
    
    KE_total = 0.5 * <u² + v²>
    KE_zonal = 0.5 * <ū²>               (ū = <u>_x)
    KE_eddy  = KE_total - KE_zonal

    逆能量串聯有效時 KE_zonal / KE_total → 大 (木星觀測 ~60-80%)
    
    Returns dict:
      KE_total, KE_zonal, KE_eddy, zonal_fraction
    """
    if ux_np is None:
        ux_np = ux_field.to_numpy()
    if uy_np is None:
        uy_np = uy_field.to_numpy()

    KE_total = float(0.5 * np.mean(ux_np**2 + uy_np**2))

    u_bar = ux_np.mean(axis=1)   # (NY,)
    v_bar = uy_np.mean(axis=1)
    KE_zonal = float(0.5 * np.mean(u_bar**2 + v_bar**2))

    KE_eddy = KE_total - KE_zonal
    zonal_frac = KE_zonal / KE_total if KE_total > 0 else 0.0

    return {
        'KE_total'      : KE_total,
        'KE_zonal'      : KE_zonal,
        'KE_eddy'       : KE_eddy,
        'zonal_fraction': zonal_frac,
    }


# ──────────────────────────────────────────────
# 5. Reynolds stress  <u'v'>_x
# ──────────────────────────────────────────────
def reynolds_stress(ux_np: np.ndarray = None,
                    uy_np: np.ndarray = None) -> dict:
    """
    Reynolds stress (渦流動量通量) <u'v'>_x。
    
    u' = u - ū,  v' = v - v̄
    RS(y) = <u'v'>_x  → (NY,) profile
    RS_mean = domain-averaged |RS|
    
    正值 RS 梯度 → 動量向極輸送（形成西風噴流）
    對應 Andrews & McIntyre EP-flux 理論。
    """
    if ux_np is None:
        ux_np = ux_field.to_numpy()
    if uy_np is None:
        uy_np = uy_field.to_numpy()

    u_bar = ux_np.mean(axis=1, keepdims=True)
    v_bar = uy_np.mean(axis=1, keepdims=True)
    u_prime = ux_np - u_bar
    v_prime = uy_np - v_bar
    RS_profile = (u_prime * v_prime).mean(axis=1)   # (NY,)

    return {
        'RS_profile': RS_profile,
        'RS_mean'   : float(np.mean(np.abs(RS_profile))),
    }


# ──────────────────────────────────────────────
# 6. 渦度統計：RMS + 偏態
# ──────────────────────────────────────────────
def vorticity_stats(omega_np: np.ndarray) -> dict:
    """
    渦度場的統計量。
    
    omega_rms   : sqrt(<ω²>)  → 湍流強度
    omega_skew  : 偏態 = <ω³> / <ω²>^{3/2}
                  木星觀測：負偏（反氣旋大渦較多且強）
                  純 2D 湍流：接近 0
                  
    Returns dict: omega_rms, omega_skew
    """
    flat = omega_np.flatten()
    rms  = float(np.sqrt(np.mean(flat**2)))
    std  = float(np.std(flat))
    if std < 1e-12:
        skew = 0.0
    else:
        skew = float(np.mean((flat - flat.mean())**3) / std**3)
    return {
        'omega_rms' : rms,
        'omega_skew': skew,
    }
