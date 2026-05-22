# physics/nondimensional.py  ── 無因次化工具
# 對應 PDF §7 參數掃描設計（Ro, Bu, Re 等無因次數）
#
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg


def rossby_number(U: float = None, L: float = None, f0: float = None) -> float:
    """
    Rossby 數 Ro = U / (f0 * L)
    衡量慣性力 vs. Coriolis 力的比值。
    Ro << 1: Coriolis 主導（類木星帶狀流）
    """
    U  = U  or cfg.U_MAX
    L  = L  or float(cfg.NX // 4)
    f0 = f0 or cfg.F0
    return U / max(f0 * L, 1e-12)


def reynolds_number(U: float = None, L: float = None, nu: float = None) -> float:
    """Re = U * L / ν"""
    U  = U  or cfg.U_MAX
    L  = L  or float(cfg.NX // 4)
    nu = nu or cfg.NU
    return U * L / max(nu, 1e-12)


def burger_number(N: float, H: float, f0: float = None, L: float = None) -> float:
    """
    Burger 數 Bu = (N*H)^2 / (f0*L)^2
    N: 浮力頻率, H: 深度尺度（2D 中取 NY）
    """
    f0 = f0 or cfg.F0
    L  = L  or float(cfg.NX // 4)
    return (N * H) ** 2 / max((f0 * L) ** 2, 1e-12)


def rhines_wavenumber(u_rms: float, beta: float = None) -> float:
    """
    Rhines 波數 k_β = sqrt(β / (2*U_rms))
    對應帶狀流特徵尺度的反轉波數。
    """
    beta = beta or cfg.BETA
    return float(np.sqrt(beta / max(2.0 * u_rms, 1e-12)))


def print_dimensionless_summary(u_rms: float = None):
    """列印當前參數的無因次數摘要"""
    u_rms = u_rms or cfg.U_MAX
    Ro  = rossby_number()
    Re  = reynolds_number()
    k_b = rhines_wavenumber(u_rms)
    L_R = float(np.sqrt(cfg.U_MAX / max(cfg.BETA, 1e-12)))
    print("─── 無因次數摘要 ───────────────────────")
    print(f"  Rossby 數   Ro  = {Ro:.4f}")
    print(f"  Reynolds 數 Re  = {Re:.1f}")
    print(f"  Rhines 波數 k_β = {k_b:.5f}  (格^-1)")
    print(f"  Rossby 半徑 L_R = {L_R:.1f}  (格)")
    print(f"  β 參數          = {cfg.BETA:.2e}")
    print(f"  ΔT 溫差         = {cfg.DELTA_T:.3f}")
    print("─────────────────────────────────────────")
