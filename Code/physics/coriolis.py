# physics/coriolis.py  ── β-plane Coriolis 物理模組
# 對應 PDF §3 控制方程（Eq.2: f(y) = f0 + βy）
#
# 本模組提供 Coriolis 參數的計算與轉換函式，
# 實際施力在 core/forcing.py 的 update_forcing_kernel 中完成。
#
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg


def coriolis_profile_numpy() -> np.ndarray:
    """
    返回 (NY,) 陣列，每個 y 格點的 Coriolis 參數 f(y)。
    赤道在 y = NY//2，往極地（y=0 或 y=NY-1）f 增大。
    
    f(y) = f0 + β * (y - NY/2)
    
    注意：2D LBM 中 Coriolis 以外力形式進入（不是真實 3D 旋轉），
    此函式用於分析與視覺化。
    """
    y_idx   = np.arange(cfg.NY, dtype=np.float32)
    f_y     = cfg.F0 + cfg.BETA * (y_idx - cfg.NY // 2)
    return f_y


def rossby_radius(nu: float = None, beta: float = None) -> float:
    """
    Rossby 形變半徑估算（格單位）：
      L_R = sqrt(U_MAX / β)
    用於判斷帶狀流的特徵波長。
    """
    if beta is None:
        beta = cfg.BETA
    if nu is None:
        nu = cfg.NU
    return float(np.sqrt(cfg.U_MAX / max(beta, 1e-12)))


def rhines_scale(u_rms: float, beta: float = None) -> float:
    """
    Rhines 尺度：帶狀流與等向性湍流的分界波數。
      L_Rh = π * sqrt(2 * U_rms / β)
    u_rms: 當前均方根速度（格單位）
    """
    if beta is None:
        beta = cfg.BETA
    return float(np.pi * np.sqrt(2.0 * u_rms / max(beta, 1e-12)))
