# analysis/spectrum.py  ── FFT 能量譜分析
# 對應 PDF §6 能量譜分析（Eq.6: E(k) = 1/2(|û_x|^2 + |û_y|^2)）
#
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.collision import ux_field, uy_field


def compute_energy_spectrum(ux_np: np.ndarray = None,
                            uy_np: np.ndarray = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    計算二維速度場的等向能量譜 E(k)。
    對應 PDF §6 Eq.6：E(k) = 1/2 * (|û_x|^2 + |û_y|^2)
    
    Parameters
    ----------
    ux_np, uy_np : (NY, NX) float32 陣列。若為 None，從 GPU 讀取。
    
    Returns
    -------
    k_bins  : (M,) 波數陣列（格^-1，正規化至 [0, π]）
    E_bins  : (M,) 對應的能量密度
    """
    if ux_np is None:
        ux_np = ux_field.to_numpy()
    if uy_np is None:
        uy_np = uy_field.to_numpy()

    ny, nx = ux_np.shape

    # 2D FFT（加 Hann 窗抑制頻譜洩漏）
    win  = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    Ux   = np.fft.fft2(ux_np * win)
    Uy   = np.fft.fft2(uy_np * win)

    # 能量密度（對應 Eq.6）
    E2D  = 0.5 * (np.abs(Ux)**2 + np.abs(Uy)**2) / (nx * ny)**2

    # 波數網格（正規化至格點單位）
    kx = np.fft.fftfreq(nx) * nx   # ∈ [-nx/2, nx/2)
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    K  = np.sqrt(KX**2 + KY**2)

    # 環形平均（等向性假設）
    k_max  = min(nx, ny) // 2
    k_bins = np.arange(0.5, k_max + 0.5, 1.0)
    E_bins = np.zeros(len(k_bins))

    for idx, k_c in enumerate(k_bins):
        mask = (K >= k_c - 0.5) & (K < k_c + 0.5)
        E_bins[idx] = E2D[mask].sum()

    k_bins = k_bins * (2.0 * np.pi / min(nx, ny))   # 轉換為物理波數
    return k_bins, E_bins


def compute_zonal_mean_spectrum(ux_np: np.ndarray = None
                                ) -> tuple[np.ndarray, np.ndarray]:
    """
    計算緯向平均速度的緯向波數譜（偵測帶狀流）。
    u_bar(y) = <ux>_x → FFT 沿 y 方向。
    
    Returns
    -------
    k_y   : 緯向波數陣列
    E_zonal : 對應能量
    """
    if ux_np is None:
        ux_np = ux_field.to_numpy()

    u_bar = ux_np.mean(axis=1)                         # (NY,) 緯向平均
    U_bar = np.fft.rfft(u_bar * np.hanning(len(u_bar)))
    k_y   = np.fft.rfftfreq(len(u_bar)) * len(u_bar)
    E_zonal = 0.5 * np.abs(U_bar)**2 / len(u_bar)**2
    return k_y, E_zonal


def kolmogorov_slope(k: np.ndarray, E: np.ndarray,
                     k_lo: float = 2.0, k_hi: float = 20.0) -> float:
    """
    在對數空間擬合 E(k) ~ k^α，回傳斜率 α。
    Kolmogorov: α = -5/3（3D湍流）
    逆能量串聯（2D）：α ≈ -5/3 or -3
    """
    mask = (k >= k_lo) & (k <= k_hi) & (E > 0)
    if mask.sum() < 3:
        return np.nan
    log_k = np.log10(k[mask])
    log_E = np.log10(E[mask])
    coeffs = np.polyfit(log_k, log_E, 1)
    return float(coeffs[0])
