# utils/fft.py  ── FFT 工具函式
# 對應 PDF §4 成員 B（加速 FFT 運算）
#
import numpy as np


def rfft2_magnitude(arr: np.ndarray, window: bool = True) -> np.ndarray:
    """
    計算 2D 實數 FFT 的幅度譜（正規化）。
    window=True 使用 Hann 窗抑制頻譜洩漏。
    """
    if window:
        ny, nx = arr.shape
        win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
        arr = arr * win
    F   = np.fft.rfft2(arr)
    amp = np.abs(F) / (arr.shape[0] * arr.shape[1])
    return amp


def wavenumber_grid(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    回傳 (KX, KY) 波數網格（格點單位）。
    KX.shape == KY.shape == (ny, nx//2+1)（對應 rfft2 輸出）
    """
    kx = np.fft.rfftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    return KX, KY


def shell_average(E2D: np.ndarray, k_max: int = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    對 2D 能量譜做環形平均（等向性假設），得到 1D E(k)。
    E2D: (ny, nx) 能量密度。
    """
    ny, nx = E2D.shape
    k_max  = k_max or (min(ny, nx) // 2)

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_bins = np.arange(0, k_max + 1, dtype=float)
    E_bins = np.zeros(len(k_bins))
    for idx in range(len(k_bins)):
        mask = (K >= idx - 0.5) & (K < idx + 0.5)
        if mask.any():
            E_bins[idx] = E2D[mask].sum()

    return k_bins, E_bins
