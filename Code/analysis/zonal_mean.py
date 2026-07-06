# analysis/zonal_mean.py  ── 帶狀平均風速分析
# 對應 PDF §9 預期結果：東西向帶狀風系
#
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.collision import ux_field, uy_field


def compute_zonal_mean() -> dict:
    """
    計算帶狀（緯向）平均速度剖面。
    
    Returns
    -------
    dict 包含：
      'y'      : y 格點索引 (NY,)
      'u_bar'  : <ux>_x (NY,)  ── 緯向平均東西風
      'v_bar'  : <uy>_x (NY,)  ── 緯向平均南北風
      'u_std'  : 緯向速度標準差
      'KE_bar' : 緯向平均動能 = 0.5*(u_bar^2 + v_bar^2)
    """
    ux_np = ux_field.to_numpy()
    uy_np = uy_field.to_numpy()

    u_bar = ux_np.mean(axis=1)           # 對 x 平均 → (NY,)
    v_bar = uy_np.mean(axis=1)
    u_std = ux_np.std(axis=1)

    return {
        'y'     : np.arange(cfg.NY, dtype=np.float32),
        'u_bar' : u_bar,
        'v_bar' : v_bar,
        'u_std' : u_std,
        'KE_bar': 0.5 * (u_bar**2 + v_bar**2),
    }


def smooth_profile(u_bar: np.ndarray, window: int = None) -> np.ndarray:
    """
    對緯向平均剖面 ū(y) 做空間滑動平均（邊界用 edge 值延伸，非週期性），
    去除小尺度渦流雜訊，避免其被誤判為噴流峰值或被微分放大。

    window: 滑動視窗大小（格點），須為奇數；預設取 cfg.JET_SMOOTH_WINDOW。
    """
    if window is None:
        window = cfg.JET_SMOOTH_WINDOW
    if window <= 1 or len(u_bar) < window:
        return u_bar.astype(np.float64)

    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(u_bar.astype(np.float64), pad, mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def count_jet_streams(u_bar, threshold=None):
    """
    偵測噴流數量：先做空間平滑再抓局部極值，避免小尺度渦流雜訊
    疊在大尺度噴流上造成過度偵測。
    """
    if threshold is None:
        threshold = cfg.JET_THRESHOLD
    u_smooth = smooth_profile(u_bar)
    u_std = np.std(u_smooth)
    if u_std < 1e-9: return 0
    u_norm = u_smooth / u_std
    peaks = 0
    for i in range(1, len(u_norm)-1):
        if abs(u_norm[i]) > threshold:
            if (u_norm[i] > u_norm[i-1] and u_norm[i] > u_norm[i+1]) or \
               (u_norm[i] < u_norm[i-1] and u_norm[i] < u_norm[i+1]):
                peaks += 1
    return peaks


def compute_eddy_flux(step_data: list[dict]) -> np.ndarray:
    """
    計算渦流動量通量 <u'v'>_x（Reynolds stress）。
    step_data: 多個時步的 {'ux': ..., 'uy': ...} 清單。
    用於分析帶狀流形成機制（逆能量串聯）。
    """
    if len(step_data) < 2:
        return np.zeros(cfg.NY)

    fluxes = []
    for d in step_data:
        ux, uy = d['ux'], d['uy']
        u_bar  = ux.mean(axis=1, keepdims=True)
        v_bar  = uy.mean(axis=1, keepdims=True)
        u_prime = ux - u_bar
        v_prime = uy - v_bar
        fluxes.append((u_prime * v_prime).mean(axis=1))

    return np.array(fluxes).mean(axis=0)   # 時間平均
