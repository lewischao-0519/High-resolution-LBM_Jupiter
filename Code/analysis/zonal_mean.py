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


def count_jet_streams(u_bar: np.ndarray, threshold: float = 0.3) -> int:
    """
    計算帶狀平均速度剖面中的「噴射流」數量。
    定義：|u_bar| 超過 U_MAX * threshold 的局部極值數。
    """
    u_norm = u_bar / max(np.abs(u_bar).max(), 1e-9)
    peaks  = 0
    for i in range(1, len(u_norm) - 1):
        if abs(u_norm[i]) > threshold:
            if ((u_norm[i] > u_norm[i-1] and u_norm[i] > u_norm[i+1]) or
                (u_norm[i] < u_norm[i-1] and u_norm[i] < u_norm[i+1])):
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
