# utils/plotting.py  ── 視覺化工具
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg


def plot_velocity_magnitude(ux: np.ndarray, uy: np.ndarray,
                             step: int, save_path: str = None,
                             vmax: float = None) -> plt.Figure:
    """速度量值彩圖"""
    u_mag = np.sqrt(ux**2 + uy**2)
    vmax  = vmax or float(u_mag.max()) or cfg.U_MAX

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(u_mag, cmap='inferno', vmin=0, vmax=vmax,
                   origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='|u| (lattice units)')
    ax.set_title(f"Velocity Magnitude  –  Step {step}")
    ax.set_xlabel("x (zonal)")
    ax.set_ylabel("y (latitude)")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_vorticity(omega: np.ndarray, step: int,
                   save_path: str = None) -> plt.Figure:
    """渦度彩圖（紅藍對稱色標）"""
    vmax = float(np.abs(omega).max()) or 1e-3

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(omega, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='ω_z')
    ax.set_title(f"Vorticity  –  Step {step}")
    ax.set_xlabel("x (zonal)")
    ax.set_ylabel("y (latitude)")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_zonal_profile(y: np.ndarray, u_bar: np.ndarray,
                       step: int, save_path: str = None) -> plt.Figure:
    """帶狀平均速度剖面（偵測帶狀風系）"""
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(u_bar, y, color='royalblue', linewidth=1.5)
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.fill_betweenx(y, u_bar, 0,
                     where=(u_bar > 0), alpha=0.25, color='red',  label='Eastward')
    ax.fill_betweenx(y, u_bar, 0,
                     where=(u_bar < 0), alpha=0.25, color='blue', label='Westward')
    ax.set_xlabel("⟨u⟩_x  (lattice units)")
    ax.set_ylabel("y  (latitude index)")
    ax.set_title(f"Zonal Mean Wind  –  Step {step}")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_energy_spectrum(k: np.ndarray, E: np.ndarray,
                         step: int, slope: float = None,
                         save_path: str = None) -> plt.Figure:
    """對數-對數能量譜圖，並標示 Kolmogorov -5/3 參考線"""
    mask = (k > 0) & (E > 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k[mask], E[mask], 'o-', markersize=3,
              linewidth=1.2, label='E(k)', color='steelblue')

    if mask.sum() > 5:
        k_ref = k[mask]
        E_ref_53 = E[mask].max() * (k_ref / k_ref[0]) ** (-5/3)
        ax.loglog(k_ref, E_ref_53, 'k--', linewidth=0.8, label='k^{-5/3}')
        E_ref_3  = E[mask].max() * (k_ref / k_ref[0]) ** (-3)
        ax.loglog(k_ref, E_ref_3,  'r--', linewidth=0.8, label='k^{-3}')

    if slope is not None:
        ax.set_title(f"Energy Spectrum  –  Step {step}   (fitted slope: {slope:.2f})")
    else:
        ax.set_title(f"Energy Spectrum  –  Step {step}")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def make_animation_writer(filename: str, fps: int = 20) -> animation.FFMpegWriter:
    """建立 FFMpegWriter，回傳 writer 物件"""
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    return writer, filename
