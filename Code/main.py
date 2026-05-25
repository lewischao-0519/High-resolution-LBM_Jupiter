import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── 設定 Python path ──
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as cfg
from config import init_constants

# ── 核心模組 ──
from core.collision import (
    f, f_new,
    rho_field, ux_field, uy_field,
    Fx_field, Fy_field,
    init_fields, bgk_collision_kernel,apply_boundary_y_free_slip, swap_fields,
    compute_macro
)
#from core.forcing import update_forcing_kernel

# ── 分析模組 ──
from analysis.vorticity  import get_vorticity_numpy
from analysis.spectrum   import compute_energy_spectrum, kolmogorov_slope
from analysis.zonal_mean import compute_zonal_mean, count_jet_streams

# ── 視覺化工具 ──
from utils.plotting import (
    plot_velocity_magnitude,
    plot_vorticity,
    plot_zonal_profile,
    plot_energy_spectrum
)

def setup_output_dirs():
    """建立輸出資料夾"""
    for d in [cfg.OUTPUT_DIR, cfg.DATA_DIR, "output/frames"]:
        os.makedirs(d, exist_ok=True)


def run_simulation():
    # ── 1. 初始化 ──
    print("🪐 Jupiter LBM Simulation — initializing...")
    init_constants()
    init_fields()
    setup_output_dirs()

    # ── 2. 資料記錄器 ──
    log = {
        'step'        : [],
        'u_rms'       : [],
        'jet_count'   : [],
        'E_slope'     : [],
        'T_std'       : [],
    }
    # 用於計算渦流通量的歷史資料（保留最近 10 幀）
    velocity_history = []
    # ── 影片設定（已禁用，仅保存静态图）──
    print(f"🚀 Starting main loop  (MAX_STEPS={cfg.MAX_STEPS}) ... (video disabled)")
    
    for step in range(cfg.MAX_STEPS):
        # ── A. 更新外力場（Coriolis + 熱力）──
        #update_forcing_kernel(cfg.F0, cfg.BETA, cfg.ALPHA_T)

        # ── B. BGK 碰撞 + 串流（Loop Fusion）──
        bgk_collision_kernel(cfg.OMEGA)
        #apply_boundary_y_free_slip() 
        swap_fields()
        compute_macro()

        # ── C. 每 SAVE_EVERY 步做記錄與保存靜態圖 ──
        if step % cfg.SAVE_EVERY == 0:
            ux_np = ux_field.to_numpy()
            uy_np = uy_field.to_numpy()

            # 基本統計
            u_rms  = float(np.sqrt((ux_np**2 + uy_np**2).mean()))
            zm     = compute_zonal_mean()
            jets   = count_jet_streams(zm['u_bar'])

            # 能量譜
            k_arr, E_arr = compute_energy_spectrum(ux_np, uy_np)
            slope        = kolmogorov_slope(k_arr, E_arr)

            # 記錄
            log['step'].append(step)
            log['u_rms'].append(u_rms)
            log['jet_count'].append(jets)
            log['E_slope'].append(slope if not np.isnan(slope) else 0.0)

            # 渦度
            omega_np = get_vorticity_numpy()

            # 保存靜態圖
            u_mag = np.sqrt(ux_np**2 + uy_np**2)
            plot_velocity_magnitude(ux_np, uy_np, step, save_path=f"output/frames/vel_{step:06d}.jpg")
            plot_vorticity(omega_np, step, save_path=f"output/frames/vort_{step:06d}.jpg")
            if step % cfg.SAVE_EVERY == 0:
                ux_np = ux_field.to_numpy()
                uy_np = uy_field.to_numpy()
                u_mag = np.sqrt(ux_np**2 + uy_np**2)
                print(f"Step {step}: |u| min={u_mag.min():.2e}, max={u_mag.max():.2e}")
            # 每 20000 步保存剖面和能譜圖
            if step % 20000 == 0:
                plot_zonal_profile(zm['y'], zm['u_bar'], step, save_path=f"output/frames/zonal_{step:06d}.png")
                plot_energy_spectrum(k_arr, E_arr, step, slope=slope, save_path=f"output/frames/spectrum_{step:06d}.png")
                print(f"  Step {step:6d} | U_rms={u_rms:.5f} | Jets={jets} | E_slope={slope:.2f}")

    print("✅ Simulation done! Generating summary plots...")
    _save_summary(log)
    print(f"📁 All outputs saved to '{cfg.OUTPUT_DIR}/'")


def _save_summary(log: dict):
    """儲存最終時間序列摘要圖"""
    steps = log['step']
    if len(steps) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Jupiter LBM — Simulation Summary", fontsize=13)

    axes[0, 0].plot(steps, log['u_rms'], color='steelblue')
    axes[0, 0].set_title("RMS Velocity")
    axes[0, 0].set_ylabel("|u|_rms")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    axes[0, 1].plot(steps, log['jet_count'], color='darkorange', drawstyle='steps-mid')
    axes[0, 1].set_title("Jet Stream Count")
    axes[0, 1].set_ylabel("# of jets")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    axes[1, 0].plot(steps, log['E_slope'], color='firebrick')
    axes[1, 0].axhline(-5/3, color='k', linestyle='--', linewidth=0.8, label='-5/3')
    axes[1, 0].axhline(-3,   color='b', linestyle='--', linewidth=0.8, label='-3')
    axes[1, 0].set_title("Energy Spectrum Slope")
    axes[1, 0].set_ylabel("α  (E~k^α)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)


    for ax in axes.flat:
        ax.set_xlabel("Step")

    plt.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "summary.png"), dpi=200)
    plt.close(fig)

    # 儲存數值資料
    np.save(os.path.join(cfg.DATA_DIR, "log.npy"), log)
    print(f"  → summary.png and log.npy saved.")


if __name__ == "__main__":
    run_simulation()