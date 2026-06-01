#外掛工具
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#儲存路徑
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

#核心模組
import config as cfg
from config import init_constants
from core.lattice import feq_single,init_mrt_matrices
from core.collision import (
    f, f_new,
    rho_field, ux_field, uy_field,
    Fx_field, Fy_field,
    swap_fields,mrt_collision_kernel,init_fields,
)
from core.forcing import (
    #apply_coriolis_drag_update_f,
    update_zonal_noise,
    #apply_zonal_ar1_forcing,
    init_noise_fields
)
#分析模組
from analysis.vorticity  import get_vorticity_numpy
from analysis.spectrum   import compute_energy_spectrum, kolmogorov_slope
from analysis.zonal_mean import compute_zonal_mean, count_jet_streams

#視覺化工具
from utils.plotting import (
    plot_velocity_magnitude,
    plot_vorticity,
    plot_zonal_profile,
    plot_energy_spectrum
)

#創建資料夾
def setup_output_dirs():
    for d in [cfg.OUTPUT_DIR, cfg.DATA_DIR, "output/frames"]:
        os.makedirs(d, exist_ok=True)
    print("🪐 Jupiter LBM Simulation — initializing...")

#主程式
def run_simulation():
    #工具初始化
    setup_output_dirs()    
    init_constants()
    init_fields()
    init_mrt_matrices()
    init_noise_fields()
    print("Initialize finish.")

    #記錄台
    log = {'step': [], 'u_rms': [], 'jet_count': [], 'E_slope': [], 'T_std': []}

    #運行迴圈
    for step in range(cfg.MAX_STEPS):
        #模擬
        if step >= (cfg.WARMUP_STEPS) and step % 100 == 0:
            update_zonal_noise(cfg.alpha, cfg.sigma)          #仍先更新AR(1)噪音
        #純LBM+外力（streaming+MRT+forcing）
        mrt_collision_kernel(cfg.OMEGA,cfg.s1,cfg.s2,cfg.s4,cfg.s6,cfg.F0,cfg.BETA,cfg.EPSILON)   #已包含科氏力、阻尼、噪音
        swap_fields()

        #紀錄
        if step % cfg.SAVE_EVERY == 0:
            ux_np = ux_field.to_numpy()
            uy_np = uy_field.to_numpy()
            u_rms = float(np.sqrt((ux_np**2 + uy_np**2).mean()))
            zm    = compute_zonal_mean()
            jets  = count_jet_streams(zm['u_bar'])
            k_arr, E_arr = compute_energy_spectrum(ux_np, uy_np)
            slope = kolmogorov_slope(k_arr, E_arr)

            log['step'].append(step)
            log['u_rms'].append(u_rms)
            log['jet_count'].append(jets)
            log['E_slope'].append(slope if not np.isnan(slope) else 0.0)

            omega_np = get_vorticity_numpy()
            plot_velocity_magnitude(ux_np, uy_np, step,
                save_path=f"output/frames/vel_{step:06d}.jpg")
            plot_vorticity(omega_np, step,
                save_path=f"output/frames/vort_{step:06d}.jpg")
            print(f"Step {step:6d} | U_rms={u_rms:.5f} | Jets={jets} | "
                  f"E_slope={slope:.2f} | AR={'ON' if step>=(cfg.WARMUP_STEPS) else 'OFF'}")

            if step % cfg.SAVE_SPECTRUM == 0:
                plot_zonal_profile(zm['y'], zm['u_bar'], step,
                    save_path=f"output/frames/zonal_{step:06d}.png")
                plot_energy_spectrum(k_arr, E_arr, step, slope=slope,
                    save_path=f"output/frames/spectrum_{step:06d}.png")

    _save_summary(log)

#記錄總結
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