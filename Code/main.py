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
from analysis.vorticity   import get_vorticity_numpy
from analysis.spectrum    import compute_energy_spectrum, kolmogorov_slope
from analysis.zonal_mean  import compute_zonal_mean, count_jet_streams
from analysis.diagnostics import (
    rhines_scale, rossby_beta,
    jet_positions, kinetic_energy_decomp,
    reynolds_stress, vorticity_stats,
)

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
    log = {
        'step'          : [],
        'u_rms'         : [],
        'u_max'         : [],
        'jet_count'     : [],
        'E_slope'       : [],
        # ── 新增診斷量 ──
        'L_beta'        : [],   # Rhines 尺度 (格點)
        'Ro_beta'       : [],   # β-Rossby 數
        'jet_positions' : [],   # 噴流 y-index 列表
        'KE_zonal'      : [],   # 帶狀動能
        'KE_eddy'       : [],   # 渦流動能
        'zonal_frac'    : [],   # KE_zonal / KE_total
        'RS_mean'       : [],   # Reynolds stress 平均值
        'omega_rms'     : [],   # 渦度 RMS
        'omega_skew'    : [],   # 渦度偏態（負 → 反氣旋主導）
    }

    #運行迴圈
    for step in range(cfg.MAX_STEPS+1):
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
            u_mag = np.sqrt(ux_np**2 + uy_np**2)
            u_rms = float(u_mag.mean())
            u_max = float(u_mag.max())
            zm    = compute_zonal_mean()
            jets  = count_jet_streams(zm['u_bar'])
            k_arr, E_arr = compute_energy_spectrum(ux_np, uy_np)
            slope = kolmogorov_slope(k_arr, E_arr)

            # ── 新增診斷量 ──
            L_b   = rhines_scale(u_rms)
            Ro_b  = rossby_beta(u_rms)
            jpos  = jet_positions(zm['u_bar'])
            ke    = kinetic_energy_decomp(ux_np, uy_np)
            omega_np = get_vorticity_numpy()
            rs    = reynolds_stress(ux_np, uy_np)
            vs    = vorticity_stats(omega_np)

            log['step'].append(step)
            log['u_rms'].append(u_rms)
            log['u_max'].append(u_max)
            log['jet_count'].append(jets)
            log['E_slope'].append(slope if not np.isnan(slope) else 0.0)
            log['L_beta'].append(L_b)
            log['Ro_beta'].append(Ro_b)
            log['jet_positions'].append(jpos)
            log['KE_zonal'].append(ke['KE_zonal'])
            log['KE_eddy'].append(ke['KE_eddy'])
            log['zonal_frac'].append(ke['zonal_fraction'])
            log['RS_mean'].append(rs['RS_mean'])
            log['omega_rms'].append(vs['omega_rms'])
            log['omega_skew'].append(vs['omega_skew'])

            if u_max > 0.3:
                print(f"⚠️ WARNING: Maximum velocity u_max = {u_max:.5f} exceeds the stability limit 0.3!")

            plot_velocity_magnitude(ux_np, uy_np, step,
                save_path=f"output/frames/vel_{step:06d}.jpg")
            plot_vorticity(omega_np, step,
                save_path=f"output/frames/vort_{step:06d}.jpg")
            print(f"Step {step:6d} | U_rms={u_rms:.5f} | U_max={u_max:.5f} | Jets={jets} | "
                  f"E_slope={slope:.2f} | L_β={L_b:.1f} | KE_zon%={ke['zonal_fraction']*100:.1f}% | "
                  f"ω_skew={vs['omega_skew']:+.3f} | AR={'ON' if step>=(cfg.WARMUP_STEPS) else 'OFF'}")

            if step % cfg.SAVE_SPECTRUM == 0:
                plot_zonal_profile(zm['y'], zm['u_bar'], step,
                    save_path=f"output/frames/zonal_{step:06d}.png")
                plot_energy_spectrum(k_arr, E_arr, step, slope=slope,
                    save_path=f"output/frames/spectrum_{step:06d}.png")
    
    _save_summary(log)
        
#記錄總結
def _save_summary(log: dict):
    """儲存最終時間序列摘要圖（2 張：基礎 + 延伸診斷）"""
    steps = log['step']
    if len(steps) == 0:
        return

    # ── 圖1：基礎摘要（原有）──
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

    axes[1, 1].plot(steps, log['zonal_frac'], color='purple')
    axes[1, 1].axhline(0.6, color='k', linestyle='--', linewidth=0.8, label='Jupiter ~60%')
    axes[1, 1].set_title("Zonal KE Fraction  (KE_zonal / KE_total)")
    axes[1, 1].set_ylabel("fraction")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    for ax in axes.flat:
        ax.set_xlabel("Step")
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "summary.png"), dpi=200)
    plt.close(fig)

    # ── 圖2：延伸診斷（新增）──
    fig2, axs = plt.subplots(2, 3, figsize=(16, 8))
    fig2.suptitle("Jupiter LBM — Extended Diagnostics (vs. Observations)", fontsize=13)

    # (0,0) Rhines 尺度
    axs[0, 0].plot(steps, log['L_beta'], color='teal')
    axs[0, 0].set_title("Rhines Scale  L_β = √(U_rms/β)")
    axs[0, 0].set_ylabel("L_β  (lattice units)")
    # 理論噴流間距 ≈ 2π L_β
    ax2 = axs[0, 0].twinx()
    ax2.plot(steps, [2*np.pi*v for v in log['L_beta']], color='teal', alpha=0.3, linestyle=':')
    ax2.set_ylabel("2π L_β (predicted jet spacing)", color='teal', fontsize=8)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # (0,1) β-Rossby 數
    axs[0, 1].plot(steps, log['Ro_beta'], color='goldenrod')
    axs[0, 1].set_title("β-Rossby Number  Ro_β")
    axs[0, 1].set_ylabel("Ro_β")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # (0,2) KE 分解
    axs[0, 2].plot(steps, log['KE_zonal'], label='KE_zonal', color='royalblue')
    axs[0, 2].plot(steps, log['KE_eddy'],  label='KE_eddy',  color='salmon')
    axs[0, 2].set_title("Kinetic Energy Decomposition")
    axs[0, 2].set_ylabel("KE  (lattice units²)")
    axs[0, 2].legend(fontsize=8)
    axs[0, 2].grid(True, linestyle='--', alpha=0.5)

    # (1,0) Reynolds stress
    axs[1, 0].plot(steps, log['RS_mean'], color='darkorchid')
    axs[1, 0].set_title("Reynolds Stress  |<u'v'>|  (domain mean)")
    axs[1, 0].set_ylabel("RS_mean")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # (1,1) 渦度 RMS
    axs[1, 1].plot(steps, log['omega_rms'], color='darkgreen')
    axs[1, 1].set_title("Vorticity RMS  ω_rms")
    axs[1, 1].set_ylabel("ω_rms")
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    # (1,2) 渦度偏態（最關鍵：負 → 反氣旋主導，對應木星）
    axs[1, 2].plot(steps, log['omega_skew'], color='firebrick')
    axs[1, 2].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axs[1, 2].fill_between(steps, log['omega_skew'], 0,
                            where=[v < 0 for v in log['omega_skew']],
                            alpha=0.25, color='blue', label='Anticyclone-dominant (Jupiter-like)')
    axs[1, 2].fill_between(steps, log['omega_skew'], 0,
                            where=[v >= 0 for v in log['omega_skew']],
                            alpha=0.25, color='red', label='Cyclone-dominant')
    axs[1, 2].set_title("Vorticity Skewness  (< 0 = Jupiter-like)")
    axs[1, 2].set_ylabel("skewness")
    axs[1, 2].legend(fontsize=7)
    axs[1, 2].grid(True, linestyle='--', alpha=0.5)

    for ax in axs.flat:
        ax.set_xlabel("Step")
    plt.tight_layout()
    fig2.savefig(os.path.join(cfg.OUTPUT_DIR, "summary_diagnostics.png"), dpi=200)
    plt.close(fig2)

    # ── 噴流位置時空圖（hovmöller diagram）──
    if len(log['jet_positions']) > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        for t_idx, (s, pos_list) in enumerate(zip(steps, log['jet_positions'])):
            if pos_list:
                ax3.scatter([s]*len(pos_list), pos_list,
                            s=4, c='navy', alpha=0.5)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("y (latitude index)")
        ax3.set_title("Jet Position Hovmöller Diagram")
        ax3.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        fig3.savefig(os.path.join(cfg.OUTPUT_DIR, "jet_hovmoller.png"), dpi=200)
        plt.close(fig3)

    # 儲存數值資料
    np.save(os.path.join(cfg.DATA_DIR, "log.npy"), log)
    print(f"  → summary.png and log.npy saved.")

if __name__ == "__main__":
    run_simulation()