# param_sweep.py  ── 參數掃描控制器
# 對應 PDF §7 參數掃描設計（β, ΔT, ν, ε, τ）
#
# 使用方法：
#   python param_sweep.py
#
# 每組參數會：
#   1. 修改 config 設定值
#   2. 重新初始化場
#   3. 執行縮短版模擬（SWEEP_STEPS 步）
#   4. 記錄能量譜斜率、噴射流數量等指標
#
import os, sys, itertools
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── 掃描範圍（對應 PDF §7 表格）──
PARAM_GRID = {
    'BETA'    : [1e-5, 1e-4, 1e-3],          # Coriolis 梯度
    'DELTA_T' : [0.01, 0.05, 0.1],           # 熱梯度強度
    'NU'      : [0.001, 0.005, 0.01],         # 黏滯係數
    'NOISE_AMP': [1e-4, 1e-3, 1e-2],         # 噪音強度
}
SWEEP_STEPS  = 5000     # 每組掃描步數（短跑，快速評估）
SAVE_ROOT    = "sweep_results"


def run_single(params: dict, run_id: int) -> dict:
    """執行單組參數的模擬並回傳指標"""
    import importlib
    import config as cfg

    # ── 動態更新 config 參數 ──
    for key, val in params.items():
        setattr(cfg, key, val)
    cfg.TAU   = 3.0 * cfg.NU + 0.5
    cfg.OMEGA = float(1.0 / cfg.TAU)

    # ── 重新初始化（避免跨組污染）──
    from config import init_constants
    init_constants()

    from core.collision import init_fields
    init_fields()

    from physics.thermal import init_temperature
    init_temperature()

    from core.forcing import update_forcing_kernel, apply_noise_perturbation
    from core.collision import (bgk_collision_kernel, swap_fields,
                                apply_periodic_bc_y,
                                ux_field, uy_field)
    from physics.thermal import advect_temperature_kernel, relax_temperature_kernel
    from analysis.spectrum import compute_energy_spectrum, kolmogorov_slope
    from analysis.zonal_mean import compute_zonal_mean, count_jet_streams

    slopes, jets, u_rms_list = [], [], []

    for step in range(SWEEP_STEPS):
        update_forcing_kernel(cfg.F0, cfg.BETA, cfg.ALPHA_T)
        apply_noise_perturbation(step)
        bgk_collision_kernel(cfg.OMEGA)
        swap_fields()
        apply_periodic_bc_y()
        advect_temperature_kernel(1.0)
        relax_temperature_kernel(cfg.T0, cfg.DELTA_T, 5e-5)

        if step % 500 == 499:
            ux = ux_field.to_numpy()
            uy = uy_field.to_numpy()
            u_rms = float(np.sqrt((ux**2 + uy**2).mean()))
            k, E  = compute_energy_spectrum(ux, uy)
            sl    = kolmogorov_slope(k, E)
            zm    = compute_zonal_mean()
            jc    = count_jet_streams(zm['u_bar'])
            slopes.append(sl if not np.isnan(sl) else 0.0)
            jets.append(jc)
            u_rms_list.append(u_rms)

    result = {
        'run_id'    : run_id,
        'params'    : params.copy(),
        'mean_slope': float(np.nanmean(slopes)) if slopes else 0.0,
        'mean_jets' : float(np.mean(jets))      if jets   else 0.0,
        'mean_u_rms': float(np.mean(u_rms_list))if u_rms_list else 0.0,
        'final_slope': float(slopes[-1])         if slopes else 0.0,
        'final_jets' : int(jets[-1])             if jets   else 0,
    }
    return result


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # 產生參數組合（全因子掃描，共 3^4=81 組）
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    print(f"🔬 Parameter sweep: {len(combos)} combinations × {SWEEP_STEPS} steps each")
    print(f"   Parameters: {keys}")

    all_results = []
    for run_id, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        print(f"\n[{run_id+1:3d}/{len(combos)}] {params}")
        try:
            result = run_single(params, run_id)
            all_results.append(result)
            print(f"  → slope={result['mean_slope']:.2f}  "
                  f"jets={result['mean_jets']:.1f}  "
                  f"U_rms={result['mean_u_rms']:.5f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # 儲存結果
    out_path = os.path.join(SAVE_ROOT, "sweep_results.npy")
    np.save(out_path, all_results)
    print(f"\n✅ Sweep complete. Results saved to {out_path}")

    # 簡易摘要：找最多噴射流的參數組
    if all_results:
        best = max(all_results, key=lambda r: r['mean_jets'])
        print(f"\n🏆 Best (most jets): {best['params']}")
        print(f"   mean_jets={best['mean_jets']:.1f}  slope={best['mean_slope']:.2f}")


if __name__ == "__main__":
    main()
