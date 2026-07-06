#外掛工具
import os, sys, subprocess, shutil, csv, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# print() 大量使用 Emoji；stdout 一旦不是接到真正的終端機（被導向檔案、
# 或透過 sweep.py 平行執行時），Windows 會退回用系統 locale 編碼（例如
# cp950）而無法編碼 Emoji 直接 crash。強制用 UTF-8 避免這個跟模擬邏輯
# 無關的錯誤。
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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
    mrt_collision_kernel,init_fields,
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
    pv_gradient, pv_staircase,
    estimate_energy_injection, zonostrophy_index,
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
    for d in [cfg.OUTPUT_DIR, cfg.DATA_DIR, os.path.join(cfg.OUTPUT_DIR, "frames")]:
        os.makedirs(d, exist_ok=True)
    print("🪐 Jupiter LBM Simulation — initializing...")

# 需要記錄的 config 參數（排除 ti.field / numpy 陣列 / 函式等不適合純文字記錄的項目）
_CONFIG_PARAMS = [
    'NX', 'NY', 'MAX_STEPS', 'SAVE_EVERY', 'SAVE_SPECTRUM',
    'F0', 'NY_REF', 'BETA_REF', 'BETA',
    'EPSILON',
    'SPONGE_FRAC', 'EPSILON_MAX',
    'Tc', 'alpha', 'sigma', 'WARMUP_STEPS',
    's1', 's2', 's4', 's6',
    'U_MAX', 'NU', 'TAU', 'OMEGA',
    'JET_SMOOTH_WINDOW', 'JET_THRESHOLD',
    'PROFILE_SNAPSHOT_EVERY', 'HEAVY_CHECKPOINT_EVERY',
    'OUTPUT_DIR', 'DATA_DIR',
]

def _save_config_snapshot():
    """
    把這次模擬實際使用的 config 參數存成一份帶時間戳記的 txt，
    方便事後比對「這批數據是用哪組參數跑出來的」，避免跟其他次
    模擬的結果搞混（尤其是平滑視窗、門檻這類會直接影響診斷結果的參數）。
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(cfg.DATA_DIR, f'config_{timestamp}.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f"# Jupiter LBM config snapshot — {timestamp}\n")
        for name in _CONFIG_PARAMS:
            fh.write(f"{name} = {getattr(cfg, name)}\n")
    print(f"  → config 快照已儲存：{path}")

# 各診斷量 CSV 的欄位定義
_METRIC_HEADERS = [
    ('u_rms',         ['step', 'u_rms']),
    ('u_max',         ['step', 'u_max']),
    ('jet_count',     ['step', 'jet_count']),
    ('E_slope',       ['step', 'E_slope']),
    ('L_beta',        ['step', 'L_beta']),
    ('Ro_beta',       ['step', 'Ro_beta']),
    ('jet_positions', ['step', 'jet_positions']),
    ('KE_zonal',      ['step', 'KE_zonal']),
    ('KE_eddy',       ['step', 'KE_eddy']),
    ('zonal_frac',    ['step', 'zonal_frac']),
    ('RS_mean',       ['step', 'RS_mean']),
    ('omega_rms',        ['step', 'omega_rms']),
    ('omega_skew',       ['step', 'omega_skew']),
    # ── 新增診斷純量 ──
    ('Qy_sign_changes',    ['step', 'Qy_sign_changes']),
    ('staircase_score',    ['step', 'staircase_score']),
    ('R_beta_star',        ['step', 'R_beta_star']),
    ('epsilon_injection',  ['step', 'epsilon_injection']),
]

def _init_metric_csvs():
    """清除舊資料並為各診斷量建立帶標題的 CSV 檔案。"""
    for name, header in _METRIC_HEADERS:
        path = os.path.join(cfg.DATA_DIR, f"{name}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerow(header)
    print(f"  → {len(_METRIC_HEADERS)} 個診斷量 CSV 已初始化 ({cfg.DATA_DIR}/)")

def _append_metric_csvs(step: int, u_rms, u_max, jet_count, E_slope,
                         L_b, Ro_b, jpos, ke, rs, vs,
                         Qy_sign_changes, staircase_score, R_beta_star, eps_inj):
    """將本步診斷量逐一追加到各自的 CSV 檔案。"""
    rows = {
        'u_rms'            : f"{u_rms:.6f}",
        'u_max'            : f"{u_max:.6f}",
        'jet_count'        : jet_count,
        'E_slope'          : f"{E_slope:.4f}",
        'L_beta'           : f"{L_b:.4f}",
        'Ro_beta'          : f"{Ro_b:.6e}",
        'jet_positions'    : '|'.join(map(str, jpos)),
        'KE_zonal'         : f"{ke['KE_zonal']:.6e}",
        'KE_eddy'          : f"{ke['KE_eddy']:.6e}",
        'zonal_frac'       : f"{ke['zonal_fraction']:.4f}",
        'RS_mean'          : f"{rs['RS_mean']:.6e}",
        'omega_rms'        : f"{vs['omega_rms']:.6e}",
        'omega_skew'       : f"{vs['omega_skew']:+.4f}",
        'Qy_sign_changes'  : Qy_sign_changes,
        'staircase_score'  : f"{staircase_score:.4f}",
        'R_beta_star'      : f"{R_beta_star:.4f}",
        'epsilon_injection': f"{eps_inj:.6e}",
    }
    for name, value in rows.items():
        path = os.path.join(cfg.DATA_DIR, f"{name}.csv")
        with open(path, 'a', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerow([step, value])

# 剖面型 CSV（每列 = 一個時步的完整 y 剖面）的欄位定義
_PROFILE_HEADERS = [
    ('ubar_hovmoller', 'y'),
    ('Qy_profile',     'y'),
    ('q_profile',      'y'),
]
_JET_LIFETIME_COLS = ['jet_id', 'born_step', 'died_step', 'duration', 'mean_y', 'ended_by']

def _init_profile_csvs():
    """
    清除舊資料並為剖面型 CSV 建立標題列。這些檔案之後全部用『追加寫入』
    維護（每次只寫一列新資料），成本是 O(1)，不會隨模擬進度變慢——
    不像先前每次checkpoint都要重寫整段歷史（O(T)，隨時間增長成本暴增，
    拖慢GPU迴圈、抵銷平行運算的效能）。
    """
    for name, prefix in _PROFILE_HEADERS:
        path = os.path.join(cfg.DATA_DIR, f"{name}.csv")
        header = ['step'] + [f'{prefix}{j}' for j in range(cfg.NY)]
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerow(header)

    lt_path = os.path.join(cfg.DATA_DIR, 'jet_lifetime.csv')
    with open(lt_path, 'w', newline='', encoding='utf-8') as fh:
        csv.writer(fh).writerow(_JET_LIFETIME_COLS)

    print(f"  → 剖面 CSV（ubar_hovmoller / Qy_profile / q_profile / jet_lifetime）已初始化")

def _append_profile_row(filename: str, step: int, arr: np.ndarray):
    """對單一剖面 CSV 追加一列（O(1)，不重寫既有內容）。"""
    path = os.path.join(cfg.DATA_DIR, f"{filename}.csv")
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        csv.writer(fh).writerow([step] + [f"{v:.6e}" for v in arr])

def _append_jet_lifetime_rows(records: list):
    """對 jet_lifetime.csv 追加新結束（或存活到模擬結束）的噴流紀錄。"""
    if not records:
        return
    path = os.path.join(cfg.DATA_DIR, 'jet_lifetime.csv')
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        for rec in records:
            w.writerow([rec['jet_id'], rec['born_step'], rec['died_step'],
                        rec['duration'], rec['mean_y'], rec['ended_by']])

#主程式
def run_simulation():
    #工具初始化
    setup_output_dirs()
    _save_config_snapshot()
    _init_metric_csvs()
    _init_profile_csvs()
    init_constants()
    init_fields()
    init_mrt_matrices()
    init_noise_fields()
    print("Initialize finish.")

    #記錄台（純量時序）
    log = {
        'step'            : [],
        'u_rms'           : [],
        'u_max'           : [],
        'jet_count'       : [],
        'E_slope'         : [],
        'L_beta'          : [],   # Rhines 尺度 (格點)
        'Ro_beta'         : [],   # β-Rossby 數
        'jet_positions'   : [],   # 噴流 y-index 列表
        'KE_zonal'        : [],   # 帶狀動能
        'KE_eddy'         : [],   # 渦流動能
        'zonal_frac'      : [],   # KE_zonal / KE_total
        'RS_mean'         : [],   # Reynolds stress 空間平均純量
        'omega_rms'       : [],   # 渦度 RMS
        'omega_skew'      : [],   # 渦度偏態（負 → 反氣旋主導）
        'Qy_sign_changes'  : [],   # Q_y 符號反轉次數（正壓不穩定判準）
        'staircase_score'  : [],   # PV 階梯分數
        'R_beta_star'      : [],   # Zonostrophy index
        'epsilon_injection': [],   # 能量注入率代理 ε（R_β* 公式關鍵量）
    }

    # 剖面歷史（每個 SAVE_EVERY 步一筆，最終儲存為 NPZ + CSV）
    prof = {
        'u_bar'     : [],   # (T, NY)  Hovmöller 資料
        'RS_profile': [],   # (T, NY)  完整 Reynolds stress 剖面
        'Qy'        : [],   # (T, NY)  位渦梯度剖面
        'q_pv'      : [],   # (T, NY)  PV 剖面
    }

    # 噴流壽命追蹤
    jet_tracker    = {}   # jet_id → {'born': step, 'last_pos': y}
    jet_id_counter = 0
    jet_lifetimes  = []   # 已結束噴流的紀錄清單

    #ping-pong 緩衝（避免每步整場複製）
    src, dst = f, f_new

    #運行迴圈
    for step in range(cfg.MAX_STEPS+1):
        #模擬
        if step >= (cfg.WARMUP_STEPS) and step % 100 == 0:
            update_zonal_noise(cfg.alpha, cfg.sigma)          #仍先更新AR(1)噪音
        #純LBM+外力（streaming+MRT+forcing）
        mrt_collision_kernel(src, dst, cfg.OMEGA,cfg.s1,cfg.s2,cfg.s4,cfg.s6,cfg.F0,cfg.BETA,cfg.EPSILON)   #已包含科氏力、阻尼、噪音
        src, dst = dst, src   #交換讀寫緩衝（僅換指標，不搬資料）

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

            # ── 診斷量（純量） ──
            L_b      = rhines_scale(u_rms)
            Ro_b     = rossby_beta(u_rms)
            jpos     = jet_positions(zm['u_bar'])
            ke       = kinetic_energy_decomp(ux_np, uy_np)
            omega_np = get_vorticity_numpy()
            rs       = reynolds_stress(ux_np, uy_np)
            vs       = vorticity_stats(omega_np)

            # ── 新增診斷量 ──
            pv_grad  = pv_gradient(zm['u_bar'])
            pv_st    = pv_staircase(zm['u_bar'])
            eps_inj  = estimate_energy_injection(ke['KE_total'])
            R_b_star = zonostrophy_index(u_rms, eps_inj)

            # ── 剖面歷史（存 NPZ 用，僅留在記憶體，重量級存檔才落地） ──
            prof['u_bar'].append(zm['u_bar'].astype(np.float32))
            prof['RS_profile'].append(rs['RS_profile'].astype(np.float32))
            prof['Qy'].append(pv_grad['Qy'])
            prof['q_pv'].append(pv_st['q_profile'])

            # ── 剖面 CSV：O(1) 增量追加，不重寫歷史，維持高頻率也不影響效能 ──
            _append_profile_row('ubar_hovmoller', step, zm['u_bar'])
            if step % cfg.PROFILE_SNAPSHOT_EVERY == 0:
                _append_profile_row('Qy_profile', step, pv_grad['Qy'])
                _append_profile_row('q_profile',  step, pv_st['q_profile'])

            # ── 噴流壽命追蹤：噴流消失當下立刻把該筆紀錄追加進 CSV ──
            jet_id_counter, newly_dead = _update_jet_tracker(
                step, jpos, jet_tracker, jet_id_counter, jet_lifetimes)
            _append_jet_lifetime_rows(newly_dead)

            _append_metric_csvs(step, u_rms, u_max, jets, slope,
                                L_b, Ro_b, jpos, ke, rs, vs,
                                pv_grad['Qy_sign_changes'],
                                pv_st['staircase_score'],
                                R_b_star, eps_inj)

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
            log['Qy_sign_changes'].append(pv_grad['Qy_sign_changes'])
            log['staircase_score'].append(pv_st['staircase_score'])
            log['R_beta_star'].append(R_b_star)
            log['epsilon_injection'].append(eps_inj)

            if u_max > 0.3:
                print(f"⚠️ WARNING: Maximum velocity u_max = {u_max:.5f} exceeds the stability limit 0.3!")

            plot_velocity_magnitude(ux_np, uy_np, step,
                save_path=os.path.join(cfg.OUTPUT_DIR, "frames", f"vel_{step:08d}.jpg"))
            plot_vorticity(omega_np, step,
                save_path=os.path.join(cfg.OUTPUT_DIR, "frames", f"vort_{step:08d}.jpg"))
            print(f"Step {step:6d} | U_rms={u_rms:.5f} | U_max={u_max:.5f} | Jets={jets} | "
                  f"E_slope={slope:.2f} | L_β={L_b:.1f} | KE_zon%={ke['zonal_fraction']*100:.1f}% | "
                  f"ω_skew={vs['omega_skew']:+.3f} | AR={'ON' if step>=(cfg.WARMUP_STEPS) else 'OFF'}")

            if step % cfg.SAVE_SPECTRUM == 0:
                plot_zonal_profile(zm['y'], zm['u_bar'], step,
                    save_path=os.path.join(cfg.OUTPUT_DIR, "frames", f"zonal_{step:08d}.png"))
                plot_energy_spectrum(k_arr, E_arr, step, slope=slope,
                    save_path=os.path.join(cfg.OUTPUT_DIR, "frames", f"spectrum_{step:08d}.png"))

            # ── 重量級檢查點（低頻率）：summary.png / hovmoller.png /
            #    jet_autocorr.png / NPZ。這些需要重繪圖表、跑 FFT 自相關、
            #    輸出整段歷史陣列，成本遠高於上面的逐步增量 CSV，因此用
            #    獨立、低得多的頻率（cfg.HEAVY_CHECKPOINT_EVERY）執行，
            #    避免拖慢 GPU 迴圈、抵銷平行運算的效能 ──
            if step % cfg.HEAVY_CHECKPOINT_EVERY == 0:
                print(f"  ⏺ Step {step}: 寫入重量級檢查點...")
                _save_summary(log)
                _save_profile_artifacts(prof, log['step'])

    # 模擬結束：將仍存活的噴流記為「活到最後」，並追加進 jet_lifetime.csv
    final_survivors = [{
        'jet_id'    : jid,
        'born_step' : info['born'],
        'died_step' : cfg.MAX_STEPS,
        'duration'  : cfg.MAX_STEPS - info['born'],
        'mean_y'    : info['last_pos'],
        'ended_by'  : 'simulation_end',
    } for jid, info in jet_tracker.items()]
    _append_jet_lifetime_rows(final_survivors)

    _save_summary(log)
    _save_profile_artifacts(prof, log['step'])
    _make_videos()


def _make_videos():
    """模擬結束後自動合成影片，完成後刪除暫存 frames。"""
    script = os.path.join(ROOT, "utils", "make_video.py")
    # ffmpeg 的 concat demuxer 會把清單裡的相對路徑解析成「相對於暫存清單檔
    # 案所在目錄」而非目前工作目錄，因此這裡一律轉成絕對路徑再傳給子行程，
    # 避免平行掃描（cfg.OUTPUT_DIR 可能是相對路徑）時找不到 frame 檔案。
    out_dir    = os.path.abspath(cfg.OUTPUT_DIR)
    frames_dir = os.path.join(out_dir, "frames")
    print("\n🎬 開始合成影片...")
    result = subprocess.run(
        [sys.executable, script, "--fps", "10", "--fps-sparse", "4",
         "--frames-dir", frames_dir, "--out-dir", out_dir],
        capture_output=False
    )
    if result.returncode != 0:
        print("⚠️  影片合成失敗（ffmpeg 未安裝？），請手動執行：")
        print(f"   python3 utils/make_video.py --frames-dir {frames_dir} --out-dir {out_dir}")
        return
    # 清除暫存 frames 資料夾
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
        print(f"  🗑️  已刪除暫存 frames：{frames_dir}")


#記錄總結
def _save_summary(log: dict):
    """儲存 summary.png 與 log.csv"""
    steps = log['step']
    if len(steps) == 0:
        return

    # ── summary.png：3×3 全診斷量一圖 ──
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle("Jupiter LBM — Simulation Summary", fontsize=14)

    # Row 0
    axes[0, 0].plot(steps, log['u_rms'], color='steelblue')
    axes[0, 0].set_title("RMS Velocity")
    axes[0, 0].set_ylabel("|u|_rms")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    axes[0, 1].plot(steps, log['jet_count'], color='darkorange', drawstyle='steps-mid')
    axes[0, 1].set_title("Jet Stream Count")
    axes[0, 1].set_ylabel("# of jets")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    axes[0, 2].plot(steps, log['E_slope'], color='firebrick')
    axes[0, 2].axhline(-5/3, color='k', linestyle='--', linewidth=0.8, label='-5/3')
    axes[0, 2].axhline(-3,   color='b', linestyle='--', linewidth=0.8, label='-3')
    axes[0, 2].set_title("Energy Spectrum Slope")
    axes[0, 2].set_ylabel("α  (E~k^α)")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, linestyle='--', alpha=0.5)

    # Row 1
    axes[1, 0].plot(steps, log['zonal_frac'], color='purple')
    axes[1, 0].axhline(0.6, color='k', linestyle='--', linewidth=0.8, label='Jupiter ~60%')
    axes[1, 0].set_title("Zonal KE Fraction")
    axes[1, 0].set_ylabel("KE_zonal / KE_total")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    axes[1, 1].plot(steps, log['omega_rms'],  color='darkgreen', label='ω_rms')
    axes[1, 1].plot(steps, log['RS_mean'],     color='darkorchid', label="|<u'v'>|")
    axes[1, 1].set_title("Vorticity RMS & Reynolds Stress")
    axes[1, 1].set_ylabel("magnitude")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    axes[1, 2].plot(steps, log['omega_skew'], color='firebrick')
    axes[1, 2].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[1, 2].fill_between(steps, log['omega_skew'], 0,
                             where=[v < 0 for v in log['omega_skew']],
                             alpha=0.25, color='blue', label='Anticyclone-dominant')
    axes[1, 2].fill_between(steps, log['omega_skew'], 0,
                             where=[v >= 0 for v in log['omega_skew']],
                             alpha=0.25, color='red', label='Cyclone-dominant')
    axes[1, 2].set_title("Vorticity Skewness  (< 0 = Jupiter-like)")
    axes[1, 2].set_ylabel("skewness")
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].grid(True, linestyle='--', alpha=0.5)

    # Row 2 — 新增診斷純量
    axes[2, 0].plot(steps, log['R_beta_star'], color='teal')
    axes[2, 0].axhline(2.5, color='g', linestyle='--', linewidth=0.8, label='R_β*=2.5 (zonostrophic)')
    axes[2, 0].axhline(1.5, color='r', linestyle='--', linewidth=0.8, label='R_β*=1.5 (friction-dom)')
    axes[2, 0].set_title("Zonostrophy Index  R_β*")
    axes[2, 0].set_ylabel("R_β* = L_R / L_t")
    axes[2, 0].legend(fontsize=7)
    axes[2, 0].grid(True, linestyle='--', alpha=0.5)

    axes[2, 1].plot(steps, log['Qy_sign_changes'], color='sienna', drawstyle='steps-mid')
    axes[2, 1].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[2, 1].set_title("Rayleigh–Kuo: Q_y Sign Changes")
    axes[2, 1].set_ylabel("# sign reversals  (>0 = unstable)")
    axes[2, 1].grid(True, linestyle='--', alpha=0.5)

    axes[2, 2].plot(steps, log['staircase_score'], color='darkviolet')
    axes[2, 2].axhline(5.0, color='g', linestyle='--', linewidth=0.8, label='score=5 (staircase-like)')
    axes[2, 2].set_title("PV Staircase Score")
    axes[2, 2].set_ylabel("max|dq/dy| / mean|dq/dy|")
    axes[2, 2].legend(fontsize=8)
    axes[2, 2].grid(True, linestyle='--', alpha=0.5)

    for ax in axes.flat:
        ax.set_xlabel("Step")
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "summary.png"), dpi=200)
    plt.close(fig)
    print(f"  → summary.png saved.")

    # ── log.csv：數據資料表 ──
    csv_path = os.path.join(cfg.DATA_DIR, "log.csv")
    # jet_positions 為 list-of-lists，序列化為字串
    columns = ['step', 'u_rms', 'u_max', 'jet_count', 'E_slope',
               'L_beta', 'Ro_beta', 'jet_positions',
               'KE_zonal', 'KE_eddy', 'zonal_frac',
               'RS_mean', 'omega_rms', 'omega_skew',
               'Qy_sign_changes', 'staircase_score', 'R_beta_star',
               'epsilon_injection']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(len(steps)):
            writer.writerow([
                log['step'][i],
                f"{log['u_rms'][i]:.6f}",
                f"{log['u_max'][i]:.6f}",
                log['jet_count'][i],
                f"{log['E_slope'][i]:.4f}",
                f"{log['L_beta'][i]:.4f}",
                f"{log['Ro_beta'][i]:.6e}",
                '|'.join(map(str, log['jet_positions'][i])),
                f"{log['KE_zonal'][i]:.6e}",
                f"{log['KE_eddy'][i]:.6e}",
                f"{log['zonal_frac'][i]:.4f}",
                f"{log['RS_mean'][i]:.6e}",
                f"{log['omega_rms'][i]:.6e}",
                f"{log['omega_skew'][i]:+.4f}",
                log['Qy_sign_changes'][i],
                f"{log['staircase_score'][i]:.4f}",
                f"{log['R_beta_star'][i]:.4f}",
                f"{log['epsilon_injection'][i]:.6e}",
            ])
    print(f"  → log.csv saved  ({len(steps)} rows × {len(columns)} cols)")

def _update_jet_tracker(step: int, jpos: list,
                         jet_tracker: dict, jet_id_counter: int,
                         jet_lifetimes: list):
    """
    以 greedy 最近鄰配對追蹤噴流跨時步的身份。

    - 每個現有噴流找最近的新位置（距離 ≤ MATCH_DIST）→ 繼續存活
    - 未被配對的現有噴流 → 記錄壽命後移除
    - 未被配對的新位置 → 新噴流誕生

    回傳 (更新後的 jet_id_counter, 本次新結束的噴流紀錄列表)。
    後者供呼叫端立即追加進 jet_lifetime.csv（事件觸發式的增量寫入，
    不需要每次都重寫整份壽命紀錄表）。
    """
    MATCH_DIST = max(5, cfg.NY // 20)   # 預設 12 格（NY=256）

    new_pos    = list(jpos)
    matched_new = set()   # 已配對的新位置（index into new_pos）
    matched_old = set()   # 已配對的舊 jet_id

    # 對每個現有噴流找最近新位置
    for jid, info in jet_tracker.items():
        old_y = info['last_pos']
        best_dist, best_i = float('inf'), -1
        for ni, ny in enumerate(new_pos):
            if ni in matched_new:
                continue
            d = abs(ny - old_y)
            if d < best_dist:
                best_dist, best_i = d, ni
        if best_i >= 0 and best_dist <= MATCH_DIST:
            jet_tracker[jid]['last_pos'] = new_pos[best_i]
            matched_new.add(best_i)
            matched_old.add(jid)

    # 未配對舊噴流：記錄壽命
    newly_dead = []
    for jid in list(jet_tracker.keys()):
        if jid not in matched_old:
            info = jet_tracker.pop(jid)
            record = {
                'jet_id'    : jid,
                'born_step' : info['born'],
                'died_step' : step,
                'duration'  : step - info['born'],
                'mean_y'    : info['last_pos'],
                'ended_by'  : 'disappeared',
            }
            newly_dead.append(record)
    jet_lifetimes.extend(newly_dead)

    # 未配對新位置：誕生新噴流
    for ni, ny in enumerate(new_pos):
        if ni not in matched_new:
            jet_tracker[jet_id_counter] = {'born': step, 'last_pos': ny}
            jet_id_counter += 1

    return jet_id_counter, newly_dead


def _jet_autocorr(u_bar_hist: np.ndarray, steps: list) -> np.ndarray:
    """
    對 Hovmöller 資料 u_bar_hist (T, NY) 的每個緯度做時間自相關，
    回傳 e-folding 自相關時間 tau(y)，單位：模擬步數。
    """
    T, NY = u_bar_hist.shape
    if T < 3:
        return np.zeros(NY)
    dt = (steps[-1] - steps[0]) / max(T - 1, 1)
    tau = np.full(NY, float(steps[-1]))
    n_pad = 2 ** int(np.ceil(np.log2(2 * T)))   # FFT 補零長度

    for y in range(NY):
        sig = u_bar_hist[:, y].astype(np.float64)
        sig -= sig.mean()
        if sig.std() < 1e-12:
            continue
        F   = np.fft.rfft(sig, n=n_pad)
        acf = np.fft.irfft(F * np.conj(F))[:T]
        if acf[0] == 0:
            continue
        acf /= acf[0]
        for lag in range(1, T):
            if acf[lag] < 1.0 / np.e:
                tau[y] = lag * dt
                break
    return tau


def _save_profile_artifacts(prof: dict, steps: list):
    """
    儲存「重量級」剖面衍生輸出（NPZ + 圖表）：
      hovmoller.npz / hovmoller.png    — ū(y, t) Hovmöller 圖
      rs_profile.npz                   — 完整 Reynolds stress 剖面
      qy_profile.npz                   — Q_y 位渦梯度剖面
      pv_profile.npz                   — PV 剖面
      jet_autocorr.png                 — 各緯度自相關時間 vs 渦流翻轉時間

    這些需要重繪圖表、跑 FFT 自相關、輸出整段歷史陣列，成本遠高於逐步
    增量的 CSV 寫入，因此只在 cfg.HEAVY_CHECKPOINT_EVERY 這個低頻率下
    呼叫（見 run_simulation）。NPZ 改用 np.savez（不壓縮）而非
    savez_compressed——這裡的陣列本來就很小（NY×T），壓縮省下的磁碟空間
    可忽略，但壓縮運算本身的 CPU 成本會隨歷史長度線性增加，拖慢 GPU 迴圈。

    ubar_hovmoller.csv / Qy_profile.csv / q_profile.csv / jet_lifetime.csv
    這幾份原始資料改成逐步/逐事件增量追加寫入（見 _append_profile_row /
    _append_jet_lifetime_rows），不在這裡處理。
    """
    if len(steps) == 0:
        return

    steps_arr = np.array(steps, dtype=np.int32)
    y_arr     = np.arange(cfg.NY, dtype=np.float32)

    # ── 轉換成 2D 陣列 ──
    u_bar_hist = np.stack(prof['u_bar'],      axis=0)   # (T, NY)
    rs_hist    = np.stack(prof['RS_profile'], axis=0)
    qy_hist    = np.stack(prof['Qy'],         axis=0)
    q_hist     = np.stack(prof['q_pv'],       axis=0)

    # ── 存 NPZ（不壓縮，降低 CPU 成本） ──
    np.savez(os.path.join(cfg.DATA_DIR, 'hovmoller.npz'),
             u_bar=u_bar_hist, steps=steps_arr, y=y_arr)
    np.savez(os.path.join(cfg.DATA_DIR, 'rs_profile.npz'),
             RS_profile=rs_hist, steps=steps_arr, y=y_arr)
    np.savez(os.path.join(cfg.DATA_DIR, 'qy_profile.npz'),
             Qy=qy_hist, steps=steps_arr, y=y_arr)
    np.savez(os.path.join(cfg.DATA_DIR, 'pv_profile.npz'),
             q_pv=q_hist, steps=steps_arr, y=y_arr)
    print(f"  → 4 個剖面 NPZ 已儲存 ({cfg.DATA_DIR}/)")

    # ── Hovmöller 圖 ──
    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = float(np.percentile(np.abs(u_bar_hist), 98))
    vmax = max(vmax, 1e-6)
    im = ax.pcolormesh(steps_arr, y_arr, u_bar_hist.T,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='nearest')
    plt.colorbar(im, ax=ax, label='ū  (lattice units)')
    ax.set_xlabel("Step")
    ax.set_ylabel("y  (lattice index / latitude)")
    ax.set_title("Hovmöller Diagram  ū(y, t)")
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "hovmoller.png"), dpi=200)
    plt.close(fig)
    print(f"  → hovmoller.png saved.")

    # ── 噴流自相關時間圖 ──
    tau = _jet_autocorr(u_bar_hist, steps)

    # 渦流翻轉時間估算：τ_eddy ≈ L_β / U_rms_mean
    u_rms_mean = float(np.sqrt(np.mean(u_bar_hist**2))) or 1e-9
    L_b_mean   = float(np.sqrt(u_rms_mean / cfg.BETA)) if cfg.BETA > 0 else 0.0
    tau_eddy   = L_b_mean / u_rms_mean if u_rms_mean > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Jet Autocorrelation vs Eddy Turnover Time", fontsize=13)

    axes[0].plot(tau, y_arr, color='steelblue', linewidth=1.2)
    if tau_eddy > 0:
        axes[0].axvline(tau_eddy, color='r', linestyle='--', linewidth=1.0,
                        label=f'τ_eddy ≈ {tau_eddy:.0f} steps')
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Autocorrelation e-folding time (steps)")
    axes[0].set_ylabel("y (lattice)")
    axes[0].set_title("τ_corr(y)")
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # τ_corr / τ_eddy ratio（> 1 → 噴流比渦流翻轉更持久）
    if tau_eddy > 0:
        ratio = tau / tau_eddy
        axes[1].plot(ratio, y_arr, color='darkorange', linewidth=1.2)
        axes[1].axvline(1.0, color='k', linestyle='--', linewidth=0.8, label='ratio = 1')
        axes[1].set_xlabel("τ_corr / τ_eddy")
        axes[1].set_title("Jet Persistence Ratio  (>1 = jet survives eddy turnover)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle='--', alpha=0.4)
        axes[1].set_ylabel("y (lattice)")

    plt.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "jet_autocorr.png"), dpi=200)
    plt.close(fig)
    print(f"  → jet_autocorr.png saved.")


if __name__ == "__main__":
    run_simulation()