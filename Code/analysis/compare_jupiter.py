"""
compare_jupiter.py
──────────────────
把某個 run 的「緯向平均風剖面」ū(y)（取後段穩態的時間平均）換算成
實際物理量（風速 m/s、緯度 deg），與 NASA 參考剖面
（data/reference/jupiter_zonal_wind.csv，由 fetch_jupiter_winds.py 產生）
疊圖對比，輸出 PNG 與換算後的模擬剖面 CSV。

緯度對映（β-plane 切平面近似）：
    lat_deg(y) = degrees( (y − NY/2) · dx / R_eq )
風速換算：u_ms = u_sim · dx / dt（見 physical_units.py，dt 由該 run 的
BETA_REF 經角速度換算）。

用法：
  python3 analysis/compare_jupiter.py --data-dir data/processed/sweep/run_001 --beta-ref 7e-5
  python3 analysis/compare_jupiter.py --data-dir <單次執行的 DATA_DIR>
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # 讓獨立執行時找得到 analysis 套件
import analysis.physical_units as pu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Code/
DEFAULT_REF = os.path.join(ROOT, 'data', 'reference', 'jupiter_zonal_wind.csv')
MIN_STEP_DEFAULT = 150000     # 與 scan_summary.py 一致：丟暖身/暫態


def load_ubar_tail(hov_path: str, tail_frac: float, min_step: float):
    """讀 ubar_hovmoller.csv，丟 step<min_step 的暖身列，取後段做時間平均。
    回傳 ū(y) [格/步]，長度 = NY。"""
    with open(hov_path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    steps = np.array([float(r[0]) for r in rows])
    data  = np.array([[float(v) for v in r[1:]] for r in rows])   # (T, NY)
    keep = steps >= min_step
    if keep.sum() >= 5:
        data = data[keep]
    start = int(np.floor(len(data) * (1 - tail_frac)))
    return data[start:].mean(axis=0)                              # (NY,)


def load_reference(ref_path: str):
    lat, u = [], []
    with open(ref_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lat.append(float(row['lat_deg']))
            u.append(float(row['u_ms']))
    return np.array(lat), np.array(u)


def main():
    ap = argparse.ArgumentParser(description="模擬緯向風剖面 vs NASA 參考剖面")
    ap.add_argument('--data-dir', required=True,
                    help='含 ubar_hovmoller.csv 的資料夾（run 的 DATA_DIR）')
    ap.add_argument('--reference', default=DEFAULT_REF,
                    help=f'NASA 參考剖面 CSV（預設 {DEFAULT_REF}）')
    ap.add_argument('--beta-ref', type=float, default=None,
                    help='該 run 的 BETA_REF（決定 dt；預設用 physical_units 的退回值）')
    ap.add_argument('--tail-frac', type=float, default=0.5)
    ap.add_argument('--min-step', type=float, default=MIN_STEP_DEFAULT)
    ap.add_argument('--nx', type=int, default=pu.NX_DEFAULT)
    ap.add_argument('--out', default=None, help='輸出 PNG（預設存在 data-dir 內）')
    args = ap.parse_args()

    hov_path = os.path.join(args.data_dir, 'ubar_hovmoller.csv')
    if not os.path.isfile(hov_path):
        raise SystemExit(f"❌ 找不到 {hov_path}")

    beta_ref = args.beta_ref if args.beta_ref is not None else 5e-5

    ubar = load_ubar_tail(hov_path, args.tail_frac, args.min_step)   # (NY,) 格/步
    ny = len(ubar)
    dx = pu.grid_dx(args.nx)                                          # m/格

    y = np.arange(ny)
    lat_deg = np.degrees((y - ny / 2.0) * dx / pu.JUP_R_EQ_M)
    u_ms = pu.velocity_ms(ubar, beta_ref, nx=args.nx, ny=ny)         # m/s

    # 存換算後的模擬剖面
    prof_csv = os.path.join(args.data_dir, 'ubar_profile_physical.csv')
    with open(prof_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['lat_deg', 'u_ms'])
        w.writerows([[f"{a:.4f}", f"{b:.3f}"] for a, b in zip(lat_deg, u_ms)])
    print(f"✅ 模擬剖面（物理量）已存：{prof_csv}")

    # 疊圖
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(u_ms, lat_deg, color='firebrick', lw=1.6,
            label=f'LBM (β_ref={beta_ref:g})')
    if os.path.isfile(args.reference):
        ref_lat, ref_u = load_reference(args.reference)
        ax.plot(ref_u, ref_lat, color='navy', lw=1.2, alpha=0.8,
                label='Jupiter (Cassini)')
    else:
        print(f"⚠️  找不到參考剖面 {args.reference}（先執行 fetch_jupiter_winds.py）；"
              f"只畫模擬結果。")
    ax.axvline(0, color='grey', lw=0.6)
    ax.set_xlabel('zonal wind ū  [m/s]')
    ax.set_ylabel('latitude  [deg]')
    ax.set_title('Zonal-mean wind: LBM vs Jupiter')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    out = args.out or os.path.join(args.data_dir, 'compare_jupiter.png')
    fig.tight_layout(); fig.savefig(out, dpi=200)
    print(f"✅ 對比圖已存：{out}")


if __name__ == '__main__':
    main()
