"""
scan_summary.py
────────────────
彙整 sweep.py 產生的多組參數掃描結果：對每組 run_XXX/log.csv 只看「後段」
（預設最後 50% 的列，避免 warm-up 階段拉低平均），算出 7 個關鍵統計量的
mean/std，加上穩定性警告次數，連同該組參數彙整成一列，輸出成 scan_summary.csv
（一組一列），方便跨組比較誰更貼近木星的 zonostrophic 特徵。

參數來源：output/sweep/manifest.csv（sweep.py 執行後自動產生，含每個
run_id 的 data_dir 與掃描參數）。若找不到 manifest.csv，改成直接掃描
data/processed/sweep/ 底下的 run_* 資料夾，此時輸出不含參數欄位。

「epsilon_diverge_count」定義：後段中 u_max > 0.3（main.py 的穩定性警告
門檻，見 main.py 內 `u_max > 0.3` 那行）的列數，次數而非布林值。

用法：
    python3 analysis/scan_summary.py
    python3 analysis/scan_summary.py --tail-frac 0.5 --out data/processed/sweep/scan_summary.csv
"""
import argparse
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Code/

# 後段要彙整的 7 個統計量（對應使用者要看的 R_β*, zonal_frac, jet_count,
# E_slope, omega_skew, Qy_sign_changes, staircase_score）
STAT_COLUMNS = ['R_beta_star', 'zonal_frac', 'jet_count',
                'E_slope', 'omega_skew', 'Qy_sign_changes', 'staircase_score']

DIVERGE_COL = 'u_max'
DIVERGE_THRESHOLD = 0.3  # 對齊 main.py 的穩定性警告門檻

# manifest.csv 裡不屬於掃描參數的欄位（sweep.py write_manifest 固定寫入）
MANIFEST_NON_PARAM_COLS = {'run_id', 'status', 'elapsed_sec', 'output_dir', 'data_dir'}

# NY 固定 512x256（config.py 未提供環境變數覆寫 NX/NY），BETA_REF 即為實際
# BETA（BETA = BETA_REF * NY_REF/NY，NY_REF=NY=256 時兩者相等）。NU 目前
# sweep.py 未掃描，固定沿用 config.py 的預設值。
NU_FIXED = 0.0003


def _read_log_csv(log_path: str) -> dict:
    """讀 log.csv，回傳 {欄名: numpy array}，數值欄轉 float。"""
    with open(log_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return {}
    cols = {}
    for name in rows[0].keys():
        if name == 'jet_positions':
            continue
        try:
            cols[name] = np.array([float(r[name]) for r in rows], dtype=np.float64)
        except ValueError:
            pass
    return cols


def _tail_slice(cols: dict, tail_frac: float) -> dict:
    """取每個欄位陣列最後 tail_frac 比例的元素（依 log.csv 原始列序，等同依 step 排序後的後段）。"""
    if not cols:
        return cols
    n = len(next(iter(cols.values())))
    start = int(np.floor(n * (1 - tail_frac)))
    return {k: v[start:] for k, v in cols.items()}


def summarize_run(log_path: str, tail_frac: float) -> dict:
    cols = _read_log_csv(log_path)
    n_total = len(next(iter(cols.values()))) if cols else 0
    tail = _tail_slice(cols, tail_frac)
    n_tail = len(next(iter(tail.values()))) if tail else 0

    row = {}
    for name in STAT_COLUMNS:
        if name in tail and n_tail > 0:
            row[f'{name}_mean'] = tail[name].mean()
            row[f'{name}_std'] = tail[name].std()
        else:
            row[f'{name}_mean'] = ''
            row[f'{name}_std'] = ''

    if DIVERGE_COL in tail and n_tail > 0:
        row['epsilon_diverge_count'] = int(np.count_nonzero(tail[DIVERGE_COL] > DIVERGE_THRESHOLD))
    else:
        row['epsilon_diverge_count'] = ''

    row['n_rows_tail'] = n_tail
    row['n_rows_total'] = n_total
    return row


def load_manifest(manifest_path: str) -> dict:
    """回傳 {run_id: {'data_dir':..., 'status':..., 掃描參數...}}；找不到就回傳空 dict。"""
    if not os.path.isfile(manifest_path):
        return {}
    records = {}
    with open(manifest_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            params = {k: v for k, v in rec.items() if k not in MANIFEST_NON_PARAM_COLS}
            records[rec['run_id']] = {
                'data_dir': rec['data_dir'],
                'status': rec['status'],
                **params,
            }
    return records


def build_param_row(manifest_rec: dict) -> dict:
    """把 manifest 的掃描參數整理成使用者關心的欄位（BETA / EPSILON / EPSILON_MAX / SIGMA / NU）。"""
    row = {}
    if 'BETA_REF' in manifest_rec:
        row['BETA'] = manifest_rec['BETA_REF']
    for key in ('EPSILON', 'EPSILON_MAX', 'SIGMA'):
        if key in manifest_rec:
            row[key] = manifest_rec[key]
    row['NU'] = NU_FIXED
    row['status'] = manifest_rec.get('status', '')
    return row


def main():
    parser = argparse.ArgumentParser(description="彙整參數掃描各組後段統計量成 scan_summary.csv")
    parser.add_argument('--manifest', default=os.path.join(ROOT, 'output', 'sweep', 'manifest.csv'),
                         help='sweep.py 產生的 manifest.csv 路徑')
    parser.add_argument('--sweep-data-root', default=os.path.join(ROOT, 'data', 'processed', 'sweep'),
                         help='找不到 manifest.csv 時，退回掃描此目錄底下的 run_* 資料夾')
    parser.add_argument('--out', default=None, help='輸出 csv 路徑，預設存在 sweep-data-root/scan_summary.csv')
    parser.add_argument('--tail-frac', type=float, default=0.5,
                         help='每組 log.csv 取最後幾成的列數當「後段」，預設 0.5＝後 50%%')
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)

    if manifest:
        run_ids = sorted(manifest.keys())
    elif os.path.isdir(args.sweep_data_root):
        run_ids = sorted(d for d in os.listdir(args.sweep_data_root)
                          if os.path.isdir(os.path.join(args.sweep_data_root, d)))
    else:
        print(f"❌ 找不到 {args.manifest}，也找不到 {args.sweep_data_root}")
        return

    rows = []
    for run_id in run_ids:
        if run_id in manifest:
            data_dir = manifest[run_id]['data_dir']
        else:
            data_dir = os.path.join(args.sweep_data_root, run_id)
        log_path = os.path.join(data_dir, 'log.csv')
        if not os.path.isfile(log_path):
            print(f"⚠️  跳過 {run_id}：找不到 {log_path}")
            continue

        row = {'run_id': run_id}
        if run_id in manifest:
            row.update(build_param_row(manifest[run_id]))
        row.update(summarize_run(log_path, args.tail_frac))
        rows.append(row)
        print(f"✓ {run_id}  ({row['n_rows_tail']}/{row['n_rows_total']} rows 後段)")

    if not rows:
        print("沒有任何一組成功彙整。")
        return

    out_path = args.out or os.path.join(args.sweep_data_root, 'scan_summary.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n📋 scan_summary.csv 已儲存：{out_path}  ({len(rows)} 組)")


if __name__ == '__main__':
    main()
