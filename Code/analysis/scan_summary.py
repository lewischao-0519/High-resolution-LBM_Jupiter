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

「omega_skew_front」/「omega_skew_back」：與上面「後段」window 無關，固定
取整條 log.csv（依總列數）20%~60% 當 front、60%~100% 當 back 各自的
omega_skew 平均值，用來看渦度偏態是前段就已經定型還是後段才轉向。

用法：
    python3 analysis/scan_summary.py
    python3 analysis/scan_summary.py --tail-frac 0.5 --out data/processed/sweep/scan_summary.csv

也可以被 sweep.py 直接 import 呼叫 generate_summary()，掃描跑完後自動彙整。
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # 讓獨立執行時找得到 analysis 套件
import analysis.physical_units as pu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Code/

# 無 manifest（直接掃 run_* 目錄）時，換算物理量所需 BETA_REF 的退回值，
# 對齊 config.py 的 BETA_REF 預設。有 manifest 時一律用該 run 實際的 BETA_REF。
BETA_REF_DEFAULT = 5e-5

# 後段要彙整的 7 個統計量（對應使用者要看的 R_β*, zonal_frac, jet_count,
# E_slope, omega_skew, Qy_sign_changes, staircase_score）
STAT_COLUMNS = ['R_beta_star', 'zonal_frac', 'jet_count',
                'E_slope', 'omega_skew', 'Qy_sign_changes', 'staircase_score']

DIVERGE_COL = 'u_max'
DIVERGE_THRESHOLD = 0.3  # 對齊 main.py 的穩定性警告門檻

# 統計前先丟掉 step < MIN_STEP 的資料列，避免暖身/初始暫態污染平均值。
# config.py 的 WARMUP_STEPS=100000（前 10 萬步不注入 AR 強迫），再留一段讓
# 系統走到統計穩態，故預設門檻設在 150000。此為「依 step 過濾」，比單純取
# 後段列數（--tail-frac）更穩健：短 run 也不會把暖身段算進去。可用 --min-step
# 覆寫。若某 run 過濾後剩不到 MIN_ROWS_AFTER_FLOOR 列，退回不過濾並印警告。
MIN_STEP_DEFAULT = 150000
MIN_ROWS_AFTER_FLOOR = 5

# omega_skew_front / omega_skew_back：固定依「總步數」切，跟 --tail-frac 無關
FRONT_BACK_COL = 'omega_skew'
FRONT_START_FRAC, FRONT_END_FRAC, BACK_END_FRAC = 0.2, 0.6, 1.0

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


def _apply_min_step(cols: dict, min_step: float, run_id: str = '') -> dict:
    """丟掉 step < min_step 的列（暖身/初始暫態）。若過濾後剩太少列，退回不過濾。"""
    if not cols or min_step <= 0 or 'step' not in cols:
        return cols
    mask = cols['step'] >= min_step
    if int(mask.sum()) < MIN_ROWS_AFTER_FLOOR:
        tag = f"{run_id}：" if run_id else ""
        print(f"⚠️  {tag}step≥{int(min_step)} 只剩 {int(mask.sum())} 列，"
              f"退回不套用 min_step（改由 --tail-frac 取後段）。")
        return cols
    return {k: v[mask] for k, v in cols.items()}


def _tail_slice(cols: dict, tail_frac: float) -> dict:
    """取每個欄位陣列最後 tail_frac 比例的元素（依 log.csv 原始列序，等同依 step 排序後的後段）。"""
    if not cols:
        return cols
    n = len(next(iter(cols.values())))
    start = int(np.floor(n * (1 - tail_frac)))
    return {k: v[start:] for k, v in cols.items()}


def summarize_run(log_path: str, tail_frac: float,
                  min_step: float = MIN_STEP_DEFAULT, run_id: str = '') -> dict:
    cols = _read_log_csv(log_path)
    cols = _apply_min_step(cols, min_step, run_id)   # 先丟暖身段，再取後段
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

    if FRONT_BACK_COL in cols and n_total > 0:
        front_lo = int(np.floor(n_total * FRONT_START_FRAC))
        front_hi = int(np.floor(n_total * FRONT_END_FRAC))
        back_hi  = int(np.floor(n_total * BACK_END_FRAC))
        front = cols[FRONT_BACK_COL][front_lo:front_hi]
        back  = cols[FRONT_BACK_COL][front_hi:back_hi]
        row['omega_skew_front'] = front.mean() if len(front) > 0 else ''
        row['omega_skew_back']  = back.mean()  if len(back)  > 0 else ''
    else:
        row['omega_skew_front'] = ''
        row['omega_skew_back']  = ''

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
    """把 manifest 的掃描參數整理成使用者關心的欄位（BETA / EPSILON / EPSILON_MAX / SIGMA / K_F / NU）。"""
    row = {}
    if 'BETA_REF' in manifest_rec:
        row['BETA'] = manifest_rec['BETA_REF']
    for key in ('EPSILON', 'EPSILON_MAX', 'SIGMA_2D', 'SIGMA_ZONAL', 'K_F', 'TC'):
        if key in manifest_rec:
            row[key] = manifest_rec[key]
    row['NU'] = NU_FIXED
    row['status'] = manifest_rec.get('status', '')
    return row


def generate_summary(manifest_path: str, sweep_data_root: str,
                      out_path: str = None, tail_frac: float = 0.5,
                      min_step: float = MIN_STEP_DEFAULT) -> list:
    """讀 manifest（或退回掃描 sweep_data_root），彙整每組後段統計量並寫出
    scan_summary.csv。回傳寫入的 rows（list[dict]），沒有任何一組成功則回傳
    空 list 且不寫檔。CLI 的 main() 與 sweep.py 都呼叫這個函式。
    min_step：統計前先丟掉 step < min_step 的暖身/暫態列（見 MIN_STEP_DEFAULT）。"""
    manifest = load_manifest(manifest_path)

    if manifest:
        run_ids = sorted(manifest.keys())
    elif os.path.isdir(sweep_data_root):
        run_ids = sorted(d for d in os.listdir(sweep_data_root)
                          if os.path.isdir(os.path.join(sweep_data_root, d)))
    else:
        print(f"❌ 找不到 {manifest_path}，也找不到 {sweep_data_root}")
        return []

    rows = []
    for run_id in run_ids:
        if run_id in manifest:
            data_dir = manifest[run_id]['data_dir']
        else:
            data_dir = os.path.join(sweep_data_root, run_id)
        log_path = os.path.join(data_dir, 'log.csv')
        if not os.path.isfile(log_path):
            print(f"⚠️  跳過 {run_id}：找不到 {log_path}")
            continue

        row = {'run_id': run_id}
        if run_id in manifest:
            row.update(build_param_row(manifest[run_id]))
        row.update(summarize_run(log_path, tail_frac, min_step, run_id))
        rows.append(row)
        print(f"✓ {run_id}  ({row['n_rows_tail']}/{row['n_rows_total']} rows 後段)")

    if not rows:
        print("沒有任何一組成功彙整。")
        return []

    out_path = out_path or os.path.join(sweep_data_root, 'scan_summary.csv')
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
    return rows


# ──────────────────────────────────────────────────────────────────────
# 物理量版本：把後段穩態的無因次/格點量換算成木星實際單位（見 physical_units.py）
# ──────────────────────────────────────────────────────────────────────
def _tail_mean(tail: dict, name: str, n_tail: int):
    return float(tail[name].mean()) if (name in tail and n_tail > 0) else None


def summarize_run_physical(log_path: str, tail_frac: float, min_step: float,
                           beta_ref: float, run_id: str = '') -> dict:
    """對單一 run 的後段穩態量做物理換算。beta_ref 決定該 run 的 dt。"""
    cols = _read_log_csv(log_path)
    cols = _apply_min_step(cols, min_step, run_id)
    n_total = len(next(iter(cols.values()))) if cols else 0
    tail = _tail_slice(cols, tail_frac)
    n_tail = len(next(iter(tail.values()))) if tail else 0

    conv = pu.conversion_factors(beta_ref)
    row = {
        'dx_km'           : round(conv['dx_km'], 3),
        'dt_s'            : round(conv['dt_s'], 4),
        'channel_width_km': round(conv['channel_width_km'], 1),
    }
    row['sim_time_days'] = (round(pu.time_days(float(cols['step'].max()), beta_ref), 2)
                            if ('step' in cols and n_total > 0) else '')

    u_rms = _tail_mean(tail, 'u_rms', n_tail)
    u_max = _tail_mean(tail, 'u_max', n_tail)
    L_b   = _tail_mean(tail, 'L_beta', n_tail)
    jc    = _tail_mean(tail, 'jet_count', n_tail)

    row['u_rms_ms']     = round(pu.velocity_ms(u_rms, beta_ref), 2) if u_rms is not None else ''
    row['u_rms_ms_std'] = (round(pu.velocity_ms(float(tail['u_rms'].std()), beta_ref), 2)
                           if ('u_rms' in tail and n_tail > 0) else '')
    row['u_max_ms']     = round(pu.velocity_ms(u_max, beta_ref), 2) if u_max is not None else ''
    row['L_beta_km']    = round(pu.length_km(L_b), 1) if L_b is not None else ''
    row['jet_count']    = round(jc, 2) if jc is not None else ''
    # 噴流平均間距：徑向通道寬 / 噴流數（可直接對比木星帶狀流帶寬）
    row['jet_spacing_km'] = (round(conv['channel_width_km'] / jc, 1)
                             if (jc is not None and jc > 0) else '')

    # 無因次量：直接抄後段平均，方便與物理量並列閱讀
    for name in ('E_slope', 'R_beta_star', 'zonal_frac'):
        v = _tail_mean(tail, name, n_tail)
        row[name] = round(v, 4) if v is not None else ''

    row['n_rows_tail'] = n_tail
    return row


def generate_physical_summary(manifest_path: str, sweep_data_root: str,
                              out_path: str = None, tail_frac: float = 0.5,
                              min_step: float = MIN_STEP_DEFAULT) -> list:
    """產生 scan_summary_physical.csv：每組後段穩態量換算成木星實際物理單位
    （速度 m/s、長度 km、時間天、噴流間距 km），可直接與 NASA 觀測對比。"""
    manifest = load_manifest(manifest_path)
    if manifest:
        run_ids = sorted(manifest.keys())
    elif os.path.isdir(sweep_data_root):
        run_ids = sorted(d for d in os.listdir(sweep_data_root)
                          if os.path.isdir(os.path.join(sweep_data_root, d)))
    else:
        print(f"❌ 找不到 {manifest_path}，也找不到 {sweep_data_root}")
        return []

    rows = []
    for run_id in run_ids:
        if run_id in manifest:
            data_dir = manifest[run_id]['data_dir']
            beta_ref = float(manifest[run_id].get('BETA_REF', BETA_REF_DEFAULT))
        else:
            data_dir = os.path.join(sweep_data_root, run_id)
            beta_ref = BETA_REF_DEFAULT
        log_path = os.path.join(data_dir, 'log.csv')
        if not os.path.isfile(log_path):
            print(f"⚠️  跳過 {run_id}：找不到 {log_path}")
            continue

        row = {'run_id': run_id}
        if run_id in manifest:
            row.update(build_param_row(manifest[run_id]))
        row.update(summarize_run_physical(log_path, tail_frac, min_step, beta_ref, run_id))
        rows.append(row)

    if not rows:
        print("沒有任何一組成功彙整（物理量版）。")
        return []

    out_path = out_path or os.path.join(sweep_data_root, 'scan_summary_physical.csv')
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

    print(f"📋 scan_summary_physical.csv 已儲存：{out_path}  ({len(rows)} 組)")
    return rows


def main():
    parser = argparse.ArgumentParser(description="彙整參數掃描各組後段統計量成 scan_summary.csv")
    parser.add_argument('--manifest', default=os.path.join(ROOT, 'output', 'sweep', 'manifest.csv'),
                         help='sweep.py 產生的 manifest.csv 路徑')
    parser.add_argument('--sweep-data-root', default=os.path.join(ROOT, 'data', 'processed', 'sweep'),
                         help='找不到 manifest.csv 時，退回掃描此目錄底下的 run_* 資料夾')
    parser.add_argument('--out', default=None, help='輸出 csv 路徑，預設存在 sweep-data-root/scan_summary.csv')
    parser.add_argument('--tail-frac', type=float, default=0.5,
                         help='每組 log.csv 取最後幾成的列數當「後段」，預設 0.5＝後 50%%')
    parser.add_argument('--min-step', type=float, default=MIN_STEP_DEFAULT,
                         help=f'統計前先丟掉 step < 此值的暖身/暫態列，預設 {MIN_STEP_DEFAULT}')
    parser.add_argument('--physical', action='store_true',
                         help='額外輸出 scan_summary_physical.csv（換算成木星實際物理單位）')
    args = parser.parse_args()
    generate_summary(args.manifest, args.sweep_data_root, args.out,
                     args.tail_frac, args.min_step)
    if args.physical:
        generate_physical_summary(args.manifest, args.sweep_data_root,
                                  None, args.tail_frac, args.min_step)


if __name__ == '__main__':
    main()
