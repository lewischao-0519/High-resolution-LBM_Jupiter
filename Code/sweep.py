"""
sweep.py
────────
平行參數掃描：在同一張顯卡上，用多個獨立行程（各自 ti.init 一次）
同時跑多組參數組合，個別存進各自的 output/sweep/run_XXX、
data/processed/sweep/run_XXX 資料夾，資料夾內部格式與單次執行 main.py
完全相同，只是外層多一層 run 分類。

Taichi 的 GPU context 與 field 是行程內全域、載入時就依 NX/NY 建立好，
無法在同一個 Python 行程裡切換第二組參數，因此平行掃描一律採「多行程」：
每個 run 各自啟動一份 main.py 子行程，用環境變數覆寫 config.py 的參數與
輸出路徑，多個子行程共用同一張 GPU、由驅動排程執行。

用法：
  python3 sweep.py                    # 依下方 GRID 展開所有組合，預設同時跑 4 組
  python3 sweep.py --max-parallel 6   # 依顯卡記憶體調整同時執行數量
"""

import os, sys, csv, time, itertools, subprocess, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from analysis.scan_summary import generate_summary

# stdout 若被導向檔案（而非真正終端機），Windows 會退回用系統 locale 編碼，
# 無法編碼本檔案 print() 用到的 Emoji 而 crash，強制用 UTF-8。
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 參數網格：在此定義要掃描的組合，key 對應 config.py 的 LBM_<key> 環境變數覆寫。
#    某個參數只放一個值 = 該參數固定不掃描，只是為了記錄在 manifest 裡。
GRID = {
    'EPSILON_MAX': [1.5e-4],   # 海綿層最大阻尼
    'BETA_REF':    [7e-5],         # 科氏力梯度（基準解析度）
    'EPSILON':     [7e-5],         # Rayleigh drag 摩擦係數
    'SIGMA':       [2e-4,3e-4,4e-4],                   # AR(1) 噪音振幅
}

SWEEP_OUTPUT_ROOT = os.path.join(ROOT, "output", "sweep")
SWEEP_DATA_ROOT   = os.path.join(ROOT, "data", "processed", "sweep")
MANIFEST_PATH     = os.path.join(SWEEP_OUTPUT_ROOT, "manifest.csv")


def build_runs():
    """依 GRID 的笛卡兒積展開成一組組參數，依序編號 run_001, run_002, ..."""
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    return [(f"run_{i:03d}", dict(zip(keys, combo)))
            for i, combo in enumerate(combos, start=1)]


def run_one(run_id: str, params: dict) -> dict:
    """啟動一個子行程執行 main.py，該組參數與輸出路徑透過環境變數注入。"""
    out_dir  = os.path.join(SWEEP_OUTPUT_ROOT, run_id)
    data_dir = os.path.join(SWEEP_DATA_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    env = os.environ.copy()
    env['LBM_OUTPUT_DIR'] = out_dir
    env['LBM_DATA_DIR']   = data_dir
    for k, v in params.items():
        env[f'LBM_{k}'] = str(v)
    # 每個 run 給不同的隨機種子：main.py 透過 config.py 呼叫 ti.init() 時，
    # Taichi 若不指定 random_seed 一律用 0，平行掃描的每個子行程會用同一顆
    # 種子，AR(1) 噪音與 collision.py 的 ti.randn() 擾動在各組之間會完全相
    # 同。用 run_id 的編號（run_007 → 7）當種子，同一顆種子只在同一個
    # run_id 重跑時才會重現，跨組之間彼此不同。
    env['LBM_RANDOM_SEED'] = run_id.split('_')[-1]
    # main.py 的 print() 大量使用 Emoji；子行程 stdout 一旦被導向檔案（而非
    # 真正的終端機），Windows 會退回用系統 locale 編碼（例如 cp950），無法
    # 編碼 Emoji 而 crash。強制用 UTF-8，避免這個跟參數本身無關的錯誤。
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8']       = '1'

    log_path = os.path.join(out_dir, "run.log")
    print(f"▶ 啟動 {run_id}  {params}")
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as logf:
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "main.py")],
            cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    status  = 'ok' if result.returncode == 0 else f'failed({result.returncode})'
    mark    = '✅' if result.returncode == 0 else '❌'
    print(f"{mark} {run_id} 完成，狀態={status}，耗時={elapsed/60:.1f} 分鐘")

    return {'run_id': run_id, 'status': status, 'elapsed_sec': f"{elapsed:.1f}",
            'output_dir': out_dir, 'data_dir': data_dir, **params}


def write_manifest(records: list, keys: list) -> str:
    """寫 manifest.csv；遇到權限錯誤（例如 OneDrive/防毒軟體暫時鎖檔）重試幾
    次，還是失敗就改存成備用檔名，絕不讓已經跑完的模擬結果因為這一步而遺
    失。回傳實際寫入的路徑，供後續 generate_summary() 使用。"""
    os.makedirs(SWEEP_OUTPUT_ROOT, exist_ok=True)
    columns = ['run_id', 'status', 'elapsed_sec', 'output_dir', 'data_dir'] + keys

    def _try_write(path):
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)

    for attempt in range(5):
        try:
            _try_write(MANIFEST_PATH)
            print(f"\n📋 掃描總表已儲存：{MANIFEST_PATH}")
            return MANIFEST_PATH
        except PermissionError as e:
            print(f"⚠️  寫入 {MANIFEST_PATH} 被拒絕（第 {attempt+1} 次）：{e}，1 秒後重試...")
            time.sleep(1)

    fallback_path = os.path.join(SWEEP_OUTPUT_ROOT, f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    print(f"❌ 重試 5 次仍失敗，改存成備用檔名：{fallback_path}")
    _try_write(fallback_path)
    print(f"📋 掃描總表已儲存：{fallback_path}")
    return fallback_path


def main():
    parser = argparse.ArgumentParser(description="Jupiter LBM 平行參數掃描")
    parser.add_argument("--max-parallel", type=int, default=4,
                         help="同時執行的 GPU 行程數（預設 4，依顯卡記憶體調整）")
    args = parser.parse_args()

    runs = build_runs()
    keys = list(GRID.keys())
    print(f"🪐 共 {len(runs)} 組參數組合，最多同時執行 {args.max_parallel} 組\n")
    for run_id, params in runs:
        print(f"  {run_id}: {params}")
    print()

    records = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(run_one, run_id, params): run_id
                   for run_id, params in runs}
        for fut in as_completed(futures):
            records.append(fut.result())

    records.sort(key=lambda r: r['run_id'])
    manifest_path = write_manifest(records, keys)

    print("\n📊 自動彙整 scan_summary.csv ...")
    generate_summary(manifest_path, SWEEP_DATA_ROOT)


if __name__ == "__main__":
    main()
