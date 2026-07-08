"""
fetch_jupiter_winds.py
──────────────────────
下載木星「緯向風速剖面」(zonal wind: u vs latitude) 存成參考檔
    data/reference/jupiter_zonal_wind.csv   欄位：lat_deg, u_ms
供 compare_jupiter.py 與模擬結果疊圖對比。預設鎖定 Cassini（Porco et al.
2003 / Cassini ISS CB2）剖面。

⚠️ NASA 沒有提供「單一乾淨、可直接程式下載」的 CB2 緯向風剖面檔；資料多半
散在 PDS 影像產品或論文附錄中。因此本工具的邏輯是：
  1) 若指定了 --url（或環境變數 LBM_JUPITER_WIND_URL），就下載並解析。
     解析器容許常見格式：每列兩欄（緯度、風速），空白或逗號分隔，'#' 為註解。
     可用 --lat-col / --u-col 指定欄位索引，--u-unit 指定風速單位（m/s 或 km/s）。
  2) 若沒有可用網址或下載失敗，印出下列官方/論文來源連結，請手動下載後
     用 --file 指到本機檔案（同一個解析器）轉存成標準 CSV。

用法：
  python3 analysis/fetch_jupiter_winds.py --url <URL>
  python3 analysis/fetch_jupiter_winds.py --file <本機檔> --lat-col 0 --u-col 1 --u-unit m/s
  LBM_JUPITER_WIND_URL=<URL> python3 analysis/fetch_jupiter_winds.py
"""
import argparse
import csv
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Code/
REF_DIR = os.path.join(ROOT, 'data', 'reference')
OUT_PATH = os.path.join(REF_DIR, 'jupiter_zonal_wind.csv')

# 手動下載來源（自動下載失敗時印給使用者）
MANUAL_SOURCES = [
    ("NASA PDS Atmospheres Node（Cassini ISS 影像 / 衍生風場產品）",
     "https://pds-atmospheres.nmsu.edu/"),
    ("Porco et al. 2003, Science 299, 1541（Cassini 木星大氣，含 CB2 緯向風剖面）",
     "https://www.science.org/doi/10.1126/science.1079462"),
    ("Asay-Davis et al. 2011, Icarus（1979–2008 緯向風變化，含數位化剖面）",
     "https://doi.org/10.1016/j.icarus.2010.11.018"),
]


def parse_profile(text: str, lat_col: int, u_col: int, u_unit: str):
    """把任意「每列含緯度與風速」的文字表解析成 [(lat_deg, u_ms), ...]。"""
    scale = 1000.0 if u_unit.lower() in ('km/s', 'kms', 'km') else 1.0
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in '#;/':
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) <= max(lat_col, u_col):
            continue
        try:
            lat = float(parts[lat_col])
            u   = float(parts[u_col]) * scale
        except ValueError:
            continue          # 標題列或非數值列，略過
        if -90.0 <= lat <= 90.0:
            rows.append((lat, u))
    rows.sort(key=lambda r: r[0])
    return rows


def save_csv(rows, out_path=OUT_PATH):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['lat_deg', 'u_ms'])
        w.writerows([[f"{lat:.4f}", f"{u:.3f}"] for lat, u in rows])
    print(f"✅ 已存 {len(rows)} 筆 → {out_path}")


def print_manual_help():
    print("\n❌ 沒有可用網址或下載/解析失敗。請手動取得 Cassini 緯向風剖面：")
    for name, url in MANUAL_SOURCES:
        print(f"   • {name}\n     {url}")
    print("\n下載成兩欄文字表（緯度、風速）後，執行：")
    print("   python3 analysis/fetch_jupiter_winds.py --file <你的檔> "
          "--lat-col 0 --u-col 1 --u-unit m/s")
    print(f"就會轉存成標準格式 {OUT_PATH}（lat_deg, u_ms）。")


def main():
    ap = argparse.ArgumentParser(description="下載/轉存木星緯向風剖面參考檔")
    ap.add_argument('--url', default=os.environ.get('LBM_JUPITER_WIND_URL', ''),
                    help='資料來源 URL（或設環境變數 LBM_JUPITER_WIND_URL）')
    ap.add_argument('--file', default='', help='改用本機已下載的檔案（跳過網路）')
    ap.add_argument('--lat-col', type=int, default=0, help='緯度欄索引（預設 0）')
    ap.add_argument('--u-col',  type=int, default=1, help='風速欄索引（預設 1）')
    ap.add_argument('--u-unit', default='m/s', help='原始風速單位：m/s 或 km/s（預設 m/s）')
    ap.add_argument('--out', default=OUT_PATH, help=f'輸出路徑（預設 {OUT_PATH}）')
    args = ap.parse_args()

    text = None
    try:
        if args.file:
            with open(args.file, encoding='utf-8', errors='replace') as fh:
                text = fh.read()
        elif args.url:
            print(f"⬇️  下載 {args.url} ...")
            with urllib.request.urlopen(args.url, timeout=30) as resp:
                text = resp.read().decode('utf-8', errors='replace')
    except Exception as e:                       # noqa: BLE001 — 任何失敗都退回手動指引
        print(f"⚠️  取得資料失敗：{e}")

    if not text:
        print_manual_help()
        sys.exit(1)

    rows = parse_profile(text, args.lat_col, args.u_col, args.u_unit)
    if not rows:
        print("⚠️  解析不到有效的 (緯度, 風速) 資料列。")
        print_manual_help()
        sys.exit(1)
    save_csv(rows, args.out)


if __name__ == '__main__':
    main()
