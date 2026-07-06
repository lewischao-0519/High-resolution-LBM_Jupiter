"""
utils/make_video.py
───────────────────
把 output/frames/ 下的快照合成為 MP4 影片。
預設輸出：
  output/vel_evolution.mp4    速度量值
  output/vort_evolution.mp4   渦度
  output/zonal_evolution.mp4  帶狀平均剖面（每 20000 步）
  output/spectrum_evolution.mp4 能量譜（每 20000 步）

用法：
  python3 utils/make_video.py             # 全部合成
  python3 utils/make_video.py --type vel  # 只合成速度場
  python3 utils/make_video.py --fps 30    # 自訂幀率
"""

import os, sys, glob, subprocess, argparse, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 預設路徑（獨立執行時沿用舊行為）；平行掃描時由 main.py 以 --frames-dir /
# --out-dir 指向各自的輸出資料夾，避免多組結果互相覆蓋。
FRAMES_DIR = os.path.join(ROOT, "output", "frames")
OUT_DIR    = os.path.join(ROOT, "output")

# ffmpeg 執行檔路徑（check_ffmpeg 解析後填入，make_video 沿用）
FFMPEG = "ffmpeg"


def _find_ffmpeg():
    """回傳可用的 ffmpeg 執行檔路徑；找不到回傳 None。
    優先用系統 PATH 上的 ffmpeg，否則退回 imageio-ffmpeg 內建的執行檔
    （pip install imageio-ffmpeg 即可，Windows 免手動配 PATH）。"""
    for exe in ("ffmpeg",):
        try:
            r = subprocess.run([exe, "-version"], capture_output=True, text=True)
            if r.returncode == 0:
                return exe
        except FileNotFoundError:
            pass
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def check_ffmpeg():
    global FFMPEG
    exe = _find_ffmpeg()
    if exe is None:
        print("❌ 找不到 ffmpeg。最簡單：pip install imageio-ffmpeg（免配 PATH）。\n"
              "   或 macOS: brew install ffmpeg；"
              "Windows: https://ffmpeg.org/download.html 下載後將 bin 加入 PATH。")
        sys.exit(1)
    FFMPEG = exe
    result = subprocess.run([exe, "-version"], capture_output=True, text=True)
    ver = result.stdout.splitlines()[0]
    print(f"✅ {ver}")


def make_video(prefix: str, ext: str, fps: int, output_name: str):
    """
    用 ffmpeg glob 模式合成指定前綴的所有 frames。
    prefix : 'vel' | 'vort' | 'zonal' | 'spectrum'
    ext    : 'jpg' | 'png'
    """
    # 確認有無 frames
    pattern = os.path.join(FRAMES_DIR, f"{prefix}_*.{ext}")
    frames  = sorted(glob.glob(pattern))
    if not frames:
        print(f"  ⚠️  找不到 {prefix}_*.{ext}，跳過。")
        return

    out_path = os.path.join(OUT_DIR, output_name)
    # 用 concat demuxer（跨平台，Windows 也支援；-pattern_type glob 僅限 Unix）
    # 寫一份 frame 清單到暫存檔，交給 ffmpeg。
    list_fd, list_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(list_fd, "w", encoding="utf-8") as f:
            for fp in frames:
                # concat 格式需正斜線且路徑跳脫單引號
                safe = fp.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")
                f.write(f"duration {1.0 / fps}\n")
            # concat demuxer 需重複最後一幀才會顯示其 duration
            last = frames[-1].replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{last}'\n")

        # -vf scale 讓寬/高都是偶數（H.264 需求）
        cmd = [
            FFMPEG, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-crf", "18",          # 品質 0–51，越小越好
            "-preset", "slow",
            out_path
        ]
        print(f"\n🎬 合成 {prefix} → {output_name}  ({len(frames)} frames @ {fps} fps)")
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        os.remove(list_path)

    if result.returncode == 0:
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"   ✅ 已存：{out_path}  ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ 失敗：{result.stderr[-500:]}")


def main():
    global FRAMES_DIR, OUT_DIR

    parser = argparse.ArgumentParser(description="Assemble LBM frames into MP4 videos")
    parser.add_argument("--type", choices=["vel", "vort", "zonal", "spectrum", "all"],
                        default="all", help="要合成的影片類型（預設 all）")
    parser.add_argument("--fps",  type=int, default=24,
                        help="影片幀率（預設 24）")
    parser.add_argument("--fps-sparse", type=int, default=8,
                        help="低頻快照（zonal/spectrum）的幀率（預設 8）")
    parser.add_argument("--frames-dir", default=FRAMES_DIR,
                        help="快照來源資料夾（預設 output/frames，平行掃描時由呼叫端指定各自路徑）")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="影片輸出資料夾（預設 output，平行掃描時由呼叫端指定各自路徑）")
    args = parser.parse_args()

    FRAMES_DIR = args.frames_dir
    OUT_DIR    = args.out_dir

    check_ffmpeg()
    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = {
        "vel"     : ("vel",      "jpg", args.fps,        "vel_evolution.mp4"),
        "vort"    : ("vort",     "jpg", args.fps,        "vort_evolution.mp4"),
        "zonal"   : ("zonal",    "png", args.fps_sparse, "zonal_evolution.mp4"),
        "spectrum": ("spectrum", "png", args.fps_sparse, "spectrum_evolution.mp4"),
    }

    targets = list(jobs.keys()) if args.type == "all" else [args.type]
    for t in targets:
        make_video(*jobs[t])

    print(f"\n🪐 完成！影片儲存於 {OUT_DIR}/")


if __name__ == "__main__":
    main()
