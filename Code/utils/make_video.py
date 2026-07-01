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

import os, sys, glob, subprocess, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(ROOT, "output", "frames")
OUT_DIR    = os.path.join(ROOT, "output")


def check_ffmpeg():
    result = subprocess.run(["ffmpeg", "-version"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ ffmpeg 未安裝。請執行：brew install ffmpeg")
        sys.exit(1)
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
    # ffmpeg 用 glob 輸入（-pattern_type glob）
    # -vf scale 讓寬/高都是偶數（H.264 需求）
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",          # 品質 0–51，越小越好
        "-preset", "slow",
        out_path
    ]
    print(f"\n🎬 合成 {prefix} → {output_name}  ({len(frames)} frames @ {fps} fps)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"   ✅ 已存：{out_path}  ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ 失敗：{result.stderr[-500:]}")


def main():
    parser = argparse.ArgumentParser(description="Assemble LBM frames into MP4 videos")
    parser.add_argument("--type", choices=["vel", "vort", "zonal", "spectrum", "all"],
                        default="all", help="要合成的影片類型（預設 all）")
    parser.add_argument("--fps",  type=int, default=24,
                        help="影片幀率（預設 24）")
    parser.add_argument("--fps-sparse", type=int, default=8,
                        help="低頻快照（zonal/spectrum）的幀率（預設 8）")
    args = parser.parse_args()

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

    print("\n🪐 完成！影片儲存於 output/")


if __name__ == "__main__":
    main()
