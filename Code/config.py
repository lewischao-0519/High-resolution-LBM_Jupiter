# config.py  ── Jupiter LBM 主設定檔
# 對應 PDF §2 研究目的、§3 控制方程、§7 參數掃描設計
#
import numpy as np
import taichi as ti

# --- 後端選擇 ---
ti.init(arch=ti.gpu, default_fp=ti.f32)

# ══════════════════════════════════════════════════════
#  網格基本參數
# ══════════════════════════════════════════════════════
NX        = 512          # x 方向格點數（緯向）
NY        = 256          # y 方向格點數（徑向 / 緯度方向）
MAX_STEPS = 300000         # 總演化步數
SAVE_EVERY = 5000          # 每幾步存一幀

# ══════════════════════════════════════════════════════
#  D2Q9 離散速度
# ════════════════════════════════════════════════════
CX_NP  = np.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=np.int32)
CY_NP  = np.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=np.int32)
W_NP   = np.array(
    [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
    dtype=np.float32
)
OPP_NP = np.array([ 0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# --- Taichi fields（GPU 常駐）---
CX  = ti.field(ti.i32,  shape=9)
CY  = ti.field(ti.i32,  shape=9)
W   = ti.field(ti.f32,  shape=9)
OPP = ti.field(ti.i32,  shape=9)

def init_constants():
    """將 NumPy 陣列上傳至 Taichi fields"""
    CX.from_numpy(CX_NP)
    CY.from_numpy(CY_NP)
    W.from_numpy(W_NP)
    OPP.from_numpy(OPP_NP)

# ══════════════════════════════════════════════════════
#  物理參數
# ══════════════════════════════════════════════════════
# --- β-plane Coriolis 參數 ---
F0 = 0           # 科氏参数基值 (格单位)
BETA = 5.0e-4         # 科氏梯度 (格单位)
# AR(1) 參數
Tc = 10.0                     # 关联时间步数
alpha = np.exp(-1.0 / Tc)     # ≈ 0.9048
sigma = 5.0e-4                # 振幅（可调）
# --- LBM 流體參數 ---
U_MAX   = 0.1            # 特徵速度（格單位）
NU      = 0.005         # 運動黏滯係數 ν（掃描範圍 0.001–0.01）
TAU     = 3.0 * NU + 0.5 # 鬆弛時間 τ（掃描範圍 0.6–1.2）
OMEGA   = float(1.0 / TAU)
# ══════════════════════════════════════════════════════
#  輸出目錄
# ══════════════════════════════════════════════════════
OUTPUT_DIR    = "output"
DATA_DIR      = "data/processed"