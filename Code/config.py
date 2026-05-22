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
NX        = 1024          # x 方向格點數（緯向）
NY        = 512           # y 方向格點數（徑向 / 緯度方向）
MAX_STEPS = 50000         # 總演化步數
SAVE_EVERY = 200          # 每幾步存一幀

# ══════════════════════════════════════════════════════
#  D2Q9 離散速度（NumPy 版供初始化用）
# ══════════════════════════════════════════════════════
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
#  物理參數（對應 PDF §3 控制方程 & §7 參數掃描）
# ══════════════════════════════════════════════════════

# --- LBM 流體參數 ---
U_MAX   = 0.05            # 特徵速度（格單位）
NU      = 1e-3            # 運動黏滯係數 ν（掃描範圍 0.001–0.01）
TAU     = 3.0 * NU + 0.5 # 鬆弛時間 τ（掃描範圍 0.6–1.2）
OMEGA   = float(1.0 / TAU)

# --- Coriolis / β-plane（對應 PDF §3, Eq.2: f(y)=f0+βy）---
F0      = 1e-4            # 基礎 Coriolis 參數 f0
BETA    = 1e-4            # β 梯度（掃描範圍 1e-5–1e-3）
# y 方向中央為赤道（y=NY//2），往極地遞增 Coriolis
# 實際 f(y) 在 forcing.py 的 kernel 裡計算

# --- 熱力參數（對應 PDF §5, Eq.4-5: T(y)=T0-ΔT*y^2）---
T0          = 1.0         # 參考溫度（格單位）
DELTA_T     = 0.05        # 溫差強度 ΔT（掃描範圍 0.01–0.1）
ALPHA_T     = 0.001       # 熱膨脹係數 α（熱力加速度強度）

# --- 隨機噪音（對應 PDF §7, ε）---
NOISE_AMP   = 1e-3        # 噪音強度 ε（掃描範圍 1e-4–1e-2）

# ══════════════════════════════════════════════════════
#  輸出目錄
# ══════════════════════════════════════════════════════
OUTPUT_DIR    = "output"
DATA_DIR      = "data/processed"
