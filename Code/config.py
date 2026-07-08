#外掛程式
import os
import numpy as np
import taichi as ti

#並行參數掃描：允許以環境變數覆寫特定參數（由 sweep.py 為每個子行程設定），
#未設定時沿用下方預設值，因此單獨執行 main.py 的行為完全不變。
def _env_override(name, default, cast=float):
    v = os.environ.get(f"LBM_{name}")
    return cast(v) if v is not None else default

#隨機種子：Taichi 預設 random_seed=0，若不指定，sweep.py 平行掃描的每個子
#行程都會用同一顆種子，AR(1) 噪音與 collision.py 的 ti.randn() 擾動在各組
#之間完全相同，等於少了一個獨立變因。sweep.py 會替每個 run 注入不同的
#LBM_RANDOM_SEED；單獨執行 main.py 時沒有設定，沿用 0（行為不變）。
RANDOM_SEED = _env_override('RANDOM_SEED', 0, cast=int)

#後端選擇
ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=RANDOM_SEED)

#網格基本參數
NX        = 1024           # x 方向格點數（緯向）
NY        = 512           # y 方向格點數（徑向 / 緯度方向）
MAX_STEPS = _env_override('MAX_STEPS', 3000000, int)        # 總演化步數
SAVE_EVERY = 5000         # 每幾步存一幀
SAVE_SPECTRUM = 4*SAVE_EVERY    # 每幾步存數據

#D2Q9離散速度
CX_NP  = np.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=np.int32)
CY_NP  = np.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=np.int32)
W_NP   = np.array(
    [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
    dtype=np.float32
)
OPP_NP = np.array([ 0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

#Taichi
CX  = ti.field(ti.i32,  shape=9)
CY  = ti.field(ti.i32,  shape=9)
W   = ti.field(ti.f32,  shape=9)
OPP = ti.field(ti.i32,  shape=9)

#參數初始化，將陣列上傳至Taichi
def init_constants():
    CX.from_numpy(CX_NP)
    CY.from_numpy(CY_NP)
    W.from_numpy(W_NP)
    OPP.from_numpy(OPP_NP)

#物理參數
#β-plane模型科氏力參數
F0 = 0                #科氏力參數基準值（根據模擬緯度調整）
# β 以「基準解析度」定義，任意 NY 自動換算：
#   不變量 = 極區 f_cor 邊界值 = F0 + BETA·(NY/2)
#   要它與解析度無關，須 BETA·NY = const → BETA = BETA_REF · NY_REF / NY
# 這樣顯式科氏力每步注入的能量(∝f_cor²)不隨解析度暴增，避免發散。
NY_REF   = 256           #參數校正時的基準解析度
BETA_REF = _env_override('BETA_REF', 5e-5)          #基準解析度下的每格梯度
BETA = BETA_REF * NY_REF / NY   #依實際 NY 動態換算（任意解析度皆可）

#Reighly drag
EPSILON = _env_override('EPSILON', 7e-5)           #摩擦係數

#海綿層參數
SPONGE_FRAC = 0.15        #海綿層厚度占總高度的比例
EPSILON_MAX = _env_override('EPSILON_MAX', 1.5e-4)       #海綿層最大阻尼（外側）

# AR參數
Tc = _env_override('TC', 400.0)                    #時間尺度
alpha = np.exp(-1.0 / Tc)     #自相關係數
SIGMA_2D    = _env_override('SIGMA_2D',    1.5e-6)      #2D 各向同性強迫振幅
SIGMA_ZONAL = _env_override('SIGMA_ZONAL', 3e-6)        #1D 緯向 AR(1) 強迫振幅
WARMUP_STEPS = 100000

# AR強迫的空間相關結構：對每格獨立的白噪音做高斯低通濾波，讓等向能量譜
# E(k) ∝ k·exp(-σ²k²)（k 為 analysis/spectrum.py 定義的波數單位，殼層模態數
# ∝k 的因子已含在內）出現峰值於 k_f，而非未濾波（delta-correlated）時能量
# 集中在 Nyquist 附近、強迫直接打在耗散尺度上。
# 峰值位置 k_f = 1/(σ√2) → σ = 1/(k_f·√2)（格點單位）。
K_F = _env_override('K_F', 1.2)                         #強迫目標波數（同 spectrum.py 的 k 單位）
FORCE_SIGMA  = 1.0 / (K_F * np.sqrt(2.0))               #對應高斯濾波標準差（格點）
FORCE_RADIUS = max(2, int(np.ceil(3.0 * FORCE_SIGMA)))  #濾波核半徑（涵蓋 ~3σ）
_force_r  = np.arange(-FORCE_RADIUS, FORCE_RADIUS + 1, dtype=np.float64)
_force_w  = np.exp(-_force_r**2 / (2.0 * FORCE_SIGMA**2))
_force_w /= _force_w.sum()
FORCE_KERNEL = _force_w.astype(np.float32)              #1D 高斯核（分離式，x/y 各套用一次）
FORCE_NORM   = float(1.0 / np.sum(_force_w**2))         #還原兩次卷積（x、y）造成的方差衰減

#MRT矩陣參數
s1, s2, s4, s6 = 1.2, 1.2, 1.8, 1.8

#LBM流體參數
U_MAX   = 0.07                 # 特徵速度
NU      = 0.0003               # 運動黏滯係數 (調大以符合 k^-3 斜率)
TAU     = 3.0 * NU + 0.5       # 鬆弛時間（現已被MRT取代，但保留以供參考）
OMEGA   = float(1.0 / TAU)     # 單純BGK鬆弛頻率（現已被MRT取代）

#輸出目錄（sweep.py 平行掃描時，各子行程各自覆寫成獨立資料夾，避免互相覆蓋）
OUTPUT_DIR    = os.environ.get('LBM_OUTPUT_DIR', "output")
DATA_DIR      = os.environ.get('LBM_DATA_DIR', "data/processed")

#噴流偵測與 PV 梯度計算的平滑參數
# ū(y) 在偵測噴流峰值 / 計算 Q_y 二階微分前，先做空間滑動平均，
# 避免小尺度渦流雜訊被誤判為噴流，或被二階微分放大成假的變號次數。
JET_SMOOTH_WINDOW = 25      #滑動平均視窗（格點，建議 5~10，需為奇數）
JET_THRESHOLD     = 1.3    #噴流偵測門檻（normalized std 單位，原 0.8 提高避免雜訊）

#檢查點頻率
# 純量 CSV / ubar_hovmoller.csv / jet_lifetime.csv 都是「逐步/逐事件增量
# 追加」，成本極低，維持高頻率沒問題。但 summary.png、hovmoller.png、
# jet_autocorr.png、NPZ 等重量級輸出需要重繪圖表、跑 FFT 自相關、壓縮
# 整段歷史陣列，成本隨模擬進度線性(甚至更快)增長，若跟輕量 CSV 用同一個
# 高頻率存檔會嚴重拖慢 GPU 迴圈、抵銷平行運算的效能。因此分開兩種頻率：
PROFILE_SNAPSHOT_EVERY = 50000                  #Qy_profile.csv / q_profile.csv 快照間隔（步數）
HEAVY_CHECKPOINT_EVERY = SAVE_SPECTRUM * 10     #重量級圖表 + NPZ 的存檔間隔（步數）