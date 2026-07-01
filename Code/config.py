#外掛程式
import numpy as np
import taichi as ti

#後端選擇
ti.init(arch=ti.gpu, default_fp=ti.f32)

#網格基本參數
NX        = 512           # x 方向格點數（緯向）
NY        = 256           # y 方向格點數（徑向 / 緯度方向）
MAX_STEPS = 700000        # 總演化步數
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
F0 = 2e-4                #科氏力參數基準值（根據模擬緯度調整）
# β 以「基準解析度」定義，任意 NY 自動換算：
#   不變量 = 極區 f_cor 邊界值 = F0 + BETA·(NY/2)
#   要它與解析度無關，須 BETA·NY = const → BETA = BETA_REF · NY_REF / NY
# 這樣顯式科氏力每步注入的能量(∝f_cor²)不隨解析度暴增，避免發散。
NY_REF   = 256           #參數校正時的基準解析度
BETA_REF = 8e-5          #基準解析度下的每格梯度
BETA = BETA_REF * NY_REF / NY   #依實際 NY 動態換算（任意解析度皆可）

#Reighly drag
EPSILON = 3e-5           #摩擦係數

#海綿層參數
SPONGE_FRAC = 0.15        #海綿層厚度占總高度的比例
EPSILON_MAX = 7e-5       #海綿層最大阻尼（外側）

# AR參數
Tc = 400.0                    #時間尺度
alpha = np.exp(-1.0 / Tc)     #自相關係數
sigma = 1e-6                  #振幅
WARMUP_STEPS = 100000 

#MRT矩陣參數
s1, s2, s4, s6 = 1.2, 1.2, 1.8, 1.8

#LBM流體參數
U_MAX   = 0.07                 # 特徵速度
NU      = 0.0002               # 運動黏滯係數 (調大以符合 k^-3 斜率)
TAU     = 3.0 * NU + 0.5       # 鬆弛時間（現已被MRT取代，但保留以供參考）
OMEGA   = float(1.0 / TAU)     # 單純BGK鬆弛頻率（現已被MRT取代）

#輸出目錄
OUTPUT_DIR    = "output"
DATA_DIR      = "data/processed"