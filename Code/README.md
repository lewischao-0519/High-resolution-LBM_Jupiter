# Jupiter LBM — 旋轉流體帶狀風系模擬

使用格子波茲曼法（Lattice Boltzmann Method, LBM）在 GPU（Taichi）上模擬旋轉流體，
觀察 β-plane 科氏力效應下帶狀噴流（zonal jet）的自發形成，對應木星大氣的條紋狀風系。

## 快速開始

```bash
pip install -r requirements.txt
python main.py
```

執行後會在 `output/` 產生逐步快照，模擬結束時自動輸出：
- `output/summary.png`（3×3 診斷總覽圖）
- `output/hovmoller.png`（Hovmöller 時空圖）
- `output/jet_autocorr.png`（噴流持續時間分析）
- 4 支 mp4 影片
- `data/processed/` 下的個別診斷 CSV 與剖面 NPZ 檔

---

## 程式架構

```
Code/
├── main.py              ← 主程式入口：初始化 → 模擬迴圈 → 存檔 → 合成影片
├── config.py             ← 全域參數（網格大小、物理常數、D2Q9 常數、輸出路徑）
│
├── core/
│   ├── lattice.py        ← D2Q9 平衡分布函數 f^eq、MRT 轉換矩陣 M / M⁻¹
│   ├── collision.py       ← 所有 GPU 場（f, rho, ux, uy, Fx, Fy）定義 + MRT 碰撞核心
│   │                        （已將串流、科氏力、海綿層阻尼、噪音注入融合在同一核函數）
│   ├── streaming.py       ← 獨立的週期邊界串流核心（保留參考，實際未使用，已被
│   │                        collision.py 中的 pull-scheme 取代）
│   └── forcing.py         ← 科氏力旋轉、AR(1) 緯向噪音場更新（部分函數保留備用）
│
├── analysis/
│   ├── vorticity.py       ← 渦度場 ω_z = ∂uy/∂x − ∂ux/∂y（中央差分）
│   ├── spectrum.py        ← 2D FFT 等向能量譜 E(k)、Kolmogorov 斜率擬合
│   ├── zonal_mean.py      ← 緯向平均風速剖面、噴流數量統計
│   └── diagnostics.py     ← 延伸診斷量（見「診斷量一覽」）
│
├── utils/
│   ├── fft.py             ← 通用 FFT 工具（環形平均、波數網格）
│   ├── plotting.py        ← 視覺化（速度場、渦度、帶狀剖面、能量譜）
│   └── make_video.py      ← 用 ffmpeg 把 output/frames/ 的快照合成 mp4
│
├── data/
│   └── processed/
│       ├── log.csv            ← 所有診斷純量的彙整表（每存檔步一列）
│       ├── u_rms.csv          ┐
│       ├── u_max.csv          │
│       ├── jet_count.csv      │
│       ├── E_slope.csv        │  各診斷量的獨立 CSV（每步即時寫入，
│       ├── L_beta.csv         │  模擬中斷也不損失已記錄資料）
│       ├── Ro_beta.csv        │
│       ├── jet_positions.csv  │
│       ├── KE_zonal.csv       │
│       ├── KE_eddy.csv        │
│       ├── zonal_frac.csv     │
│       ├── RS_mean.csv        │
│       ├── omega_rms.csv      │
│       ├── omega_skew.csv     │
│       ├── Qy_sign_changes.csv│
│       ├── staircase_score.csv│
│       ├── R_beta_star.csv    ┘
│       ├── hovmoller.npz      ← ū(y,t) 二維場（Hovmöller 資料）
│       ├── rs_profile.npz     ← ⟨u′v′⟩(y,t) 完整剖面歷史
│       ├── qy_profile.npz     ← Q_y(y,t) 位渦梯度剖面歷史
│       └── pv_profile.npz     ← q(y,t) PV 剖面歷史
│
└── output/
    ├── frames/                ← 模擬過程中的逐步快照（模擬結束後自動清除）
    ├── summary.png            ← 3×3 診斷總結圖（9 個純量時序面板）
    ├── hovmoller.png          ← ū(y,t) Hovmöller 圖（y 為緯度、x 為時間、色為ū）
    ├── jet_autocorr.png       ← 各緯度自相關時間 τ_corr(y) 及與渦流翻轉時間的比值
    ├── vel_evolution.mp4      ← 速度量值場動畫
    ├── vort_evolution.mp4     ← 渦度場動畫
    ├── zonal_evolution.mp4   ← 帶狀平均風速剖面動畫
    └── spectrum_evolution.mp4 ← 能量譜動畫
```

## 資料流

```
config.py（網格、物理參數初始化）
   │
   ▼
core/collision.py  ── 定義所有 GPU 場：f, f_new, rho, ux, uy, Fx, Fy, noise_zonal
   │
   ├── core/forcing.py::update_zonal_noise()  ── AR(1) 更新緯向噪音場
   │
   ▼
core/collision.py::mrt_collision_kernel()
   ├─ 1. Pull-scheme 串流 + 南北牆半程反彈
   ├─ 2. 計算巨觀量 rho, ux, uy
   ├─ 3. 海綿層阻尼係數 + 科氏力 + 噪音 → 合成外力 Fx, Fy
   ├─ 4. Guo forcing 半步速度修正
   └─ 5. MRT 矩量空間碰撞 + 反轉換回分布函數
            │
            ▼（交換讀寫緩衝 src ↔ dst）
   analysis/（vorticity, spectrum, zonal_mean, diagnostics）
            │
            ▼（每 SAVE_EVERY 步）
   ├─ 16 個純量 → 各自 CSV（即時 append）
   └─ 4 個剖面（u_bar, RS_profile, Qy, q_pv）→ 暫存記憶體
            │
            ▼（模擬結束）
   main.py::_save_summary() → output/summary.png, data/processed/log.csv
   main.py::_save_profiles() → output/hovmoller.png, output/jet_autocorr.png
                                data/processed/*.npz
   main.py::_make_videos()  → utils/make_video.py → output/*.mp4
```

## 邊界條件

| 方向 | 條件 | 實作位置 |
|------|------|----------|
| x（緯向 / zonal） | 週期邊界 | `collision.py` pull-scheme 中 `% NX` |
| y（緯度 / meridional） | 半程反彈牆（no-penetration）+ 海綿層阻尼 | `collision.py` 中 `py < 0 or py >= NY` 反射，及 `SPONGE_FRAC` 阻尼帶 |

---

## 診斷量一覽

### 純量時序（每步存 CSV + `log.csv`）

| 欄位 | 物理意義 | 函式 |
|------|----------|------|
| `u_rms` | 全場速度 RMS | — |
| `u_max` | 全場最大速度（穩定性監控，> 0.3 警告） | — |
| `jet_count` | 帶狀噴流數量 | `zonal_mean::count_jet_streams` |
| `E_slope` | 能量譜斜率 α（E∼k^α，理論值 -5/3 或 -3） | `spectrum::kolmogorov_slope` |
| `L_beta` | Rhines 尺度（格點） | `diagnostics::rhines_scale` |
| `Ro_beta` | β-Rossby 數 | `diagnostics::rossby_beta` |
| `jet_positions` | 各噴流 y-index 列表（`\|` 分隔） | `diagnostics::jet_positions` |
| `KE_zonal` | 帶狀動能 | `diagnostics::kinetic_energy_decomp` |
| `KE_eddy` | 渦流動能 | 同上 |
| `zonal_frac` | KE_zonal / KE_total（目標 ~60–80%） | 同上 |
| `RS_mean` | Reynolds stress ⟨u′v′⟩ 空間平均 | `diagnostics::reynolds_stress` |
| `omega_rms` | 渦度 RMS | `diagnostics::vorticity_stats` |
| `omega_skew` | 渦度偏態（負值 → 反氣旋主導，木星特徵） | 同上 |
| `Qy_sign_changes` | Q_y 符號反轉次數（> 0 → 正壓不穩定） | `diagnostics::pv_gradient` |
| `staircase_score` | PV 階梯分數（>> 1 → 尖銳輸送屏障） | `diagnostics::pv_staircase` |
| `R_beta_star` | Zonostrophy index（> 2.5 → 穩定噴流 regime） | `diagnostics::zonostrophy_index` |

### 剖面歷史（存 NPZ，形狀 (T, NY)）

| 檔案 | 內容 | 對應圖 |
|------|------|--------|
| `hovmoller.npz` | `u_bar(y, t)`：緯向平均東風剖面隨時間演化 | `hovmoller.png` |
| `rs_profile.npz` | `⟨u′v′⟩(y, t)`：Reynolds stress 完整 y 剖面 | — |
| `qy_profile.npz` | `Q_y(y, t) = β − ∂²ū/∂y²`：位渦梯度剖面 | — |
| `pv_profile.npz` | `q(y, t) = β·y − dū/dy`：PV 剖面 | — |

NPZ 讀取範例：
```python
import numpy as np
data = np.load('data/processed/hovmoller.npz')
u_bar = data['u_bar']   # shape (T, NY)
steps = data['steps']   # shape (T,)
y     = data['y']       # shape (NY,)
```

---

## 專有名詞解釋

### 數值方法

**LBM（Lattice Boltzmann Method，格子波茲曼法）**
一種以粒子分布函數 `f_i(x, t)` 描述流體的數值方法，取代傳統直接求解 Navier–Stokes
方程式。流體在每個時間步分成「碰撞（collision）」與「串流（streaming）」兩階段演化，
巨觀流體量（密度 ρ、速度 u）由分布函數的統計矩求得。適合 GPU 平行運算。

**D2Q9**
二維 LBM 常用的離散速度模型：每個格點有 9 個離散速度方向（1 個靜止 + 4 個軸向 + 4 個
對角），對應 `config.py` 中的 `CX_NP` / `CY_NP`（速度分量）與 `W_NP`（各方向權重）。

**平衡分布函數 f^eq（equilibrium distribution）**
假設流體處於局部熱平衡時，分布函數應趨近的目標值，是 Maxwell–Boltzmann 分布的低速
展開近似。對應 `core/lattice.py::feq_single()`。碰撞步驟本質上就是讓 `f` 朝
`f^eq` 鬆弛。

**BGK（Bhatnagar–Gross–Krook）碰撞模型**
最簡單的 LBM 碰撞算子，用單一鬆弛時間 τ 讓分布函數線性趨近平衡態：
`f_new = f - (f - f^eq) / τ`。`config.py` 中的 `TAU`、`OMEGA = 1/TAU` 是為此模型
保留的參數，目前實際碰撞已由更穩定的 MRT 取代。

**MRT（Multiple-Relaxation-Time，多重鬆弛時間模型）**
BGK 的進階版本：先用轉換矩陣 `M` 把分布函數投影到「矩空間（moment space）」
（密度、動量、應力等物理量各自獨立的空間），對每個矩量施加不同的鬆弛係數
（`s1, s2, s4, s6, omega_shear` 等），再用 `M⁻¹` 轉換回分布函數空間。比 BGK
更穩定、數值耗散更可控，是本程式實際使用的碰撞模型（`core/collision.py::mrt_collision_kernel`）。

**Pull-scheme 串流**
串流步驟的一種實作方式：對每個格點「主動去抓取」上一時刻鄰居格點傳來的分布函數值
（`f_src[i, y - CY[i], x - CX[i]]`），而非「主動推送」給鄰居（push-scheme）。
Pull-scheme 的優點是可以直接寫入目標格點、避免 GPU 平行寫入衝突。

**半程反彈（Half-way bounce-back）**
一種無滑移固壁邊界條件：分布函數碰到邊界時，沿原路徑反彈回入射方向的相反分量
（`f_local[i] = f_src[OPP[i], y, x]`），模擬流體在牆面被彈回、法向速度為零的物理現象。
本程式用於南北（y 方向）邊界。

**Guo forcing scheme**
一種將外力（如科氏力、摩擦力）正確導入 LBM 的方法（Guo, Zheng & Shi, 2002）。
若直接把外力當作額外速度硬加上去會有二階誤差，Guo's scheme 改為在碰撞前用
「半步速度」`u + F·dt/(2ρ)` 計算平衡矩，並在矩空間額外疊加一個外力矩修正項
（`core/collision.py::forcing_moments`），確保外力對動量方程的貢獻是二階精確的。

### 旋轉流體力學 / 木星氣候動力學

**科氏力（Coriolis force）**
旋轉參考系中，流體因慣性偏轉所感受到的視外力，是行星尺度大氣環流（信風、噴流、
颱風）形成的根本原因。程式中以科氏參數 `f_cor` 表示，並施加垂直於速度方向的力
（`Fx_cor = ρ·f_cor·vy`, `Fy_cor = -ρ·f_cor·vx`）。

**β-plane 近似**
把球面上隨緯度變化的科氏參數 `f = 2Ω sin(φ)` 在局部區域線性化為
`f(y) = f₀ + β·y`，其中 `f₀` 是參考緯度的科氏參數，`β = df/dy` 是其緯向梯度。
這是研究行星尺度帶狀流最常用的簡化模型。對應 `config.py` 的 `F0`、`BETA`。

**Rhines 尺度（Rhines scale）L_β**
`L_β = √(U_rms / β)`，是行星波（Rossby 波）與湍流渦漩能量相當時的特徵長度尺度，
理論上帶狀噴流的間距 ≈ `2π·L_β`。這個尺度解釋了為什麼旋轉湍流會自發組織成
一條條規則的緯向風帶，而不是保持隨機渦漩。對應 `analysis/diagnostics.py::rhines_scale()`。

**β-Rossby 數 Ro_β**
無因次數，衡量慣性力與 β 效應（行星渦度梯度）的相對強弱：
`Ro_β = U_rms / (β·L²)`。當 `Ro_β ~ O(1)` 附近時，系統最容易形成帶狀噴流。
對應 `analysis/diagnostics.py::rossby_beta()`。

**帶狀流 / 噴流（Zonal jet）**
沿緯度方向（東西向）延伸、南北方向速度快速變化的窄長高速氣流帶，是木星表面
可見條紋的成因，也是本模擬要重現的核心現象。程式透過緯向平均風速剖面
`⟨u_x⟩_x(y)` 的局部極值來偵測噴流位置與數量。

**Hovmöller 圖（Hovmöller diagram）**
以 y 軸（緯度）為空間維度、x 軸為時間、顏色代表 ū(y,t) 的二維時空圖，是追蹤
「噴流生成 → 漂移 → 合併 → 消失」動力學過程最直接的工具。本程式在模擬結束後輸出
`output/hovmoller.png`，原始資料存在 `data/processed/hovmoller.npz`。

**噴流自相關時間 τ_corr**
對固定緯度 y 的 ū(y, t) 時序做 FFT 自相關，找到訊號衰減至 e⁻¹ 所需的時間。
與渦流翻轉時間（eddy turnover time）τ_eddy ≈ L_β / U_rms 的比值 τ_corr / τ_eddy
直接量化噴流是否「站穩」：比值 >> 1 表示噴流持續時間遠超一個渦流翻轉週期，
可認為具有統計意義的穩定性。對應 `main.py::_jet_autocorr()`，輸出 `output/jet_autocorr.png`。

**Rayleigh–Kuo 正壓不穩定判據**
位渦梯度 `Q_y(y) = β − ∂²ū/∂y²` 在 y 方向變號是正壓不穩定的必要條件。若剖面中有
n 處符號反轉，表示流場中有 n 個潛在的不穩定區，帶狀噴流容易被剪切力撕裂。
本程式追蹤 `Qy_sign_changes(t)` 並保存完整 `Q_y(y, t)` 剖面於 `qy_profile.npz`。
對應 `analysis/diagnostics.py::pv_gradient()`。

**PV staircase（位渦階梯結構）**
穩定帶狀噴流對應的位渦分布 `q(y) = β·y − dū/dy` 應呈現階梯狀——「平台（混合均勻區）+
陡峭邊界（輸送屏障）」，陡峭的 PV 跳躍代表強韌的渦流輸送屏障。
本程式用 `staircase_score = max|dq/dy| / mean|dq/dy|` 量化：接近 1 表示平滑漸變（屏障弱），
>> 5 表示真正的 PV 階梯結構。對應 `analysis/diagnostics.py::pv_staircase()`。

**Zonostrophy index R_β***
判斷模擬是否落在「zonostrophic turbulence」（穩定帶狀流）regime 的無因次指標：

```
R_β* = L_R / L_t
  L_R = √(2·U_rms / β)      ← Rhines 尺度
  L_t = (ε / β³)^{1/5}      ← 渦流翻轉時間 = Rossby 波週期的過渡尺度
  ε   = EPSILON · KE_total   ← 穩態耗散率（= 能量注入率）
```

判斷依據（Sukoriansky et al. 2007）：
- `R_β* ≳ 2.5` → zonostrophic regime，帶狀流穩定主導
- `R_β* ≲ 1.5` → 摩擦主導，噴流無法持續形成

對應 `analysis/diagnostics.py::zonostrophy_index()`，每步結果存入 `R_beta_star.csv`。

**Reynolds stress（雷諾應力）⟨u′v′⟩**
擾動速度分量的協方差 `⟨(u - ū)(v - v̄)⟩_x`，代表渦漩對平均流的動量輸送。
其 y 梯度的輻合 / 輻散決定噴流的加速或減速。本程式保存完整 y 剖面（`rs_profile.npz`），
而非僅空間平均純量（`RS_mean`），以便觀察噴流被剪切力破壞前的輻合結構瓦解過程。

**帶狀動能 / 渦流動能分解（KE_zonal / KE_eddy）**
把總動能分解為「緯向平均流動能」（`KE_zonal = 0.5⟨ū² + v̄²⟩`）與「相對於平均流
的渦漩動能」（`KE_eddy = KE_total - KE_zonal`）。當逆能量串聯（inverse energy
cascade）有效時，`KE_zonal / KE_total` 會顯著升高（木星觀測約 60–80%）。
對應 `analysis/diagnostics.py::kinetic_energy_decomp()`。

**逆能量串聯（Inverse energy cascade）**
二維湍流的特有現象：能量傾向從小尺度渦漩向大尺度結構匯聚（與三維湍流的能量
正向串聯相反），是行星大氣中小尺度擾動能自發組織成大尺度帶狀噴流的物理機制。

**能量譜 E(k)**
速度場依波數 k（空間尺度的倒數）分解後的能量密度分布，透過 2D FFT 計算並做
環形平均（等向性假設）得到。對應 `analysis/spectrum.py::compute_energy_spectrum()`。

**Kolmogorov 斜率 α**
在對數座標下擬合 `E(k) ~ k^α` 得到的斜率。三維各向同性湍流理論值為 `α = -5/3`；
二維湍流逆能量串聯區域理論值可達 `α ≈ -3`。對應 `analysis/spectrum.py::kolmogorov_slope()`。

**海綿層（Sponge layer）**
在計算域南北邊界附近額外施加的強阻尼區，用來吸收向邊界傳播的波動、避免非物理
的反射污染內部流場。對應 `config.py` 的 `SPONGE_FRAC`（厚度佔比）、`EPSILON_MAX`（邊界最大阻尼）。

**Rayleigh drag（線性摩擦阻尼）**
正比於局部速度、方向相反的阻尼力 `F_damp = -ε·ρ·u`，用來代表次網格尺度的
摩擦耗散，讓系統能達到統計穩態。對應 `config.py::EPSILON`。

**AR(1) 噪音（一階自迴歸噪音）**
`noise(t) = α·noise(t-1) + σ·randn()`，一種帶有記憶性（時間相關）的隨機噪音，
用來在緯向（k_x = 0）尺度持續注入小擾動、打破對稱性並維持湍流的非平衡狀態。
對應 `core/forcing.py::update_zonal_noise()`，`config.py` 的 `Tc`、`alpha`、`sigma`。

**渦度（Vorticity）ω_z**
流體局部旋轉強弱的量度，定義為 `ω_z = ∂u_y/∂x − ∂u_x/∂y`。
程式以中央差分計算（`analysis/vorticity.py::compute_vorticity_kernel`）。

**渦度偏態（Vorticity skewness）**
渦度分布的三階統計量 `⟨ω³⟩ / ⟨ω²⟩^{3/2}`，用來判斷渦漩的不對稱性。
木星觀測顯示渦度分布呈負偏態（反氣旋 anticyclone 較強較多），本程式以此
作為模擬是否重現真實木星動力學特徵的診斷指標。

### 其他工程名詞

**Taichi**
一個 Python 的高效能運算框架，可以把用 Python 語法寫的核函數（`@ti.kernel` /
`@ti.func`）自動編譯到 GPU（CUDA/Metal/Vulkan）執行，是本程式達到大規模格點
（512×256、70 萬步）即時模擬的關鍵。

**ping-pong 緩衝（Double buffering）**
用兩塊記憶體（`f` 與 `f_new`）交替作為讀取來源與寫入目的地，每步只需交換指標
（`src, dst = dst, src`），避免同一時刻讀寫同一塊記憶體造成的資料競爭，同時省去
整場複製的開銷。

---

## 輸出說明

### 圖表

| 檔案 | 內容 |
|------|------|
| `output/summary.png` | **3×3** 診斷總覽圖：U_rms、噴流數、能量譜斜率、帶狀動能佔比、渦度 RMS/Reynolds stress、渦度偏態、R_β*、Q_y 符號反轉、PV 階梯分數 |
| `output/hovmoller.png` | ū(y,t) Hovmöller 圖（RdBu_r 配色，y=緯度、x=時間） |
| `output/jet_autocorr.png` | 各緯度自相關時間 τ_corr(y) 及比值 τ_corr / τ_eddy |

### 影片

| 檔案 | 內容 |
|------|------|
| `output/vel_evolution.mp4` | 速度量值場隨時間演化 |
| `output/vort_evolution.mp4` | 渦度場隨時間演化 |
| `output/zonal_evolution.mp4` | 帶狀平均風速剖面演化（低頻採樣） |
| `output/spectrum_evolution.mp4` | 能量譜演化（低頻採樣） |

### 資料檔

| 檔案 | 格式 | 內容 |
|------|------|------|
| `data/processed/log.csv` | CSV | 所有診斷純量彙整（17 欄） |
| `data/processed/<metric>.csv` | CSV | 各診斷量獨立時序（每步即時寫入，共 16 個） |
| `data/processed/hovmoller.npz` | NPZ | `u_bar(T, NY)`、`steps(T)`、`y(NY)` |
| `data/processed/rs_profile.npz` | NPZ | `RS_profile(T, NY)`、`steps`、`y` |
| `data/processed/qy_profile.npz` | NPZ | `Qy(T, NY)`、`steps`、`y` |
| `data/processed/pv_profile.npz` | NPZ | `q_pv(T, NY)`、`steps`、`y` |

## 主要參數（`config.py`）

| 參數 | 意義 | 預設值 |
|------|------|--------|
| `NX`, `NY` | 網格解析度（緯向 × 緯度方向） | 512 × 256 |
| `MAX_STEPS` | 總演化步數 | 700000 |
| `SAVE_EVERY` | 每幾步存一幀並計算診斷量 | 5000 |
| `F0`, `BETA` | β-plane 科氏參數基準值與梯度 | 2e-4, 依 NY 換算 |
| `EPSILON` | Rayleigh drag 摩擦係數（同時用於估算能量注入率 ε） | 3e-5 |
| `SPONGE_FRAC`, `EPSILON_MAX` | 海綿層厚度比例、最大阻尼 | 0.15, 7e-5 |
| `Tc`, `sigma` | AR(1) 噪音相關時間、振幅 | 400, 1e-6 |
| `WARMUP_STEPS` | 噪音注入前的暖機步數 | 100000 |
| `NU`, `TAU` | 運動黏滯係數、鬆弛時間 | 0.0002, 3ν+0.5 |
| `s1, s2, s4, s6` | MRT 各矩量鬆弛係數 | 1.2, 1.2, 1.8, 1.8 |
