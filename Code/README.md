# LBM_Jupiter — 旋轉流體帶狀風系模擬

基於 PDF「利用高精度波茲曼格子法探討旋轉流體中帶狀風系之自發形成機制」

## 程式架構

```
LBM_Jupiter/
├── main.py              ← 主程式入口（模擬迴圈）
├── config.py            ← 全域參數（網格、物理、LBM 常數）
├── param_sweep.py       ← 參數掃描控制器（PDF §7）
│
├── core/
│   ├── lattice.py       ← D2Q9 常數 & f^eq 計算（PDF §4 Eq.3）
│   ├── collision.py     ← BGK 碰撞 + Guo's forcing（GPU fields 定義於此）
│   ├── streaming.py     ← Pull-scheme 串流（已融合於 collision.py）
│   └── forcing.py       ← Coriolis + 熱力外力更新（寫入 Fx/Fy fields）
│
├── physics/
│   ├── coriolis.py      ← β-plane 剖面 & Rhines 尺度（PDF §3 Eq.2）
│   ├── thermal.py       ← 溫度場演化（PDF §5 Eq.4-5）
│   └── nondimensional.py← Ro, Re, Rhines 等無因次數
│
├── analysis/
│   ├── vorticity.py     ← 渦度 ω_z = ∂uy/∂x − ∂ux/∂y
│   ├── spectrum.py      ← FFT 能量譜 E(k)（PDF §6 Eq.6）
│   └── zonal_mean.py    ← 帶狀平均風速 & 噴射流計數
│
├── utils/
│   ├── fft.py           ← FFT 工具（shell average, 視窗函數）
│   └── plotting.py      ← 視覺化（速度場、渦度、能量譜、帶狀剖面）
│
└── data/
    ├── nasa_raw/        ← NASA 原始數據
    └── processed/       ← 處理後數據（log.npy 等）
```

## 資料流

```
config.py（參數）
   │
   ▼
core/collision.py  ← 定義所有 GPU fields (f, rho, ux, uy, Fx, Fy)
   │
   ├── physics/thermal.py   → T_field
   │        │
   │        ▼
   └── core/forcing.py  → update_forcing_kernel() → Fx_field, Fy_field
            │
            ▼
   bgk_collision_kernel(omega)  [Guo's forcing scheme]
            │
            ▼
   swap_fields() + apply_periodic_bc_y()
            │
            ▼
   analysis/ (vorticity, spectrum, zonal_mean)
            │
            ▼
   utils/plotting.py → output/
```

## 邊界條件

| 方向 | 條件 | 位置 |
|------|------|------|
| x（緯向） | 週期 | collision.py Pull-scheme 中 `% NX` |
| y（緯度） | 週期 | collision.py Pull-scheme 中 `% NY` + `apply_periodic_bc_y()` |

## 物理公式對應

| PDF 公式 | 程式碼位置 |
|---------|----------|
| Eq.1 控制方程 | `core/forcing.py::update_forcing_kernel` |
| Eq.2 β-plane  | `core/forcing.py::update_forcing_kernel` & `physics/coriolis.py` |
| Eq.3 LBM 碰撞  | `core/collision.py::bgk_collision_kernel` |
| Eq.4 溫度剖面  | `physics/thermal.py::init_temperature_kernel` |
| Eq.5 熱力      | `core/forcing.py::update_forcing_kernel` |
| Eq.6 能量譜    | `analysis/spectrum.py::compute_energy_spectrum` |

## 執行

```bash
# 主模擬
python main.py

# 參數掃描（PDF §7，81 組）
python param_sweep.py
```

## 參數掃描範圍（PDF §7）

| 參數 | 符號 | 範圍 | config 變數 |
|------|------|------|------------|
| Coriolis 梯度 | β | 1e-5–1e-3 | `BETA` |
| 熱梯度強度 | ΔT | 0.01–0.1 | `DELTA_T` |
| 黏滯係數 | ν | 0.001–0.01 | `NU` |
| 噪音強度 | ε | 1e-4–1e-2 | `NOISE_AMP` |
| 鬆弛時間 | τ | 自動由 ν 計算 | `TAU = 3ν+0.5` |
