#外掛工具
import taichi as ti
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.collision import f, f_new, rho_field

#串流模組：獨立的 Push-scheme 串流（週期邊界）（現已被取代，保留作為參考）
@ti.kernel
def stream_kernel():
    for y, x in rho_field:
        for i in ti.static(range(9)):
            nx_idx = (x + cfg.CX[i] + cfg.NX) % cfg.NX
            ny_idx = (y + cfg.CY[i] + cfg.NY) % cfg.NY
            f_new[i, ny_idx, nx_idx] = f[i, y, x]