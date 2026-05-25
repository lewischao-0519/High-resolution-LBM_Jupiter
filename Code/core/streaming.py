# core/streaming.py  ── 串流步（文件結構完整性，實際串流融合在 collision.py）
#
# 注意：本專案採用「Loop Fusion」策略，串流與碰撞合併在
# collision.py 的 bgk_collision_kernel 中（Pull-scheme）。
# 本檔案提供獨立串流 kernel，供除錯或 MRT 碰撞使用。

import taichi as ti
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from core.collision import f, f_new, rho_field


@ti.kernel
def stream_kernel():
    """
    Push-scheme 串流（獨立版，週期邊界）
    主流程使用 Pull-scheme（已融合在 bgk_collision_kernel），
    此函式僅供替代使用。
    """
    for y, x in rho_field:
        for i in ti.static(range(9)):
            nx_idx = (x + cfg.CX[i] + cfg.NX) % cfg.NX
            ny_idx = (y + cfg.CY[i] + cfg.NY) % cfg.NY
            f_new[i, ny_idx, nx_idx] = f[i, y, x]
