# -*- coding: utf-8 -*-
"""
EPANET 第二课：从零搭建一个管网（不依赖任何现成 .inp）
====================================================
场景：给一座迷你"校园"配水——
  水源（水库，水头 65m）→ 干管 → 3 栋楼（3 个用水节点，日夜双峰需水）

四步：定义组件 → 定义需水曲线 → 水力求解 → 导出 .inp
最后加一个彩蛋：在干管上人为加一个漏口，看压力怎么变
（这就是模型法漏损定位的原理，也是 BattLeDIM 数据的生成方式）
"""
import os
import sys

import matplotlib.pyplot as plt
import wntr
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(BASE, "..", "figures")

# ===== ① 从零创建模型：空模型开始，一个组件一个组件加 =====
wn = wntr.network.WaterNetworkModel()
wn.options.time.duration = 24 * 3600          # 模拟 24 小时
wn.options.time.hydraulic_timestep = 3600     # 1 小时步长
wn.options.time.pattern_timestep = 3600

# 需水曲线：24 个小时系数（早晚双峰——和 water.db 里你见过的形状一致）
wn.add_pattern("daily", [0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.9, 1.2,
                         1.3, 1.1, 0.9, 0.8, 0.8, 0.8, 0.8, 0.9,
                         1.0, 1.2, 1.4, 1.3, 1.1, 0.9, 0.7, 0.6])

# 水源：水库（水头 65m，相当于高位水池）
wn.add_reservoir("R1", base_head=65.0, coordinates=(0, 50))

# 三栋楼：节点（标高、基础需水量 m³/s、坐标、绑定的需水曲线）
wn.add_junction("J1", base_demand=0.010, elevation=20,
                demand_pattern="daily", coordinates=(100, 50))
wn.add_junction("J2", base_demand=0.008, elevation=25,
                demand_pattern="daily", coordinates=(200, 65))
wn.add_junction("J3", base_demand=0.006, elevation=22,
                demand_pattern="daily", coordinates=(300, 50))

# 管道：长度 m / 直径 mm / 海曾-威廉粗糙系数
wn.add_pipe("P1", "R1", "J1", length=500, diameter=0.30, roughness=110, minor_loss=0.0)
wn.add_pipe("P2", "J1", "J2", length=400, diameter=0.20, roughness=110, minor_loss=0.0)
wn.add_pipe("P3", "J1", "J3", length=450, diameter=0.20, roughness=110, minor_loss=0.0)

print(f"从零搭建完成：{len(wn.nodes)} 节点 / {len(wn.links)} 管段")

# ===== ② 水力求解 =====
results = wntr.sim.EpanetSimulator(wn).run_sim()
pressure = results.node["pressure"]
print(f"模拟完成：{pressure.shape[0]} 个时刻 × {pressure.shape[1]} 个节点")
print(f"各节点压力范围（m）：")
for col in pressure.columns:
    print(f"  {col}: {pressure[col].min():.1f} ~ {pressure[col].max():.1f}")

# ===== ③ 导出 .inp —— 你自己的管网文件，EPANET GUI 可直接打开 =====
wntr.network.write_inpfile(wn, os.path.abspath(os.path.join(FIGDIR, "..", "data", "my_campus.inp")))
print("已导出 data/my_campus.inp（用 EPANET GUI → File → Open 打开它）")

# ===== ④ 画图：24 小时压力曲线 + 网络图 =====
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
for col in pressure.columns:
    if col != "R1":
        axes[0].plot(pressure.index / 3600, pressure[col], marker="o", ms=3, label=col)
axes[0].set_xlabel("时间（小时）")
axes[0].set_ylabel("压力 m")
axes[0].set_title("从零搭建的校园管网 · 24 小时压力曲线（双峰需水驱动）")
axes[0].legend()

wntr.graphics.plot_network(wn, node_attribute=pressure.iloc[-1],
                           node_size=80, ax=axes[1],
                           title="网络图（节点颜色 = 末时刻压力）")
axes[1].set_axis_off()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "EPANET02_从零搭建.png"),
            dpi=140, bbox_inches="tight")
print("图已保存：figures/EPANET02_从零搭建.png")

# ===== ⑤ 彩蛋：人为加漏，看压力怎么"出卖"漏点 =====
# 在 J3 加一个 5 cm² 的漏口，12 点出现、20 点修复，重跑模拟对比
j3 = wn.get_node("J3")
j3.add_leak(wn, area=0.0005, start_time=12 * 3600, end_time=20 * 3600)  # 5cm²
results_leak = wntr.sim.WNTRSimulator(wn).run_sim()  # EpanetSimulator 不模拟节点漏口，须用 WNTRSimulator
pl = results_leak.node["pressure"]

win = (pl.index >= 12 * 3600) & (pl.index < 20 * 3600)
print("\n彩蛋：J3 漏口（5cm²）活跃期间的压力对比")
for col in ["J1", "J2", "J3"]:
    no_leak = pressure.loc[win, col].mean()
    with_leak = pl.loc[win, col].mean()
    print(f"  {col}: 无漏 {no_leak:5.2f} m → 有漏 {with_leak:5.2f} m"
          f"（下降 {no_leak - with_leak:4.2f} m）")
print("→ 离漏口最近的 J3 压力下降最明显：这就是'压力-流量残差法'定位漏点的物理原理")

axes[0].plot(pl.index[win] / 3600, pl.loc[win, "J3"], marker="x", ms=4,
             color="red", ls="--", label="J3 有漏口时")
axes[0].legend()
fig.savefig(os.path.join(FIGDIR, "EPANET02_从零搭建.png"),
            dpi=140, bbox_inches="tight")
print("对比图已更新至 figures/EPANET02_从零搭建.png")
