# -*- coding: utf-8 -*-
"""
EPANET 第一课：认识管网水力模型
====================================
L-TOWN.inp 就是 BattLeDIM 那个虚拟城市的完整管网模型（782 根管道）。

EPANET 干的事，翻译成你的专业课语言：
  给定水源、水池、泵、管道拓扑和需水模式，
  联立求解全部节点的连续方程和环路的能量方程——
  就是"管网平差"的电脑版，只不过解的是几百个未知数，
  而且是随时变化的动态水力模拟。

输出：每个节点任意时刻的压力、每根管道的流量/流速/水头损失。
这就是智慧水务平台里"水力模型在线模拟"功能的引擎。
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
DATA = os.path.join(BASE, "..", "data")
FIGDIR = os.path.join(BASE, "..", "figures")

# ===== ① 载入管网模型 =====
wn = wntr.network.WaterNetworkModel(os.path.join(DATA, "L-TOWN.inp"))
print(f"管网规模：节点 {len(wn.nodes)} | 管段 {len(wn.links)} | "
      f"水源 {len(list(wn.reservoirs()))} | 水池 {len(list(wn.tanks()))} | "
      f"水泵 {len(list(wn.pumps()))} | 阀门 {len(list(wn.valves()))}")
print(f"inp 自带模拟时长：{wn.options.time.duration / 3600:.0f} 小时，"
      f"水力步长 {wn.options.time.hydraulic_timestep / 60:.0f} 分钟")

# ===== ② 把模拟压到 24 小时（原始设置可能很长，跑不动）=====
wn.options.time.duration = 24 * 3600
wn.options.time.hydraulic_timestep = 15 * 60      # 15 分钟步长，与 SCADA 对齐

# ===== ③ 水力计算（EPA 的 EPANET 引擎在后台求解）=====
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

pressure = results.node["pressure"]               # 行=时刻，列=节点
flow = results.link["flowrate"]
print(f"模拟完成：压力表 {pressure.shape}，流量表 {flow.shape}")
print(f"全网压力范围：{pressure.min().min():.1f} ~ {pressure.max().max():.1f} m")

# ===== ④ 网络压力地图（模拟末时刻，颜色=压力）=====
fig, ax = plt.subplots(figsize=(11, 9))
wntr.graphics.plot_network(
    wn, node_attribute=pressure.iloc[-1], link_attribute=None,
    node_size=12, node_colorbar_label="压力 m",
    title="L-TOWN 管网压力分布（EPANET 水力模拟，末时刻）", ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "EPANET01_压力地图.png"),
            dpi=140, bbox_inches="tight")
print(f"压力地图已保存：figures/EPANET01_压力地图.png")

# ===== ⑤ 抽两个节点的 24 小时压力曲线 =====
fig, ax = plt.subplots(figsize=(11, 4))
for node in ["n1", "n215"]:                        # n215 就是那个会"冻结"的传感器节点！
    if node in pressure.columns:
        ax.plot(pressure.index / 3600, pressure[node], label=node, lw=1.2)
ax.set_xlabel("模拟时间（小时）")
ax.set_ylabel("压力 m")
ax.set_title("节点 24 小时压力曲线（EPANET 模拟）")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "EPANET01_压力曲线.png"),
            dpi=140, bbox_inches="tight")
print("压力曲线已保存：figures/EPANET01_压力曲线.png")
