# -*- coding: utf-8 -*-
"""
GIS 桥第一课：把 L-TOWN 管网导出成 GeoPackage（QGIS 原生格式）
================================================================
管网模型和 GIS 本是同一份数据的两种说法：
  - EPANET .inp：拓扑 + 水力属性（节点坐标、管长、管径）
  - GIS 图层：空间几何 + 属性表
本脚本把 L-TOWN 的节点和管道各导成一个图层，附带 24 小时模拟的末时刻压力——
在 QGIS 里打开后，按 pressure 字段染色，就是上节课那张压力地图的"真 GIS 版"。
"""
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import wntr
from shapely.geometry import LineString, Point

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")
FIGDIR = os.path.join(BASE, "..", "figures")
GPKG = os.path.join(DATA, "ltown_v2.gpkg")

# ===== ① 载入模型 + 跑一次 24 小时水力模拟 =====
wn = wntr.network.WaterNetworkModel(os.path.join(DATA, "L-TOWN.inp"))
wn.options.time.duration = 24 * 3600
wn.options.time.hydraulic_timestep = 15 * 60
pressure = wntr.sim.EpanetSimulator(wn).run_sim().node["pressure"]
final_p = pressure.iloc[-1]                      # 末时刻各节点压力

# ===== ② 节点图层：每个节点一个点，属性带上类型和压力 =====
node_rows = []
for name, node in wn.nodes():
    node_rows.append({
        "name": name,
        "type": node.node_type,
        "pressure_m": round(float(final_p.get(name, np.nan)), 2),
        "geometry": Point(*node.coordinates),
    })
gdf_nodes = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")

# ===== ③ 管段图层：全部 909 根 link（管道 + 水泵 + 阀门），属性带类型 =====
link_rows = []
for name, link in wn.links():
    u = wn.get_node(link.start_node).coordinates
    v = wn.get_node(link.end_node).coordinates
    link_rows.append({
        "name": name,
        "type": link.link_type,                              # Pipe / Pump / Valve
        "diameter_mm": round(link.diameter * 1000) if link.link_type == "Pipe" else None,
        "length_m": round(link.length, 1) if getattr(link, "length", None) else None,
        "geometry": LineString([u, v]),
    })
gdf_links = gpd.GeoDataFrame(link_rows, geometry="geometry", crs="EPSG:4326")

n_pipe = sum(1 for r in link_rows if r["type"] == "Pipe")
n_pump = sum(1 for r in link_rows if r["type"] == "Pump")
n_valve = sum(1 for r in link_rows if r["type"] == "Valve")
print(f"节点图层 {len(gdf_nodes)} 个要素 | 管段图层 {len(gdf_links)} 个要素"
      f"（管道 {n_pipe}、水泵 {n_pump}、阀门 {n_valve}）")

# ===== ④ 写 GeoPackage（一个文件装两个图层）=====
if os.path.exists(GPKG):
    os.remove(GPKG)
gdf_links.to_file(GPKG, layer="links", driver="GPKG")
gdf_nodes.to_file(GPKG, layer="nodes", driver="GPKG")
print(f"已导出：{GPKG}")

# ===== ⑤ 自检：用 geopandas 读回来画一张（QGIS 里打开后就是类似效果）=====
fig, ax = plt.subplots(figsize=(10, 9))
gdf_links.plot(ax=ax, color="gray", lw=0.6, aspect=None)
gdf_nodes.plot(ax=ax, column="pressure_m", cmap="viridis", markersize=6, aspect=None,
               legend=True, legend_kwds={"label": "末时刻压力 m"})
ax.set_title("L-TOWN 管网（GeoPackage 预览，QGIS 打开效果类似）")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "GIS01_ltown_gpkg.png"), dpi=140, bbox_inches="tight")
print("预览图已保存：figures/GIS01_ltown_gpkg.png")
