# -*- coding: utf-8 -*-
"""
EDA 第二课：零缺失 ≠ 干净 —— NaN 之外的三种传感器病
====================================================
2018/2019 两年数据零缺失（用户已验证），但真实 SCADA 的传感器故障
通常不以 NaN 出现——采集系统掉线时自动"保持最后读数"，数值就冻结了。
本脚本查三种病：
  1. 卡死（flatline）：连续 N 个采样点数值完全相同
  2. 尖峰（spike）：相邻采样点之间的异常大幅跳变
  3. 物理矛盾：水泵停机时水箱液位反而上升（违反水量平衡）
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")
FIGDIR = os.path.join(BASE, "..", "figures")


def load(year: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, f"{year}_SCADA_{kind}.csv"),
                       sep=";", decimal=",", parse_dates=["Timestamp"],
                       index_col="Timestamp")


def longest_flatline(s: pd.Series):
    """最长连续相同值游程：(长度, 开始时间, 结束时间)"""
    grp = (s != s.shift()).cumsum()          # 行号差分法——Q19 的老朋友
    run_len = s.groupby(grp).size()
    gid = run_len.idxmax()
    end = s.index[grp == gid][-1]
    start = s.index[grp == gid][0]
    return run_len.max(), start, end


frames = {}
for year in ["2018", "2019"]:
    for kind in ["flows", "pressures", "levels"]:
        frames[(year, kind)] = load(year, kind)

# ===== 1. 卡死体检：各传感器最长"冻结"游程 =====
print("=" * 64)
print("1) 卡死（flatline）体检：最长连续相同值游程 TOP8（5分钟粒度）")
results = []
for (year, kind), df in frames.items():
    for col in df.columns:
        run_len, start, end = longest_flatline(df[col])
        results.append((run_len, f"{year}/{kind}:{col}", start, end))
results.sort(reverse=True)
for run_len, name, start, end in results[:8]:
    print(f"  {name:<22} 连续 {run_len:>4} 个点 = {run_len * 5 / 60:5.1f} 小时"
          f"（{start} ~ {end}）")

# ===== 2. 尖峰体检：相邻采样的最大跳变 =====
print("\n2) 尖峰（spike）体检：单步最大跳变 TOP8")
spikes = []
for (year, kind), df in frames.items():
    for col in df.columns:
        diff = df[col].diff().abs()
        ts = diff.idxmax()
        spikes.append((diff.max(), f"{year}/{kind}:{col}", ts,
                       df[col].asof(ts - pd.Timedelta(minutes=5)),
                       df[col].asof(ts)))
spikes.sort(reverse=True)
for jump, name, ts, before, after in spikes[:8]:
    print(f"  {name:<22} {before:>8.2f} → {after:>8.2f}  单步跳 {jump:8.2f}  @ {ts}")

# ===== 3. 物理矛盾：泵停机时水箱液位不该上升 =====
print("\n3) 物理矛盾体检：PUMP_1 停机期间水箱液位变化")
for year in ["2018", "2019"]:
    lvl = frames[(year, "levels")]["T1"]
    pump = frames[(year, "flows")]["PUMP_1"]
    pump_off = pump < 0.5
    d_lvl = lvl.diff()
    rises = d_lvl[pump_off & (d_lvl > 1e-6)]
    print(f"  {year}: 泵停机时长占比 {pump_off.mean():.0%}，"
          f"停机中液位不降反升的采样点 {len(rises)} 个"
          + (f"（最大上升 {rises.max():.3f} m @ {rises.idxmax()}）" if len(rises) else ""))

# ===== 4. 画一张可疑时段的放大图（卡死最严重的传感器）=====
worst = results[0]
name, start, end = worst[1], worst[2], worst[3]
year, kind_col = name.split("/")
kind, col = kind_col.split(":")
s = frames[(year, kind)][col]
window = s[(s.index >= start - pd.Timedelta(hours=6))
           & (s.index <= end + pd.Timedelta(hours=6))]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(window.index, window, marker=".", markersize=2, lw=0.8)
ax.axvspan(start, end, color="red", alpha=0.2,
           label=f"卡死区段：{worst[0]}点/{worst[0] * 5 / 60:.1f}小时")
ax.set_title(f"最长卡死时段放大：{name}")
ax.legend()
fig.autofmt_xdate()
out = os.path.join(FIGDIR, "EDA02_卡死时段.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\n图已保存：{out}")
