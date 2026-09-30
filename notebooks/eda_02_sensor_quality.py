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


def freeze_events(s: pd.Series, min_points: int = 12, tol: float = 0.0) -> pd.DataFrame:
    """提取所有"冻结"游程台账：连续 min_points 个采样点内数值变化不超过 tol。

    比逐事件建全表掩码快：位置号按游程分组一次算完（向量化，无逐事件循环）。
    12 个点 = 1 小时（5 分钟粒度）；tol>0 可捕捉"带微小抖动的卡死"。
    """
    change = s.diff().abs().gt(tol)      # 相对上一点变化超过 tol → 新游程开始
    change.iloc[0] = True
    grp = change.cumsum()
    pos = pd.Series(np.arange(len(s)), index=s.index)
    stats = pos.groupby(grp.values).agg(lens="size", end_pos="max")
    stats = stats[stats["lens"] >= min_points]
    return pd.DataFrame({
        "开始": s.index[stats["end_pos"] - stats["lens"] + 1].values,
        "结束": s.index[stats["end_pos"].values].values,
        "点数": stats["lens"].values,
        "持续小时": (stats["lens"].values * 5 / 60).round(1),
        "冻结值": s.values[stats["end_pos"].values],
    })


frames = {}
for year in ["2018", "2019"]:
    for kind in ["flows", "pressures", "levels"]:
        frames[(year, kind)] = load(year, kind)
# ===== 1. 卡死体检：全传感器冻结事件台账（≥1小时）=====
print("=" * 64)
print("1) 卡死（flatline）体检：冻结 ≥1 小时的事件台账")
register = []
for (year, kind), df in frames.items():
    for col in df.columns:
        ev = freeze_events(df[col], min_points=12)
        ev.insert(0, "传感器", f"{year}/{kind}:{col}")
        register.append(ev)
register = pd.concat(register, ignore_index=True).sort_values("持续小时",
                                                              ascending=False)

def 判读(row):
    if row["传感器"].endswith("PUMP_1") and row["冻结值"] < 0.5:
        return "正常停机"
    if row["持续小时"] >= 6:
        return "高度可疑（长冻结）"
    return "短时平稳段"
register["判读"] = register.apply(判读, axis=1)

print(register["判读"].value_counts().to_string())
print("\n【高度可疑】长冻结事件（≥6 小时，非泵停机）：")
sus = register[register["判读"] == "高度可疑（长冻结）"]
print(sus.head(12).to_string(index=False))
register.to_csv(os.path.join(BASE, "..", "传感器冻结事件台账.csv"),
                index=False, encoding="utf-8-sig")
print("台账已保存：传感器冻结事件台账.csv（utf-8-sig，Excel 可直接打开）")
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

# ===== 4. 画一张可疑时段的放大图（台账中卡死最严重的传感器）=====
worst = register.iloc[0]
name, start, end = worst["传感器"], worst["开始"], worst["结束"]
year, kind_col = name.split("/")
kind, col = kind_col.split(":")
s = frames[(year, kind)][col]
window = s[(s.index >= start - pd.Timedelta(hours=6))
           & (s.index <= end + pd.Timedelta(hours=6))]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(window.index, window, marker=".", markersize=2, lw=0.8)
ax.axvspan(start, end, color="red", alpha=0.2,
           label=f"卡死区段：{worst['点数']}点/{worst['持续小时']}小时"
                 f"（{worst['判读']}）")
ax.set_title(f"最长卡死时段放大：{name}")
ax.legend()
fig.autofmt_xdate()
out = os.path.join(FIGDIR, "EDA02_卡死时段.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\n图已保存：{out}")
