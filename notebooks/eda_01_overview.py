# -*- coding: utf-8 -*-
"""
EDA 第一课：BattLeDIM 2018 历史数据全景盘点
=============================================
真实数据第一坑：欧式格式——分号分隔、逗号做小数点（77,77 就是 77.77）。
本脚本回答四个问题：
  1. 数据完整吗？（行数、时间范围、缺失值）
  2. 传感器都长什么样？（流量/液位/压力全景图）
  3. 漏损事件在数据里长什么样？（对照官方漏损档案）
  4. 哪些传感器掉线最严重？（数据质量黑名单）
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")
FIGDIR = os.path.join(BASE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load_scada(name: str) -> pd.DataFrame:
    """欧式格式读取：分号分隔 + 逗号小数点"""
    return pd.read_csv(os.path.join(DATA, name), sep=";", decimal=",",
                       parse_dates=["Timestamp"], index_col="Timestamp")


# ===== 1. 逐表盘点 =====
files = {
    "flows": "2018_SCADA_Flows.csv",
    "pressures": "2018_SCADA_Pressures.csv",
    "demands": "2018_SCADA_Demands.csv",
    "levels": "2018_SCADA_Levels.csv",
    "leakages": "2018_Leakages.csv",
}
data = {}
print("=" * 62)
for key, fname in files.items():
    df = load_scada(fname)
    data[key] = df
    n_missing = int(df.isna().sum().sum())
    print(f"{fname:<28} {df.shape[0]:>6} 行 × {df.shape[1]:>2} 列 | "
          f"{df.index.min()} ~ {df.index.max()} | 缺失 {n_missing}")

# ===== 2. 漏损事件验证：Leakages.csv 的非零时段 vs 官方档案 =====
print("\n漏损标准答案（Leakages.csv 非零时段）：")
leaks = data["leakages"]
for col in leaks.columns:
    active = leaks[col][leaks[col] > 0]
    if len(active):
        print(f"  {col:<6} {active.index.min()} ~ {active.index.max()}  "
              f"峰值 {active.max():>8.1f} L/h")

# ===== 3. 数据质量黑名单：缺失最多的传感器 =====
print("\n缺失最多的传感器 TOP5（各表合计）：")
all_missing = pd.concat([data[k].isna().sum() for k in ["flows", "pressures", "levels"]])
print(all_missing.sort_values(ascending=False).head(5).to_string())

# ===== 4. 全景图 =====
fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

daily_f = data["flows"].resample("D").mean()
for col in daily_f.columns:
    axes[0].plot(daily_f.index, daily_f[col], label=col, lw=1)
axes[0].set_ylabel("流量 m³/h")
axes[0].set_title("2018 全年 SCADA 全景：流量 / 水箱液位 / 压力 / 漏损事件")
axes[0].legend(loc="upper left", ncol=3, fontsize=8)

lvl = data["levels"].resample("D").mean()
axes[1].plot(lvl.index, lvl["T1"], color="tab:blue", lw=1)
axes[1].set_ylabel("水箱液位 m")

daily_p = data["pressures"].resample("D").mean()
for col in ["n1", "n105", "n415", "n613"]:
    axes[2].plot(daily_p.index, daily_p[col], label=col, lw=1)
axes[2].set_ylabel("压力 m")
axes[2].legend(loc="upper left", ncol=4, fontsize=8)

leak_daily = leaks.sum(axis=1).resample("D").mean()
axes[3].fill_between(leak_daily.index, leak_daily, color="crimson", alpha=0.6, lw=0)
axes[3].set_ylabel("漏损流量 L/h")
axes[3].set_xlabel("2018 年")

fig.tight_layout()
out = os.path.join(FIGDIR, "EDA01_2018全景.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\n图已保存：{out}")
