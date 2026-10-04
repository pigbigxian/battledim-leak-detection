# -*- coding: utf-8 -*-
"""
ML 第三课：漏损检测（分类任务）
====================================
任务：判断 2018 年每一天"是否有突发型(abrupt)漏损处于活跃期"。

- 第一课：回归（预测 hourly 流量，其实是查表）
- 第二课：时序回归（预测明天的日均供水量）
- 这一课：分类（判断状态）。新指标：查准率/查全率——
  对水司来说，漏报 = 白白流走的水钱，误报 = 排人白跑一趟管网

明星特征：夜间最小流量（凌晨2~4点，真实用水趋近于零，
漏损占比达到全天最大——给排水常识直接变成特征列）

纪律不变：按时间切分（1~6月训练，7~12月考试），特征只用当日及以前。
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, "data")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load(year: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, f"{year}_SCADA_{kind}.csv"),
                       sep=";", decimal=",", parse_dates=["Timestamp"],
                       index_col="Timestamp")


# ===== ① 标准答案：从官方 yaml 解析"突发型"漏损事件 =====
with open(os.path.join(DATA, "dataset_configuration.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
events = []
for item in cfg["leakages"]:
    if not item or item.startswith("#"):
        continue
    parts = [p.strip() for p in item.split(",")]
    events.append({"link": parts[0], "start": pd.to_datetime(parts[1]),
                   "end": pd.to_datetime(parts[2]), "type": parts[4]})
abrupt = [e for e in events if e["type"] == "abrupt"
          and e["start"].year == 2018]
print(f"2018 年突发型漏损事件：{len(abrupt)} 起")
for e in abrupt:
    print(f"  {e['link']:<6}{e['start']} ~ {e['end']}")

# 逐日标签：当天有突发漏损活跃 = 1
days = pd.date_range("2018-01-01", "2018-12-31", freq="D")
label = pd.Series(0, index=days)
for e in abrupt:
    s = max(e["start"], pd.Timestamp("2018-01-01")).normalize()
    t = min(e["end"], pd.Timestamp("2018-12-31 23:55")).normalize()
    label[s:t] = 1

# ===== ② 特征工程：日级 SCADA 统计 =====
flows = load("2018", "flows")
flows["inflow"] = flows["p227"] + flows["p235"]          # 两个进水口合计
night = flows.loc[flows.index.hour.isin([2, 3, 4 ]), "inflow"]
night_daily = night.groupby(night.index.normalize()).mean()   # ★夜间最小流量

daily = flows["inflow"].resample("D").agg(inflow_mean="mean", inflow_max="max")
daily["night_flow"] = night_daily                          # ★明星特征
levels = load("2018", "levels")
daily["lvl_mean"] = levels["T1"].resample("D").mean()
daily["lvl_std"] = levels["T1"].resample("D").std()
p_daily = load("2018", "pressures").resample("D").mean()
p_daily.columns = [f"p_{c}" for c in p_daily.columns]
daily = daily.join(p_daily)
daily["dow"] = daily.index.dayofweek
daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
daily["label"] = label.reindex(daily.index).fillna(0).astype(int)

FEATURES = [c for c in daily.columns if c != "label"]
train = daily[daily.index < "2018-07-01"]
test = daily[daily.index >= "2018-07-01"]
print(f"\n训练集 {len(train)} 天（正样本 {train['label'].sum()}）｜"
      f"测试集 {len(test)} 天（正样本 {test['label'].sum()}）")

# ===== ③ 训练 + 考试 =====
model = RandomForestClassifier(n_estimators=300, random_state=42,
                               n_jobs=-1, class_weight="balanced")
model.fit(train[FEATURES], train["label"])
test = test.copy()
test["proba"] = model.predict_proba(test[FEATURES])[:, 1]
test["pred"] = (test["proba"] >= 0.5).astype(int)
test["pred_3d"]=((test["pred"].rolling(window=2).sum()==2).fillna(test['pred'].astype(bool)).astype(int))

print(f"\n查准率 precision：{precision_score(test['label'], test['pred_3d']):.2f}"
      f"（报警了，多少次是真的）")
print(f"查全率 recall   ：{recall_score(test['label'], test['pred_3d']):.2f}"
      f"（真漏了，多少次被抓到）")
print("混淆矩阵 [真负, 误报 / 漏报, 真正]：")
print(confusion_matrix(test["label"], test["pred_3d"]))
print("\n特征重要性 TOP8：")
for name, imp in sorted(zip(FEATURES, model.feature_importances_),
                        key=lambda x: -x[1])[:8]:
    print(f"  {name:<12}{imp:.0%}")

# ===== ④ 逐事件核对：测试期 3 起突发漏损，每起抓到没有 =====
print("\n逐事件核对：")
for e in abrupt:
    if e["end"] < pd.Timestamp("2018-07-01"):
        continue
    s = max(e["start"], pd.Timestamp("2018-07-01")).normalize()
    t = min(e["end"], pd.Timestamp("2018-12-31 23:55")).normalize()
    seg = test.loc[s:t]
    hit = (seg["proba"] >= 0.5).mean()
    print(f"  {e['link']:<6}{s.date()} ~ {t.date()}  "
          f"事件内报警天占比 {hit:.0%}")

# ===== ⑤ 画图：夜间流量（上）与模型报警概率（下）=====
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for e in abrupt:
    if e["end"] >= pd.Timestamp("2018-07-01"):
        axes[0].axvspan(max(e["start"], pd.Timestamp("2018-07-01")),
                        min(e["end"], pd.Timestamp("2018-12-31")),
                        color="crimson", alpha=0.15)
        axes[1].axvspan(max(e["start"], pd.Timestamp("2018-07-01")),
                        min(e["end"], pd.Timestamp("2018-12-31")),
                        color="crimson", alpha=0.15)
axes[0].plot(test.index, test["night_flow"], color="tab:blue", lw=1.2)
axes[0].set_ylabel("夜间最小流量 m³/h")
axes[0].set_title("2018 下半年 · 突发漏损检测（红底=真实漏损期）")
axes[1].plot(test.index, test["proba"], color="tab:orange", lw=1.2,
             label="模型报警概率")
axes[1].axhline(0.5, color="gray", ls="--", lw=1, label="报警阈值 0.5")
axes[1].set_ylabel("报警概率")
axes[1].set_ylim(0, 1)
axes[1].legend(loc="upper right")
fig.autofmt_xdate()
out = os.path.join(FIGDIR, "ML03_漏损检测.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\n图已保存：{out}")
