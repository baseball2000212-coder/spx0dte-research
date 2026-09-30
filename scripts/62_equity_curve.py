"""
2022-05 ~ 현재 누적 손익 (SPX ATM 콜 1계약). 계산은 spx0dte/realistic.py:
  매도호가 매수 + 체결 지연 5초·10초 (2023-03-28 전은 10:00:00 매도호가), −90% 손절(초 단위 중간가, 마감 직전 포함;
  초 단위 호가가 없는 날은 1분), 수수료 편도 $5·정산 $5, CME 시세료 월 $228.80. 백테스트 + 앞으로 기록.
결과: output/equity_2022_now.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from spx0dte.config import OUT
from spx0dte.exits import FEE, EXERCISE
from spx0dte.core import load_spx_ohlc
from spx0dte.config import SPX_CSV
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_rule, STOP
DELAYS = (5, 10)   # 체결 지연(초)

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
BLUE, ORANGE, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8a85", "#0b0b0b", "#5c5b55", "#e6e5e0"
CME = 228.80

from spx0dte import realistic as RL
TK = RL.load_ticks(); RATIO = RL.ratios(TK); LG = RL.load_legs()
SER = {k: RL.trades(LG, k, TK, RATIO)["손익$"] for k in DELAYS}
ref = {"0초·매도호가": RL.trades(LG, 0, TK, RATIO)["손익$"].sum(), **{f"{k}초·매도호가": v.sum() for k, v in SER.items()}}
bt_end = LG.index.max()
FW = {k: RL.forward(k, RATIO) for k in DELAYS}
FW = {k: v.loc[v.index > bt_end, "손익$"] for k, v in FW.items()}
d49 = pd.Timestamp("2025-04-09")
BS = {k: pd.concat([SER[k], FW[k]]).sort_index() for k in DELAYS}
first = min(v.index.min() for v in BS.values()); last = max(v.index.max() for v in BS.values())
mon = lambda idx: np.array([(t.year - first.year) * 12 + t.month - first.month + 1 for t in idx])
def net(x):
    return x.cumsum() - mon(x.index) * CME
nmon = mon([last])[0]
yrs = (last - first).days / 365.25

def dd_stats(c):
    dd = c - c.cummax(); i = dd.idxmin()
    peak = c[:i].idxmax()
    under = (dd < 0).astype(int).values; run = best = 0; e = 0
    for k, u in enumerate(under):
        run = run + 1 if u else 0
        if run > best: best, e = run, k
    s0, s1 = c.index[e - best + 1], c.index[e]
    return dd, dd.min(), peak, i, (s1 - s0).days, s0, s1

def boot(x, n=5000, seed=0):
    """거래 순서를 무작위로 섞어 최대낙폭 분포 (CME는 거래당 평균으로 나눠 뺌)."""
    rng = np.random.default_rng(seed); v = x.values - CME * nmon / len(x)
    out = np.empty(n)
    for k in range(n):
        c = np.cumsum(rng.choice(v, len(v), replace=True)); out[k] = (c - np.maximum.accumulate(np.maximum(c, 0))).min()
    return np.percentile(out, [50, 5, 1])

fig, (ax, dx) = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.3], "hspace": 0.08})
fig.patch.set_facecolor("#fcfcfb")
for x in (ax, dx):
    x.set_facecolor("#fcfcfb"); x.grid(axis="y", color=GRID, lw=0.8); x.set_axisbelow(True)
    for sp in ("top", "right"): x.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): x.spines[sp].set_color("#c9c8c1")
    x.tick_params(colors=INK2, labelsize=9)
rows = []
COL = {5: BLUE, 10: ORANGE}
for k in DELAYS:
    for drop, ls in ((False, "-"), (True, (0, (4, 2.5)))):
        x = BS[k].drop(d49) if drop else BS[k]; c = net(x)
        dd, mdd, pk, tr, days, s0, s1 = dd_stats(c); q = boot(x)
        nm = f"{k}초 지연" + (" · 4/9 뺌" if drop else "")
        rows.append((nm, c.iloc[-1], c.iloc[-1] / yrs, mdd, pk, tr, days, s0, s1, q))
        ax.step(c.index, c / 1000, where="post", color=COL[k], lw=2 if not drop else 1.6, ls=ls, label=f"{nm}   {c.iloc[-1] / 1000:+.1f}k · 최대낙폭 -{-mdd / 1000:.1f}k")
        ax.text(c.index[-1] + pd.Timedelta(days=12), c.iloc[-1] / 1000 + (-2.6 if k == 5 else 2.6), f"{k}초 ${c.iloc[-1] / 1000:,.1f}k", color=COL[k], fontsize=9.5, va="center", fontweight="bold")
        if not drop:
            dx.step(dd.index, dd / 1000, where="post", color=COL[k], lw=1.3)
            dx.fill_between(dd.index, dd / 1000, 0, step="post", color=COL[k], alpha=0.10, lw=0)
A = net(BS[5])
ax.axhline(0, color="#c9c8c1", lw=1)
ax.annotate("2025-04-09 하루 +$37k", xy=(d49, A.loc[d49] / 1000), xytext=(d49 - pd.Timedelta(days=330), A.loc[d49] / 1000 - 12),
            fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
for y in range(2023, last.year + 1):
    for x in (ax, dx): x.axvline(pd.Timestamp(f"{y}-01-01"), color=GRID, lw=0.8, zorder=0)
ax.set_ylabel("누적 손익 (천 달러)", color=INK2, fontsize=10); dx.set_ylabel("고점 대비 (천 달러)", color=INK2, fontsize=10)
ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK)
ax.set_title(f"SPX 0DTE 네이키드 매수 전략 — 누적 손익 (SPX 1계약, 2022-05 ~ {last:%Y-%m-%d})", loc="left", fontsize=13, color=INK, pad=26)
ax.text(0, 1.015, r"매도호가 매수 · 체결 지연 5초/10초 (2023-03 전은 10:00:00 매도호가) · -90% 손절 · 수수료 편도 \$5 + 정산 \$5 가정 · CME 시세료 월 \$228.80 · 백테스트 + 앞으로 기록", transform=ax.transAxes, fontsize=9, color=INK2)
dx.set_ylim(-27, 1)
dx.xaxis.set_major_locator(mdates.YearLocator()); dx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(first - pd.Timedelta(days=15), last + pd.Timedelta(days=150))
fig.savefig(OUT / "equity_2022_now.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
for nm, tot, yr, mdd, pk, tr, days, s0, s1, q in rows:
    print(f"{nm}: 총 {tot:,.0f}, 연 {yr:,.0f}, 최대낙폭 {mdd:,.0f} ({pk:%Y-%m-%d}→{tr:%Y-%m-%d}), 연수익/낙폭 {yr / -mdd:.2f}, "
          f"가장 긴 물린 기간 {days}일 ({s0:%Y-%m}~{s1:%Y-%m}), 재표본 낙폭 중앙값 {q[0]:,.0f} / 최악 5% {q[1]:,.0f} / 최악 1% {q[2]:,.0f}")

print("비율", {k: round(v, 4) for k, v in RATIO.items()}); print("참고 (CME 전, 수수료 $5):", {k: round(v) for k, v in ref.items()})
