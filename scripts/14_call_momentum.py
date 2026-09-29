"""
후보: '개장 후 올랐으면 ATM 콜 1개 매수' (09:45·10:00·10:30), 만기 보유 vs -30% 손절. 12 결과 사용, 몇 초.
결과: output/D4_call_momentum.png, D4_call_momentum.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT
from spx0dte.exits import FEE, EXERCISE

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
SPLIT = pd.Timestamp("2025-01-01")
ENTRY = ["09:45", "10:00", "10:30"]

L = pd.read_pickle(OUT / "leg_paths.pkl"); FP = pd.read_pickle(OUT / "f_paths.pkl")
f_open = FP[list(FP.columns[:5])].bfill(axis=1).iloc[:, 0]
ret = lambda g, c: g[c].sum() / g.cost.sum() * 100

rows, trades = [], {}
for e in ENTRY:
    s = L[L.entry == e].copy()
    s["r"] = s.F0.values / f_open.reindex(s.date).values - 1
    for leg, cond in (("call", s.r > 0), ("put", s.r < 0)):          # 상승일 콜 / 하락일 풋 (비교용)
        h = s[cond].sort_values("date").copy()
        h["cost"] = h[f"{leg}_ask"]; pay = h[f"{leg}_pay"]
        h["만기"] = pay - h.cost - FEE - np.where(pay > 0, EXERCISE, 0)
        sl = []
        for r, c, p in zip(h[f"{leg}_bid"], h.cost, h["만기"]):
            hit = np.isfinite(r) & (r <= c * 0.7)
            sl.append(r[hit.argmax()] - c - 2 * FEE if hit.any() else p)
        h["-30% 손절"] = sl
        trades[(e, leg)] = h
        for rule in ("만기", "-30% 손절"):
            eq = (h[rule] * 100).cumsum()
            top10 = h.drop(h[rule].nlargest(10).index)
            rows.append({"진입": e, "매수": "상승일 콜" if leg == "call" else "하락일 풋", "청산": rule, "거래": len(h),
                         "평균 프리미엄$": h.cost.mean() * 100, "학습%": ret(h[h.date < SPLIT], rule),
                         "검증%": ret(h[h.date >= SPLIT], rule), "승률%": (h[rule] > 0).mean() * 100,
                         "총손익$": eq.iloc[-1], "최대낙폭$": (eq - eq.cummax()).min(),
                         "최고 10일 빼면%": ret(top10, rule)})
S = pd.DataFrame(rows).round(1)
S.to_csv(OUT / "D4_call_momentum.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
for e, col in zip(ENTRY, ("C0", "C1", "C2")):
    h = trades[(e, "call")]
    ax[0].plot(h.date, (h["만기"] * 100).cumsum() / 1000, color=col, label=f"{e} 만기")
    ax[0].plot(h.date, (h["-30% 손절"] * 100).cumsum() / 1000, color=col, ls=":", label=f"{e} -30% 손절")
    p = trades[(e, "put")]
    ax[1].plot(p.date, (p["만기"] * 100).cumsum() / 1000, color=col, label=f"{e} 만기")
for x in ax[:2]:
    x.axhline(0, color="k", lw=1); x.axvline(SPLIT, color="grey", ls="--", lw=1); x.legend(fontsize=8); x.grid(alpha=.3)
ax[0].set_title("개장 후 올랐으면 콜 1개 매수: 누적 손익 (천 달러)")
ax[1].set_title("(비교) 개장 후 내렸으면 풋 1개 매수: 누적 손익 (천 달러)")
h = trades[("10:30", "call")]
v = np.sort((h["만기"] * 100).values)[::-1]
ax[2].plot(np.arange(1, len(v) + 1), v.cumsum() / 1000, color="C2")
ax[2].axhline(0, color="k", lw=1); ax[2].grid(alpha=.3)
ax[2].set_xlabel("좋은 날부터 정렬한 거래 수"); ax[2].set_title("10:30 콜 (만기): 좋은 날 순 누적 → 이익이 소수의 날에 몰림")
fig.tight_layout(); fig.savefig(OUT / "D4_call_momentum.png", dpi=115); plt.close(fig)
pd.set_option("display.width", 250)
print(S.to_string(index=False))
