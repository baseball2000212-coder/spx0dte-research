"""
일목 1차 점검: 10:30 진입 때 SPX가 구름 위/안/아래 어디냐 → '오른 날 콜' / '내린 날 풋' 성과가 갈리나 (ATM, 만기).
  - 1분봉: 10:30 봉까지, 5분봉: 10:25 봉(10:25~10:29)까지, 60분봉: 09:30 봉(09:30~10:29)까지 = 진입 시점에 완성된 봉만
  - 4/9(관세 유예) 뺀 검증도 같이
결과: output/I_filter.png, I_filter.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT
from spx0dte.exits import FEE, EXERCISE
from spx0dte.ichimoku import bars, ichimoku

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
SPLIT, ODD = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-09")
LAST_BAR = {1: "10:30", 5: "10:25", 60: "09:30"}
STATES = ["구름 위", "구름 안", "구름 아래"]

FP = pd.read_pickle(OUT / "f_paths.pkl")
f_open = FP[list(FP.columns[:5])].bfill(axis=1).iloc[:, 0]
L = pd.read_pickle(OUT / "leg_paths.pkl"); L = L[L.entry == "10:30"].set_index("date")
L["r"] = L.F0 / f_open.reindex(L.index) - 1

for n, hhmm in LAST_BAR.items():
    b = bars(FP, n); ic = ichimoku(b)
    ts = L.index + pd.to_timedelta(hhmm + ":00")
    c, top, bot = b.close.reindex(ts).values, ic["구름위"].reindex(ts).values, ic["구름아래"].reindex(ts).values
    L[f"{n}분"] = np.select([c > top, c < bot, (c >= bot) & (c <= top)], STATES, default="")

rows = []
for leg, cond in (("오른 날 콜", L.r > 0), ("내린 날 풋", L.r < 0)):
    h = L[cond].copy()
    k = "call" if "콜" in leg else "put"
    h["cost"], pay = h[f"{k}_ask"], h[f"{k}_pay"]
    h["pnl"] = pay - h.cost - FEE - np.where(pay > 0, EXERCISE, 0)
    for n in LAST_BAR:
        for st in STATES:
            g = h[h[f"{n}분"] == st]
            tr, va = g[g.index < SPLIT], g[g.index >= SPLIT]
            va2 = va[va.index != ODD]
            r = lambda x: x.pnl.sum() / x.cost.sum() * 100 if len(x) else np.nan
            rows.append({"매수": leg, "봉": f"{n}분", "위치": st, "건수": len(g), "학습%": r(tr), "검증%": r(va),
                         "검증(4/9 제외)%": r(va2), "승률%": (g.pnl > 0).mean() * 100 if len(g) else np.nan})
R = pd.DataFrame(rows).round(1)
R.to_csv(OUT / "I_filter.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(2, 3, figsize=(18, 8))
for i, leg in enumerate(("오른 날 콜", "내린 날 풋")):
    for j, col in enumerate(("학습%", "검증%", "검증(4/9 제외)%")):
        H = R[R.매수 == leg].pivot(index="위치", columns="봉", values=col).reindex(index=STATES, columns=["1분", "5분", "60분"])
        N = R[R.매수 == leg].pivot(index="위치", columns="봉", values="건수").reindex(index=STATES, columns=["1분", "5분", "60분"])
        x = ax[i, j]
        x.imshow(H.values, cmap="RdYlGn", vmin=-30, vmax=30, aspect="auto")
        for a in range(3):
            for b2 in range(3):
                x.text(b2, a, f"{H.values[a, b2]:.0f}\n({N.values[a, b2]:.0f}건)", ha="center", va="center", fontsize=9)
        x.set_xticks(range(3), H.columns); x.set_yticks(range(3), H.index)
        x.set_title(f"10:30 {leg} | {col.replace('%', '')} (수익률 %)")
fig.suptitle("10:30 진입 때 SPX가 일목 구름 위/안/아래 → 성과 (ATM, 만기 보유, 건수는 전체 기간)")
fig.tight_layout(); fig.savefig(OUT / "I_filter.png", dpi=115); plt.close(fig)
# ── 후보: 오른 날 + 60분봉 구름 위 → ATM 콜, 누적 손익 비교 ──
h = L[L.r > 0].sort_index().copy()
h["pnl"] = h.call_pay - h.call_ask - FEE - np.where(h.call_pay > 0, EXERCISE, 0)
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
yr_rows = []
for name, g, col in (("오른 날 콜 전체", h, "grey"), ("+ 60분봉 구름 위", h[h["60분"] == "구름 위"], "C2"),
                     ("+ 60분봉 구름 위 아님", h[h["60분"] != "구름 위"], "C3")):
    ax[0].plot(g.index, (g.pnl * 100).cumsum() / 1000, color=col, label=f"{name} ({len(g)}건)")
    yr_rows.append(g.groupby(g.index.year).apply(lambda x: x.pnl.sum() / x.call_ask.sum() * 100, include_groups=False).rename(name))
ax[0].axhline(0, color="k", lw=1); ax[0].axvline(ODD, color="orange", ls=":", lw=1)
ax[0].text(ODD, ax[0].get_ylim()[1] * .9, " 4/9 관세 유예", color="orange", fontsize=8)
ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("10:30 오른 날 ATM 콜 1개, 만기 보유: 누적 손익 (천 달러)")
Y = pd.concat(yr_rows, axis=1)
Y.plot.bar(ax=ax[1], rot=0, color=["grey", "C2", "C3"]); ax[1].axhline(0, color="k", lw=1)
ax[1].set_title("연도별 수익률 %"); ax[1].set_xlabel("")
fig.tight_layout(); fig.savefig(OUT / "I_candidate.png", dpi=115); plt.close(fig)

pd.set_option("display.width", 200)
print(R.to_string(index=False))
print("\n연도별:\n", Y.round(1).to_string())
print(f"\n시도한 칸: 매수 2 × 봉 3 × 위치 3 = 18")
