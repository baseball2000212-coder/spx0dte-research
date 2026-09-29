"""
'개장 후 올랐으면 콜 매수' 행사가 비교: ATM vs 외가격 +5 ~ +30pt, 만기 보유. 16 결과 사용, 몇 초.
특이일(2025-04-09 관세 유예 +9.5%) 뺀 결과도 같이.
결과: output/O1_otm_calls.png, O1_otm_calls.csv
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
SPLIT, ODD = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-09")
ENTRY = ["09:45", "10:00", "10:30"]

O = pd.read_pickle(OUT / "otm_paths.pkl")
FP = pd.read_pickle(OUT / "f_paths.pkl")
f_open = FP[list(FP.columns[:5])].bfill(axis=1).iloc[:, 0]
O["r"] = O.F0.values / f_open.reindex(O.date).values - 1
O["pnl"] = O.pay - O.ask - FEE - np.where(O.pay > 0, EXERCISE, 0)
ret = lambda g: g.pnl.sum() / g.ask.sum() * 100 if len(g) else np.nan

rows = []
for (e, leg, off), g in O[O.entry.isin(ENTRY)].groupby(["entry", "leg", "offset"]):
    for cond, h in (("신호일", g[(g.r > 0) if leg == "C" else (g.r < 0)]), ("매일", g)):
        h = h.sort_values("date"); eq = (h.pnl * 100).cumsum()
        v = h[h.date >= SPLIT]
        rows.append({"진입": e, "종류": "콜" if leg == "C" else "풋", "외가격pt": off, "조건": cond, "건수": len(h),
                     "평균 프리미엄$": h.ask.mean() * 100, "학습%": ret(h[h.date < SPLIT]), "검증%": ret(v),
                     "검증(4/9 제외)%": ret(v[v.date != ODD]), "최고 10일 빼면%": ret(h.drop(h.pnl.nlargest(10).index)),
                     "승률%": (h.pnl > 0).mean() * 100, "총손익$": eq.iloc[-1] if len(eq) else np.nan,
                     "4/9 손익$": h[h.date == ODD].pnl.sum() * 100, "최대낙폭$": (eq - eq.cummax()).min() if len(eq) else np.nan})
R = pd.DataFrame(rows).round(1)
R.to_csv(OUT / "O1_otm_calls.csv", index=False, encoding="utf-8-sig")

C = R[(R.종류 == "콜") & (R.조건 == "신호일")]
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for j, col in enumerate(["학습%", "검증%", "검증(4/9 제외)%", "최고 10일 빼면%"]):
    H = C.pivot(index="외가격pt", columns="진입", values=col)[ENTRY]
    ax[j].imshow(H.values, cmap="RdYlGn", vmin=-30, vmax=30, aspect="auto")
    for a in range(H.shape[0]):
        for b in range(H.shape[1]):
            ax[j].text(b, a, f"{H.values[a, b]:.0f}", ha="center", va="center", fontsize=10)
    ax[j].set_xticks(range(len(ENTRY)), ENTRY); ax[j].set_yticks(range(len(H)), [f"ATM" if o == 0 else f"+{o}pt" for o in H.index])
    ax[j].set_title(f"오른 날 콜 매수 | {col.replace('%', '')}")
fig.suptitle("행사가별 수익률 % (만기 보유, 수수료 포함)")
fig.tight_layout(); fig.savefig(OUT / "O1_otm_calls.png", dpi=115); plt.close(fig)

pd.set_option("display.width", 260)
cols = ["진입", "외가격pt", "건수", "평균 프리미엄$", "학습%", "검증%", "검증(4/9 제외)%", "최고 10일 빼면%", "승률%", "총손익$", "4/9 손익$", "최대낙폭$"]
print("오른 날 콜:\n", C[cols].to_string(index=False))
print("\n(비교) 매일 콜:\n", R[(R.종류 == "콜") & (R.조건 == "매일")][["진입", "외가격pt", "학습%", "검증%", "검증(4/9 제외)%"]].to_string(index=False))
print("\n(비교) 내린 날 풋:\n", R[(R.종류 == "풋") & (R.조건 == "신호일")][["진입", "외가격pt", "평균 프리미엄$", "학습%", "검증%", "검증(4/9 제외)%"]].to_string(index=False))
