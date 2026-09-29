"""
사용자 물타기 규칙 비교: 1개 → -50%에 1개 추가(평단 75%) → 첫 매수가 37.5%(평단 반토막)면 전부 손절, 아니면 만기.
비교: 1계약 / 처음부터 2계약 / 기존 물타기(-40·-70%) / 새 규칙(손절 37.5%) / 새 규칙(손절 25%) / 새 규칙(손절 없음)
결과: output/L1_ladder.png, L_ladder.csv, L_ladder_yearly.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_detail, trade_ladder

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
SPLIT, ODD = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-09")

FP = pd.read_pickle(OUT / "f_paths.pkl")
X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), load_spx_ohlc(SPX_CSV))
L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date").join(X, how="inner")
L = L[(L.leg == "C") & L.signal]

PLANS = {
    "1계약": lambda r, f: trade_detail(r, averaging=False, fill=f),
    "처음부터 2계약": lambda r, f: {**(d := trade_detail(r, averaging=False, fill=f)), "pnl": d["pnl"] * 2, "cost": d["cost"] * 2, "n": 2, "stop": None},
    "기존 물타기(-40·-70%)": lambda r, f: trade_detail(r, averaging=True, fill=f),
    "새 규칙: -50% 추가, 37.5% 손절": lambda r, f: trade_ladder(r, 0.5, 0.375, f),
    "새 규칙: -50% 추가, 25% 손절": lambda r, f: trade_ladder(r, 0.5, 0.25, f),
    "새 규칙: -50% 추가, 손절 없음": lambda r, f: trade_ladder(r, 0.5, None, f),
}
rows, yearly, curves = [], [], {}
for off in (0, 10):
    sub = L[L.offset == off]
    for name, fn in PLANS.items():
        for fill in ("중간가", "매도호가"):
            t = pd.DataFrame([fn(r, fill) for r in sub.itertuples()], index=sub.index).sort_index()
            pnl = t.pnl * 100; eq = pnl.cumsum(); dd = eq - eq.cummax()
            ret = lambda m: t.pnl[m].sum() / t.cost[m].sum() * 100
            y26 = t.index.year == 2026
            key = f"{'ATM' if off == 0 else '+10pt'} · {name}"
            rows.append({"행사가": "ATM" if off == 0 else "+10pt", "운용": name, "체결": fill,
                         "전체 손익$": pnl.sum(), "4/9 빼고$": pnl.sum() - pnl.get(ODD, 0), "최고 3건 빼고$": pnl.sum() - pnl.nlargest(3).sum(),
                         "2026 손익$": pnl[y26].sum(), "최대낙폭$": dd.min(), "손익÷낙폭": pnl.sum() / -dd.min(),
                         "최악의 날$": pnl.min(), "하루 최대 투입$": (t.cost * 100).max(), "평균 투입$": (t.cost * 100).mean(),
                         "수익률%": t.pnl.sum() / t.cost.sum() * 100, "승률%": (t.pnl > 0).mean() * 100,
                         "추가 매수%": (t.n > 1).mean() * 100 if "2계약" not in name else np.nan,
                         "손절%": t["stop"].notna().mean() * 100 if "stop" in t else 0,
                         "학습%(참고)": ret(t.index < SPLIT), "검증%(참고)": ret(t.index >= SPLIT),
                         "플러스 연도": int((pnl.groupby(t.index.year).sum() > 0).sum())})
            if fill == "중간가":
                yearly.append(pnl.groupby(t.index.year).sum().rename(key))
                curves[key] = eq
R = pd.DataFrame(rows)
R.round(1).to_csv(OUT / "L_ladder.csv", index=False, encoding="utf-8-sig")
Y = pd.concat(yearly, axis=1).T.round(0)
Y.to_csv(OUT / "L_ladder_yearly.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(1, 2, figsize=(18, 5.8))
col = {"1계약": "#9AA39E", "처음부터 2계약": "#3A86A8", "기존 물타기(-40·-70%)": "#C77B2B",
       "새 규칙: -50% 추가, 37.5% 손절": "#0F6E5A", "새 규칙: -50% 추가, 25% 손절": "#7B2CBF", "새 규칙: -50% 추가, 손절 없음": "#B3261E"}
for j, off in enumerate(("ATM", "+10pt")):
    for name, c in col.items():
        e = curves[f"{off} · {name}"]
        ax[j].plot(e.index, e.values / 1000, color=c, lw=2 if "37.5" in name else 1.2, label=name)
    ax[j].axhline(0, color="k", lw=.8); ax[j].axvline(ODD, color="#B3261E", ls=":", lw=.8)
    ax[j].set_title(f"{off} 콜: 누적 손익 (천 달러, 중간가 체결)"); ax[j].legend(fontsize=8, frameon=False); ax[j].grid(alpha=.25)
    for s in ("top", "right"): ax[j].spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "L1_ladder.png", dpi=120); plt.close(fig)

pd.set_option("display.width", 280); pd.set_option("display.max_columns", 30)
show = ["행사가", "운용", "전체 손익$", "4/9 빼고$", "최고 3건 빼고$", "2026 손익$", "최대낙폭$", "손익÷낙폭", "최악의 날$",
        "하루 최대 투입$", "승률%", "추가 매수%", "손절%", "플러스 연도", "학습%(참고)", "검증%(참고)"]
print(R[R.체결 == "중간가"][show].round(1).to_string(index=False))
print("\n매도호가 체결:\n", R[R.체결 == "매도호가"][["행사가", "운용", "전체 손익$", "2026 손익$", "최대낙폭$"]].round(0).to_string(index=False))
print("\n연도별 (중간가):\n", Y.to_string())
