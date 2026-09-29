"""
기준 전략을 MNQ 대신 ES(S&P500 선물) 기준으로: 10:00 판단, ATM·+10pt 콜 1계약, 만기 보유, 중간가 체결.
  A 기존: SPX 개장 30분 상승 + MNQ 5분 구름 위
  B     : SPX 개장 30분 상승 + ES 5분 구름 위
  C     : ES 개장 30분 상승 + ES 5분 구름 위     (ES 09:30 봉 시가 → 09:59 봉 종가)
  D     : SPX 30분 상승 + MNQ·ES 5분 둘 다 구름 위
결과: output/E_es_compare.png, E_es_monthly_2026.png, E_es_compare.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc, NY
from spx0dte.touch import load_mnq, load_es, signal_grids_ext
from spx0dte.strategy import features_10, trade_detail

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
ODD = pd.Timestamp("2025-04-09")

FP = pd.read_pickle(OUT / "f_paths.pkl")
px = load_spx_ohlc(SPX_CSV)
es, nroll = load_es()
print(f"ES 월물 교체 {nroll}번 보정")
Xm = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
Xe = features_10(FP, signal_grids_ext(es, FP), px)
# ES 개장 30분: 09:30 봉 시가 → 09:59 봉 종가
eo = es.open[(es.index.hour == 9) & (es.index.minute == 30)]
ec = es.close[(es.index.hour == 9) & (es.index.minute == 59)]
eo.index, ec.index = eo.index.tz_localize(None).normalize(), ec.index.tz_localize(None).normalize()
r30_es = (ec / eo - 1).reindex(FP.index)
COND = {"A 기존 (SPX 30분 + MNQ 5분)": (Xm.r30 > 0) & (Xm.c5 == 1),
        "B SPX 30분 + ES 5분": (Xm.r30 > 0) & (Xe.c5 == 1),
        "C ES 30분 + ES 5분": (r30_es > 0) & (Xe.c5 == 1),
        "D SPX 30분 + MNQ·ES 5분 둘 다": (Xm.r30 > 0) & (Xm.c5 == 1) & (Xe.c5 == 1)}
agree = ((Xm.c5 == 1) == (Xe.c5 == 1)).mean() * 100
print(f"MNQ 5분 구름 위 vs ES 5분 구름 위 판정 일치율: {agree:.1f}%")

L = pd.read_pickle(OUT / "legs10_full.pkl")
L = L[L.leg == "C"].set_index("date")
rows, curves, monthly = [], {}, {}
for off in (0, 10):
    base = L[L.offset == off]
    for name, cond in COND.items():
        sub = base[cond.reindex(base.index).fillna(False).astype(bool)]
        t = pd.DataFrame([trade_detail(r, averaging=False) for r in sub.itertuples()], index=sub.index).sort_index()
        pnl = t.pnl * 100; eq = pnl.cumsum(); dd = (eq - eq.cummax()).min()
        y = pnl.groupby(t.index.year).sum()
        key = f"{'ATM' if off == 0 else '+10pt'} · {name}"
        rows.append({"행사가": "ATM" if off == 0 else "+10pt", "조건": name, "거래": len(t), "평균 투입$": t.cost.mean() * 100,
                     "수익률%": t.pnl.sum() / t.cost.sum() * 100, "승률%": (t.pnl > 0).mean() * 100,
                     "총손익$": pnl.sum(), "4/9 빼고$": pnl.sum() - pnl.get(ODD, 0), "최고 3건 빼고$": pnl.sum() - pnl.nlargest(3).sum(),
                     "최대낙폭$": dd, "손익÷낙폭": pnl.sum() / -dd, "플러스 연도": int((y > 0).sum()),
                     **{f"{k}$": v for k, v in y.items()}})
        curves[key] = eq
        m26 = pnl[t.index.year == 2026]
        monthly[key] = m26.groupby(m26.index.month).sum()
R = pd.DataFrame(rows)
R.round(1).to_csv(OUT / "E_es_compare.csv", index=False, encoding="utf-8-sig")

cols_c = {"A": "#9AA39E", "B": "#3A86A8", "C": "#0F6E5A", "D": "#C77B2B"}
fig, ax = plt.subplots(1, 2, figsize=(18, 5.5), sharey=True)
for j, off in enumerate(("ATM", "+10pt")):
    for name in COND:
        e = curves[f"{off} · {name}"]
        ax[j].plot(e.index, e.values / 1000, color=cols_c[name[0]], lw=1.8 if name[0] != "A" else 1.2, label=name)
    ax[j].axhline(0, color="k", lw=.8); ax[j].axvline(ODD, color="#B3261E", ls=":", lw=.8)
    ax[j].set_title(f"{off} 콜 1계약: 누적 손익 (천 달러)"); ax[j].legend(fontsize=8, frameon=False); ax[j].grid(alpha=.25)
    for s in ("top", "right"): ax[j].spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "E_es_compare.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 4.8))
w = .2
for j, name in enumerate(COND):
    m = monthly[f"ATM · {name}"].reindex(range(1, 10)).fillna(0) / 1000
    ax.bar(np.arange(9) + (j - 1.5) * w, m.values, w, color=cols_c[name[0]], label=name)
ax.set_xticks(range(9), [f"{i}월" for i in range(1, 10)]); ax.axhline(0, color="k", lw=.8)
ax.set_title("2026년 월별 손익 (ATM 1계약, 천 달러)"); ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=.25)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "E_es_monthly_2026.png", dpi=120); plt.close(fig)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
print(R.round(0).to_string(index=False))
print("\n2026 월별 (ATM, $):\n", pd.DataFrame({k.split(" · ")[1][:1]: v for k, v in monthly.items() if k.startswith("ATM")}).round(0).to_string())
