"""
기준 전략 월별 성과 (기본 2026년). 몇 분.
  python scripts/27_monthly.py [--year 2026]
결과: output/M_monthly_{년}.png, M_monthly_{년}.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_detail

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, default=2026); a = ap.parse_args()

FP = pd.read_pickle(OUT / "f_paths.pkl")
px = load_spx_ohlc(SPX_CSV)
X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date").join(X, how="inner")
L = L[(L.leg == "C") & (L.index.year == a.year)]
days = FP.index[FP.index.year == a.year]
VARS = {"ATM 1계약": (0, False), "+10pt 1계약": (10, False), "+10pt 물타기": (10, True)}

T = {}
for name, (off, avg) in VARS.items():
    sub = L[(L.offset == off) & L.signal]
    T[name] = pd.DataFrame([trade_detail(r, averaging=avg) for r in sub.itertuples()], index=sub.index)

mon = pd.period_range(f"{a.year}-01", days.max(), freq="M")
rows = []
for m in mon:
    row = {"월": str(m), "거래일": int((days.to_period("M") == m).sum())}
    spx = px.close[(px.index.to_period("M") == m)]
    prev = px.close[px.index < pd.Timestamp(m.start_time)]
    row["SPX 월 수익률%"] = (spx.iloc[-1] / prev.iloc[-1] - 1) * 100 if len(spx) and len(prev) else np.nan
    for name, t in T.items():
        g = t[t.index.to_period("M") == m]
        k = name.split()[0] + (" 물타기" if "물타기" in name else "")
        row[f"{k} 거래"] = len(g)
        row[f"{k} 승"] = int((g.pnl > 0).sum())
        row[f"{k} 손익$"] = g.pnl.sum() * 100
        row[f"{k} 수익률%"] = g.pnl.sum() / g.cost.sum() * 100 if len(g) else np.nan
        row[f"{k} 최고$"] = g.pnl.max() * 100 if len(g) else np.nan
    rows.append(row)
M = pd.DataFrame(rows)
M.round(1).to_csv(OUT / f"M_monthly_{a.year}.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
x = np.arange(len(M)); w = .27
for j, (k, col) in enumerate((("ATM", "#0F6E5A"), ("+10pt", "#3A86A8"), ("+10pt 물타기", "#C77B2B"))):
    v = M[f"{k} 손익$"] / 1000
    ax[0].bar(x + (j - 1) * w, v, w, color=col, label=k)
    ax[1].plot(x, v.cumsum(), color=col, marker="o", label=k)
for i, r in M.iterrows():
    ax[0].text(i, ax[0].get_ylim()[0], f"{int(r['ATM 거래'])}건\n승 {int(r['ATM 승'])}", ha="center", va="bottom", fontsize=8, color="#5E6A64")
ax[0].axhline(0, color="k", lw=.8); ax[1].axhline(0, color="k", lw=.8)
ax[0].set_ylabel("월 손익 (천 달러)"); ax[1].set_ylabel("누적 (천 달러)")
ax[0].set_title(f"{a.year}년 월별 손익 (1계약 기준, 중간가 체결, 수수료 포함) — 막대 아래: ATM 거래 수 / 이긴 수")
ax[1].set_xticks(x, [f"{m[5:]}월\nSPX {s:+.1f}%" for m, s in zip(M["월"], M["SPX 월 수익률%"])])
for a_ in ax:
    a_.legend(fontsize=8, frameon=False); a_.grid(alpha=.25)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / f"M_monthly_{a.year}.png", dpi=130); plt.close(fig)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(M.round(1).to_string(index=False))
for name, t in T.items():
    eq = (t.pnl * 100).cumsum()
    print(f"{name}: {len(t)}건, 승 {(t.pnl > 0).sum()}, 손익 ${t.pnl.sum() * 100:,.0f}, 수익률 {t.pnl.sum() / t.cost.sum() * 100:.1f}%, "
          f"최대낙폭 ${(eq - eq.cummax()).min():,.0f}, 최고 3건 ${t.pnl.nlargest(3).sum() * 100:,.0f} ({', '.join(f'{d:%m-%d}' for d in t.pnl.nlargest(3).index)})")
