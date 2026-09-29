"""
왜 약세 국면에서 콜 매수가 더 버나 — 신호 날(10:00 ATM 콜)만 놓고 국면별로 분해.
  옵션값(프리미엄, IV) vs 실제 움직임(10:00 → 종가), 방향(오후 상승 확률), 큰 승리 빈도, 4/9 제외
결과: output/G2_why.png, G_why.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_detail

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
ODD = pd.Timestamp("2025-04-09")
mnq = load_mnq()[0]
c = mnq.close[(mnq.index.hour == 15) & (mnq.index.minute == 59)]; c.index = c.index.tz_localize(None).normalize()
c = c[~c.index.duplicated()]
regime = (c > c.rolling(200).mean()).shift(1).map({True: "강세", False: "약세"})

FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
X = features_10(FP, signal_grids_ext(mnq, FP), px)
L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date")
L = L[(L.leg == "C") & (L.offset == 0)].join(X, how="inner")
S = L[L.signal].sort_index()
T = pd.DataFrame([trade_detail(r, averaging=False) for r in S.itertuples()], index=S.index)
daily = pd.read_csv(OUT / "daily_features.csv", parse_dates=["date"]).set_index("date")
T = T.assign(regime=regime.reindex(T.index), K=S.K, F10=S.f10, close=px.close.reindex(T.index),
             iv10=daily["iv_10:00"].reindex(T.index))
T["prem_pct"] = T.cost / T.F10 * 100                                # 콜 가격 = SPX의 몇 %
T["move"] = (T.close / T.F10 - 1) * 100                             # 10:00 → 종가 (%)
T["absmove"] = T.move.abs()
T["mult"] = T.pay / T.cost                                          # 정산 ÷ 매수가 (배수)
T = T.dropna(subset=["regime"])


def summ(g):
    return pd.Series({
        "매매": len(g),
        "콜 가격 (SPX 대비 %)": g.prem_pct.mean(),
        "10시 IV (연율 %)": g.iv10.mean(),
        "10시→종가 평균 움직임 크기 %": g.absmove.mean(),
        "움직임 ÷ 콜 가격": g.absmove.mean() / g.prem_pct.mean(),
        "오후 상승 비율 %": (g.move > 0).mean() * 100,
        "오후 평균 변화 %": g.move.mean(),
        "2배 이상 비율 %": (g.mult >= 2).mean() * 100,
        "5배 이상 비율 %": (g.mult >= 5).mean() * 100,
        "0 (전액 손실) 비율 %": (g.pay <= 0).mean() * 100,
        "수익률 %": g.pnl.sum() / g.cost.sum() * 100,
        "건당 평균 $": g.pnl.mean() * 100,
    })


R = pd.concat({"전체": T.groupby("regime").apply(summ, include_groups=False).T,
               "4/9 제외": T.drop(ODD, errors="ignore").groupby("regime").apply(summ, include_groups=False).T}, axis=1)
R.round(2).to_csv(OUT / "G_why.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
col = {"강세": "#3A86A8", "약세": "#B3261E"}
for rg, g in T.drop(ODD, errors="ignore").groupby("regime"):
    ax[0].hist(g.move, bins=40, range=(-3, 3), histtype="step", lw=1.8, density=True, color=col[rg], label=f"{rg} ({len(g)}건)")
ax[0].axvline(0, color="k", lw=.8); ax[0].set_xlabel("신호 날 10:00 → 종가 SPX 변화 %"); ax[0].legend(frameon=False)
ax[0].set_title("신호 날 오후 움직임 분포 (4/9 제외)")
k = ["콜 가격 (SPX 대비 %)", "10시→종가 평균 움직임 크기 %"]
v = R["4/9 제외"].loc[k]
v.plot.bar(ax=ax[1], rot=0, color=[col["강세"], col["약세"]]); ax[1].set_title("콜 값 vs 실제 움직임 (4/9 제외)")
ax[1].set_xticklabels(["콜 가격\n(SPX 대비 %)", "10시→종가\n움직임 크기 %"])
w = R["4/9 제외"].loc[["2배 이상 비율 %", "5배 이상 비율 %", "0 (전액 손실) 비율 %"]]
w.plot.bar(ax=ax[2], rot=0, color=[col["강세"], col["약세"]]); ax[2].set_title("결과 분포 (4/9 제외)")
ax[2].set_xticklabels(["2배 이상", "5배 이상", "전액 손실"])
for a_ in ax:
    a_.grid(alpha=.25)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "G2_why.png", dpi=120); plt.close(fig)
pd.set_option("display.width", 200)
print(R.round(2).to_string())
