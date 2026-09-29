"""
기술주 약세장에서도 통하나? (국면 점검)
  국면 = MNQ 일봉 종가(16:00 ET)가 200일 이동평균 위(강세) / 아래(약세) — 전날 기준 (미래 정보 없음)
  ① 옵션 전략(2022-05~) 국면별 성과, 2022년 월별
  ② 선물 대리 검증(2020-01~): 신호(ES 10:00 > ES 09:30 시가, MNQ 5분 구름 위) 난 날 ES 10:00 → 16:00 움직임
     = 콜이 이기려면 필요한 '오후 상승'이 신호 날에 더 자주 나오나. 국면·연도별.
결과: output/G1_regime.png, G_regime.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, load_es, signal_grids_ext, mnq_bars, _events
from spx0dte.strategy import features_10, trade_detail

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
mnq, es = load_mnq()[0], load_es()[0]


def at(m, hh, mm, col="close"):
    s = m[col][(m.index.hour == hh) & (m.index.minute == mm)]
    s.index = s.index.tz_localize(None).normalize()
    return s[~s.index.duplicated()]


# 국면: MNQ 15:59 봉 종가의 200일 평균, 전날 값 사용
nq_close = at(mnq, 15, 59)
regime = (nq_close > nq_close.rolling(200).mean()).shift(1).map({True: "강세 (200일선 위)", False: "약세 (200일선 아래)"})
nq_dd = (nq_close / nq_close.cummax() - 1).shift(1) * 100            # 전날 기준 나스닥 고점 대비 하락률

# ── ① 옵션 전략 ──
FP = pd.read_pickle(OUT / "f_paths.pkl")
X = features_10(FP, signal_grids_ext(mnq, FP), load_spx_ohlc(SPX_CSV))
L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date")
L = L[(L.leg == "C") & (L.offset == 0)].join(X, how="inner")
S = L[L.signal].sort_index()
T = pd.DataFrame([trade_detail(r, averaging=False) for r in S.itertuples()], index=S.index)
T["pnl$"] = T.pnl * 100
T["regime"] = regime.reindex(T.index)
T["nq_dd"] = nq_dd.reindex(T.index)
days_reg = regime.reindex(FP.index).value_counts()
opt = T.groupby("regime").apply(lambda g: pd.Series({"거래": len(g), "거래일 대비 신호 비율%": len(g) / days_reg[g.name] * 100,
                                                     "수익률%": g.pnl.sum() / g.cost.sum() * 100, "승률%": (g.pnl > 0).mean() * 100,
                                                     "총손익$": g["pnl$"].sum(), "건당 평균$": g["pnl$"].mean()}), include_groups=False)
T["dd_bin"] = pd.cut(T.nq_dd, [-100, -20, -10, -5, 0.01], labels=["고점 대비 −20% 이하", "−10~−20%", "−5~−10%", "−5% 이내"])
ddt = T.groupby("dd_bin", observed=False).apply(lambda g: pd.Series({"거래": len(g), "수익률%": g.pnl.sum() / g.cost.sum() * 100 if len(g) else np.nan,
                                                                    "총손익$": g["pnl$"].sum()}), include_groups=False)
m22 = T[T.index.year == 2022].groupby(T[T.index.year == 2022].index.month).agg(거래=("pnl$", "size"), 손익=("pnl$", "sum"))
nq_m22 = nq_close[nq_close.index.year == 2022].resample("ME").last().pct_change() * 100

# ── ② 선물 대리 검증 2020~ ──
b5 = mnq_bars(mnq, 5); e5 = _events(b5); e5.index = b5.index + pd.Timedelta(minutes=5)
c5 = e5.px > e5.top                                                    # 5분봉 종가 > 구름 윗선 (10:00 시점 = 09:55 봉)
c5_10 = c5[(c5.index.hour == 10) & (c5.index.minute == 0)]; c5_10.index = c5_10.index.tz_localize(None).normalize()
es_o, es_10, es_c = at(es, 9, 30, "open"), at(es, 9, 59), at(es, 15, 59)
F = pd.DataFrame({"up": es_10 > es_o, "cloud": c5_10, "aft": (es_c / es_10 - 1) * 100}).dropna()
F["regime"] = regime.reindex(F.index)
F["grp"] = np.where(F.up & F.cloud, "신호 (30분 상승 + 구름 위)", np.where(F.up, "30분 상승만", "30분 하락"))
F = F[F.index.dayofweek < 5]
big = lambda s: (s > 0.5).mean() * 100
fut = F.groupby(["regime", "grp"]).aft.agg(일수="size", 오후평균pct="mean", 오후_0_5pct이상_상승비율=big).round(3)
fut_y = F.groupby([F.index.year, "grp"]).aft.agg(일수="size", 오후평균pct="mean", 크게오른비율=big).round(3)

# ── 차트 ──
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
x = ax[0]
x.plot(nq_close.index, nq_close.values, color="#17201C", lw=.8)
x.plot(nq_close.index, nq_close.rolling(200).mean().values, color="#C77B2B", lw=1, label="200일 평균")
bear = regime.reindex(nq_close.index) == "약세 (200일선 아래)"
x.fill_between(nq_close.index, nq_close.min(), nq_close.max(), where=bear.values, color="#B3261E", alpha=.08, label="약세 국면")
x.axvline(pd.Timestamp("2022-05-16"), color="#0F6E5A", ls="--", lw=1); x.text(pd.Timestamp("2022-05-16"), nq_close.max() * .95, " 옵션 데이터 시작", color="#0F6E5A", fontsize=8)
x.set_title("MNQ(나스닥) 일봉과 약세 국면 (역조정 가격)"); x.legend(frameon=False, fontsize=8)
x = ax[1]
T2 = T.copy(); T2["cum"] = T2["pnl$"].cumsum() / 1000
x.plot(T2.index, T2.cum, color="#0F6E5A")
for d, r in T2.iterrows():
    if r.regime == "약세 (200일선 아래)":
        x.axvspan(d, d + pd.Timedelta(days=1), color="#B3261E", alpha=.15)
x.axhline(0, color="k", lw=.8); x.set_title("전략 누적 손익 (천 달러), 빨간 줄 = 약세 국면에서 난 매매")
x = ax[2]
g = F.groupby(["regime", "grp"]).aft.apply(big).unstack()
g = g[["신호 (30분 상승 + 구름 위)", "30분 상승만", "30분 하락"]]
g.plot.bar(ax=x, rot=0, color=["#0F6E5A", "#9AA39E", "#B3261E"])
x.set_ylabel("10:00→16:00 ES가 0.5% 넘게 오른 날 비율 %"); x.set_xlabel("")
x.set_title("선물 대리 검증 (2020-01~): 신호 날 오후에 크게 오르는 비율")
x.legend(frameon=False, fontsize=8)
for a_ in ax:
    a_.grid(alpha=.25)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "G1_regime.png", dpi=120); plt.close(fig)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("① 옵션 전략, 국면별:\n", opt.round(1).to_string())
print("\n나스닥 고점 대비 하락률별:\n", ddt.round(1).to_string())
print("\n2022년 월별 (옵션 전략 vs 나스닥 월 수익률):\n", m22.join(nq_m22.rename("나스닥 월%").set_axis(nq_m22.index.month)).round(1).to_string())
print("\n② 선물 대리 검증, 국면별:\n", fut.to_string())
print("\n연도별 (2020~2022 중심):\n", fut_y.loc[[2020, 2021, 2022]].to_string())
