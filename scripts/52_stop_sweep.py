"""
기준 전략 손절선 비교, 2022-05-16 ~ 2026-09-23.
타이밍: cbbo-1m의 '10:00' 호가 = 정확히 10:00:00.000 시점 스냅샷 (= 09:59 분이 끝난 순간).
  조건1 = SPX 10:00:00 선도가격 > 09:31:00, 조건2 = MNQ 09:59 봉(09:59:59 마감) 종가 > 5분봉 구름 윗선 → 둘 다 10:00:00까지 정보.
  체결은 10:00:00 호가 (지연 0초). 60초 지연은 scripts/53_rule_v2.py.
손절: 진입 후 매분 중간가가 매수가 × (1 − 손절%) 이하가 되면 그 분의 매수호가(bid)로 매도. 안 걸리면 만기 정산.
매수가: 중간가(기준) / 매도호가 두 가지. 수수료 매수·매도 각 0.025pt, 만기 내가격 정산 0.025pt.
결과: output/stops/stop_sweep.csv, stop_yearly.csv, S1_stop_sweep.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10
from spx0dte.exits import FEE, EXERCISE

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
R = OUT / "stops"; R.mkdir(exist_ok=True)
ODD = pd.Timestamp("2025-04-09")
STOPS = [None, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
L = pd.read_pickle(OUT / "legs10_full.pkl")
L = L[(L.leg == "C") & (L.offset == 0) & (L.entry == "10:00")].set_index("date").join(X[["signal"]], how="inner")
L = L[L.signal].sort_index()
print(f"매수일 {len(L)} ({L.index.min():%Y-%m-%d} ~ {L.index.max():%Y-%m-%d})")


def trade(r, sl, fill):
    cost = r.mid if fill == "중간가" else r.ask
    if sl is not None:
        hit = np.flatnonzero(np.asarray(r.mid_path, float) <= cost * (1 - sl) + 1e-6)
        if len(hit):
            i = hit[0]; b = float(r.bid_path[i])
            return (b - cost - 2 * FEE) * 100, True, i, cost
    return (r.pay - cost - FEE - (EXERCISE if r.pay > 0 else 0)) * 100, False, -1, cost


rows, yearly, per = [], [], {}
for fill in ("중간가", "매도호가"):
    for sl in STOPS:
        v = [trade(r, sl, fill) for r in L.itertuples()]
        p = pd.Series([x[0] for x in v], index=L.index); st = np.array([x[1] for x in v]); c = np.array([x[3] for x in v]) * 100
        nm = "손절 없음" if sl is None else f"-{int(sl * 100)}%"
        per[(fill, nm)] = p
        eq = p.cumsum(); hold = L.pay.values * 100 - c                     # 만기까지 들고 있었으면 (수수료 전)
        rows.append({"매수가": fill, "손절": nm, "총손익$": p.sum(), "4/9 빼고$": p.sum() - p.get(ODD, 0), "건당$": p.mean(),
                     "수익률%": p.sum() / c.sum() * 100, "최대낙폭$": (eq - eq.cummax()).min(), "최악의 날$": p.min(),
                     "손절 비율": st.mean(), "손절된 것 중 만기면 이익": (hold[st] > 0).mean() if st.any() else np.nan,
                     "잘린 이익$": hold[st & (hold > 0)].sum(), "플러스 연도": int((p.groupby(p.index.year).sum() > 0).sum())})
        yearly.append({"매수가": fill, "손절": nm, **{str(y): s for y, s in p.groupby(p.index.year).sum().items()}})
S = pd.DataFrame(rows); Y = pd.DataFrame(yearly)
S.to_csv(R / "stop_sweep.csv", index=False, encoding="utf-8-sig"); Y.to_csv(R / "stop_yearly.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(S.round(2).to_string(index=False)); print(); print(Y.round(0).to_string(index=False))

# 손절 걸린 분 분포 (−90%)
v = [trade(r, 0.9, "중간가") for r in L.itertuples()]
mins = pd.Series([x[2] + 1 for x in v if x[1]])
print(f"\n−90% 손절 걸린 시각 (10:00 이후 분): 중앙값 {mins.median():.0f}분, 14:00 전 {(mins < 240).mean():.0%}, 15:00 이후 {(mins >= 300).mean():.0%}")

fig, axs = plt.subplots(1, 3, figsize=(19, 5.2))
order = [n for n in S["손절"].unique()]
for fill, col in (("중간가", "#B23A48"), ("매도호가", "#3D5A80")):
    s = S[S["매수가"] == fill].set_index("손절").reindex(order)
    axs[0].plot(order, s["총손익$"] / 1000, "o-", color=col, label=f"{fill} 매수")
    axs[0].plot(order, s["4/9 빼고$"] / 1000, "o--", color=col, alpha=0.5, label=f"{fill} 매수 (4/9 빼고)")
    axs[1].plot(order, s["최대낙폭$"] / 1000, "o-", color=col, label=fill)
axs[0].axhline(0, color="k", lw=0.6); axs[0].set_title("총손익 (천 달러) — 손절선별"); axs[0].legend(fontsize=8)
axs[1].set_title("최대낙폭 (천 달러)"); axs[1].legend(fontsize=8)
s = S[S["매수가"] == "중간가"].set_index("손절").reindex(order)
axs[2].bar(order, s["손절 비율"] * 100, color="#9AA3AD", label="손절 걸린 비율 %")
axs[2].plot(order, s["손절된 것 중 만기면 이익"] * 100, "o-", color="#B23A48", label="손절된 것 중 만기까지 뒀으면 이익 %")
axs[2].set_title("손절 비율과 '잘린 대박'"); axs[2].legend(fontsize=8)
for ax in axs:
    ax.tick_params(axis="x", rotation=45)
fig.suptitle(f"S1 손절선 비교 (10:00:00 판단·체결): ATM 콜 {len(L)}건, 2022-05 ~ 2026-09 (중간가로 손절 판정 → 매수호가로 매도)")
fig.tight_layout(); fig.savefig(R / "S1_stop_sweep.png", dpi=110)
