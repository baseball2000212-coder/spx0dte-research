"""
계좌 크기 시뮬레이션 (기준 전략 = 10시 상승 + MNQ 5분 구름 위 → ATM 콜, 만기).
  시작 $100,000 (예시) (2022-05-16), 매 신호마다 '현재 계좌의 f%'만큼 첫 매수 (복리). SPX 1계약보다 작아서 XSP(1/10) 가정:
  XSP 1계약 = SPX 가격 × 10달러, 수수료 계약당 편도 $2.5 (미확인), 만기 내가격이면 정산 비용 $2.5.
  방식: 1계약형(물타기 없음) / -50%에 같은 수량 1번 추가, 손절 없음
결과: output/Z1_sizing.png, Z_sizing.csv
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
START, FEE_USD = 100000.0, 2.5

FP = pd.read_pickle(OUT / "f_paths.pkl")
X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), load_spx_ohlc(SPX_CSV))
L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date").join(X, how="inner")
L = L[(L.leg == "C") & (L.offset == 0) & L.signal].sort_index()
TR = {"물타기 없음": [trade_detail(r, averaging=False) for r in L.itertuples()],
      "-50% 1번 추가": [trade_ladder(r, 0.5, None) for r in L.itertuples()]}
days = FP.index[FP.index >= L.index.min()]

rows, curves = [], {}
for meth, trades in TR.items():
    for f in (0.01, 0.03, 0.05, 0.10):
        E, eq, skipped = START, {}, 0
        for d, t in zip(L.index, trades):
            p0 = t["buys"][0][1]
            n = int(np.floor(E * f / (p0 * 10)))                      # XSP 계약 수
            if n < 1:
                skipped += 1; eq[d] = E; continue
            pnl = 0.0
            for _, p in t["buys"]:
                pnl += n * ((t["pay"] - p) * 10 - FEE_USD - (FEE_USD if t["pay"] > 0 else 0))
            E = max(E + pnl, 0.0); eq[d] = E
        s = pd.Series(eq).reindex(days).ffill().fillna(START)
        peak = s.cummax(); dd = s / peak - 1
        halves = int(((dd <= -0.5) & (dd.shift(1) > -0.5)).sum())
        mon = s.resample("ME").last().pct_change().dropna()
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        under = (dd < 0).astype(int); run = under.groupby((under == 0).cumsum()).sum().max()
        rows.append({"방식": meth, "한 번에 거는 비율": f"{f:.0%}", "최종 계좌$": s.iloc[-1], "연평균 수익률%": ((s.iloc[-1] / START) ** (1 / yrs) - 1) * 100,
                     "최대낙폭%": dd.min() * 100, "반토막 횟수": halves, "가장 긴 물속 기간(거래일)": int(run),
                     "최악의 달%": mon.min() * 100, "마이너스 달 비율%": (mon < 0).mean() * 100,
                     "2026 수익률%": (s[s.index.year == 2026].iloc[-1] / s[s.index.year < 2026].iloc[-1] - 1) * 100,
                     "못 산 신호(돈 부족)": skipped})
        curves[(meth, f)] = s
Z = pd.DataFrame(rows)
Z.round(1).to_csv(OUT / "Z_sizing.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(1, 2, figsize=(18, 5.6), sharey=True)
for j, meth in enumerate(TR):
    for f, c in zip((0.01, 0.03, 0.05, 0.10), ("#9AA39E", "#3A86A8", "#0F6E5A", "#B3261E")):
        s = curves[(meth, f)]
        ax[j].plot(s.index, s.values / 1000, color=c, label=f"계좌의 {f:.0%}")
    ax[j].axhline(START / 1000, color="k", lw=.8, ls="--")
    ax[j].set_yscale("log"); ax[j].set_title(f"{meth}: 계좌 변화 (시작 $100k, 로그 눈금, 천 달러)")
    ax[j].legend(frameon=False); ax[j].grid(alpha=.25, which="both")
    for s_ in ("top", "right"): ax[j].spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "Z1_sizing.png", dpi=120); plt.close(fig)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(Z.round(1).to_string(index=False))
