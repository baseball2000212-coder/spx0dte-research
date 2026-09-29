"""
왜 MNQ 5분 구름이 ES보다 잘 나오나: 두 판정이 엇갈린 날 분석 (SPX 30분 상승일 기준, ATM 콜 1계약 만기).
  그룹: 둘 다 위 / MNQ만 위 / ES만 위 / 둘 다 아님
  + 개장 30분 동안 나스닥이 S&P보다 더 올랐나(기술주 주도) 여부
결과: output/E2_disagree.png, E2_disagree.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, load_es, signal_grids_ext
from spx0dte.strategy import features_10, trade_detail

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
ODD = pd.Timestamp("2025-04-09")
FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
mnq, es = load_mnq()[0], load_es()[0]
Gm, Ge = signal_grids_ext(mnq, FP), signal_grids_ext(es, FP)
Xm, Xe = features_10(FP, Gm, px), features_10(FP, Ge, px)


def r30(m):
    o = m.open[(m.index.hour == 9) & (m.index.minute == 30)]; c = m.close[(m.index.hour == 9) & (m.index.minute == 59)]
    o.index, c.index = o.index.tz_localize(None).normalize(), c.index.tz_localize(None).normalize()
    return (c / o - 1).reindex(FP.index)


D = pd.DataFrame({"spx_up": Xm.r30 > 0, "m": Xm.c5 == 1, "e": Xe.c5 == 1,
                  "nq30": r30(mnq), "es30": r30(es),
                  # 구름 위로 얼마나 떨어져 있나 (구름 윗선 대비 %, 0 이하면 구름 안/아래)
                  "m_dist": (Gm["px"]["10:00"] / Gm["top5"]["10:00"] - 1) * 100,
                  "e_dist": (Ge["px"]["10:00"] / Ge["top5"]["10:00"] - 1) * 100})
D["tech_lead"] = D.nq30 > D.es30
D["grp"] = np.select([D.m & D.e, D.m & ~D.e, ~D.m & D.e], ["둘 다 위", "MNQ만 위", "ES만 위"], "둘 다 아님")
L = pd.read_pickle(OUT / "legs10_full.pkl"); L = L[(L.leg == "C") & (L.offset == 0)].set_index("date")
L = L[D.spx_up.reindex(L.index).fillna(False).astype(bool)]
t = pd.DataFrame([trade_detail(r, averaging=False) for r in L.itertuples()], index=L.index)
t = t.join(D)
t["pnl$"] = t.pnl * 100

rows = []
for g in ("둘 다 위", "MNQ만 위", "ES만 위", "둘 다 아님"):
    h = t[t.grp == g]
    rows.append({"그룹": g, "날 수": len(h), "수익률%": h.pnl.sum() / h.cost.sum() * 100, "승률%": (h.pnl > 0).mean() * 100,
                 "총손익$": h["pnl$"].sum(), "4/9 빼고$": h["pnl$"].sum() - h["pnl$"].get(ODD, 0),
                 "최고 3건 빼고$": h["pnl$"].sum() - h["pnl$"].nlargest(3).sum(),
                 "나스닥이 더 오른 날%": h.tech_lead.mean() * 100,
                 "연도별$": {k: round(v) for k, v in h["pnl$"].groupby(h.index.year).sum().items()},
                 "최고 3일": ", ".join(f"{d:%Y-%m-%d}(${v:,.0f})" for d, v in h["pnl$"].nlargest(3).items())})
R = pd.DataFrame(rows)
R.round(1).to_csv(OUT / "E2_disagree.csv", index=False, encoding="utf-8-sig")

# 기술주 주도 여부로 나누면?
tl = []
for g in ("둘 다 위", "MNQ만 위", "ES만 위"):
    for lead in (True, False):
        h = t[(t.grp == g) & (t.tech_lead == lead)]
        tl.append({"그룹": g, "개장 30분": "나스닥이 더 오름" if lead else "S&P가 더 오름", "날 수": len(h),
                   "수익률%": h.pnl.sum() / h.cost.sum() * 100 if len(h) else np.nan, "총손익$": h["pnl$"].sum()})
TL = pd.DataFrame(tl)
# 전체 SPX 상승일에서 기술주 주도 여부만으로
tl_all = t.groupby("tech_lead").apply(lambda h: pd.Series({"날 수": len(h), "수익률%": h.pnl.sum() / h.cost.sum() * 100, "총손익$": h["pnl$"].sum()}), include_groups=False)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
for g, c in (("둘 다 위", "#0F6E5A"), ("MNQ만 위", "#3A86A8"), ("ES만 위", "#C77B2B"), ("둘 다 아님", "#9AA39E")):
    h = t[t.grp == g].sort_index()
    ax[0].plot(h.index, h["pnl$"].cumsum() / 1000, color=c, label=f"{g} ({len(h)}일)")
ax[0].axhline(0, color="k", lw=.8); ax[0].legend(frameon=False); ax[0].grid(alpha=.25)
ax[0].set_title("SPX 30분 상승일 ATM 콜: 5분 구름 판정 그룹별 누적 손익 (천 달러)")
sub = t[t.grp.isin(["MNQ만 위", "ES만 위"])]
for g, c in (("MNQ만 위", "#3A86A8"), ("ES만 위", "#C77B2B")):
    h = sub[sub.grp == g]
    ax[1].scatter(h.nq30 * 100 - h.es30 * 100, h.pnl / h.cost * 100, s=14, alpha=.6, color=c, label=g)
ax[1].axhline(0, color="k", lw=.8); ax[1].axvline(0, color="k", lw=.8)
ax[1].set_xlabel("개장 30분: 나스닥 수익률 − S&P 수익률 (%p)"); ax[1].set_ylabel("콜 수익률 %"); ax[1].set_ylim(-110, 400)
ax[1].legend(frameon=False); ax[1].set_title("엇갈린 날: 기술주가 더 올랐는지 vs 콜 수익")
for a_ in ax:
    for s in ("top", "right"): a_.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "E2_disagree.png", dpi=120); plt.close(fig)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 80)
print(R.round(1).to_string(index=False))
print("\n기술주 주도 여부로 나누면:\n", TL.round(1).to_string(index=False))
print("\nSPX 30분 상승일 전체, 나스닥이 더 오른 날 vs 아닌 날:\n", tl_all.round(1).to_string())
print("\n구름 윗선과의 거리 중앙값(%): MNQ", round(D.m_dist.median(), 3), " ES", round(D.e_dist.median(), 3),
      " | MNQ 구름 두께 대비 가격 변동이 큰지: MNQ 5분봉 1봉 평균 변동%", round((mnq.close.pct_change().abs().mean()) * 100 * np.sqrt(5), 3),
      " ES", round((es.close.pct_change().abs().mean()) * 100 * np.sqrt(5), 3))
