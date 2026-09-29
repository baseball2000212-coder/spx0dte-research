"""
외부 비판(ChatGPT) 대응 검증. 기준: ATM 콜 1계약, 만기, 수수료 편도 $2.5 + 정산 $2.5.
  ① 타이밍: A 현재(SPX 10:00 호가로 판단·같은 분 매수) / B SPX 09:59로 판단, 10:00 매수 / C 10:00 판단, 10:01 매수 / D 09:59 판단, 10:01 매수
  ③ MNQ 단순 모멘텀 대조 (모두 SPX 30분 상승 조건 포함): 구름(현재) vs 마지막 5분봉 양봉 vs 09:30→09:59 상승 vs 전일 종가 위 vs 5분 20봉 평균 위 vs 기준선 위
  ⑦ 부트스트랩 확장 (평균·중앙값·연환산·P(평균≤0)·샤프·최대낙폭 분포)  ⑧ 상위 1·3·5·10건 제거  ⑨ 국면 구간별
결과: output/critique/*.png, critique.json
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext, mnq_bars
from spx0dte.strategy import features_10
from spx0dte.ichimoku import ichimoku
from spx0dte.exits import FEE, EXERCISE

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
D = OUT / "critique"; D.mkdir(exist_ok=True)
ODD = pd.Timestamp("2025-04-09"); RNG = np.random.default_rng(7); NSIM = 10_000
YEARS = 1093 / 252

FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
mnq = load_mnq()[0]
X = features_10(FP, signal_grids_ext(mnq, FP), px)
L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date")
L = L[(L.leg == "C") & (L.offset == 0)].join(X, how="inner").sort_index()


def pnl_at(r, when):                                                # 달러
    p = r.mid if when == "10:00" else float(r.mid_path[0])          # mid_path[0] = 10:01
    return (r.pay - p - FEE - (EXERCISE if r.pay > 0 else 0)) * 100, p


def summary(sel, when="10:00"):
    g = L[np.asarray(sel.reindex(L.index).fillna(False), bool)]
    v = [pnl_at(r, when) for r in g.itertuples()]
    p = pd.Series([x[0] for x in v], index=g.index); c = pd.Series([x[1] * 100 for x in v], index=g.index)
    eq = p.cumsum()
    return p, {"매수": len(p), "총손익$": p.sum(), "4/9 빼고$": p.sum() - p.get(ODD, 0), "건당$": p.mean(), "수익률%": p.sum() / c.sum() * 100,
               "최대낙폭$": (eq - eq.cummax()).min(), "플러스 연도": int((p.groupby(p.index.year).sum() > 0).sum()),
               **{str(y): v for y, v in p.groupby(p.index.year).sum().items()}}


R = {}
# ── ① 타이밍 ──
up10, up959 = X.r30 > 0, FP["09:59"] / X.f_open - 1 > 0
cloud = X.c5 == 1
tim = {"A 현재: 10:00 판단 = 10:00 매수": (up10 & cloud, "10:00"), "B 09:59 판단 → 10:00 매수": (up959 & cloud, "10:00"),
       "C 10:00 판단 → 10:01 매수": (up10 & cloud, "10:01"), "D 09:59 판단 → 10:01 매수": (up959 & cloud, "10:01")}
R["timing"] = {k: summary(s, w)[1] for k, (s, w) in tim.items()}

# ── ③ MNQ 단순 모멘텀 대조 ──
def at(m, hh, mm, col="close", shift_day=False):
    s = m[col][(m.index.hour == hh) & (m.index.minute == mm)]
    s.index = s.index.tz_localize(None).normalize(); s = s[~s.index.duplicated()]
    return s.reindex(FP.index) if not shift_day else s.shift(1).reindex(FP.index)
b5 = mnq_bars(mnq, 5); ic5 = ichimoku(b5)
t955 = b5.index[(b5.index.hour == 9) & (b5.index.minute == 55)]
def at5(s):
    s = s.loc[t955]; s.index = s.index.tz_localize(None).normalize(); return s[~s.index.duplicated()].reindex(FP.index)
c955, o955 = at5(b5.close), at5(b5.open)
sma20 = at5(b5.close.rolling(20).mean()); kijun = at5(ic5["기준선"])
m_open, m_959 = at(mnq, 9, 30, "open"), at(mnq, 9, 59)
prev_close = at(mnq, 15, 59)
prev_close = prev_close.shift(1)
MOM = {"구름 위 (현재)": cloud, "마지막 5분봉 양봉": c955 > o955, "MNQ 09:30→09:59 상승": m_959 > m_open,
       "MNQ 전일 종가 위": m_959 > prev_close, "5분 20봉 평균 위": c955 > sma20, "5분 기준선(26봉) 위": c955 > kijun,
       "(조건 없음) SPX 30분 상승만": pd.Series(True, index=FP.index)}
R["momentum"] = {}
for k, m in MOM.items():
    p, s = summary(up10 & m)
    R["momentum"][k] = s
R["momentum_overlap_with_cloud%"] = {k: float(((up10 & m) & (up10 & cloud)).sum() / max((up10 & m).sum(), 1) * 100) for k, m in MOM.items()}

# ── ⑦ 부트스트랩 확장 · ⑧ 꼬리 제거 ──
base, _ = summary(up10 & cloud)
boot = {}
for lab, v in (("4/9 포함", base.values), ("4/9 제외", base.drop(ODD).values)):
    n = len(v); idx = RNG.integers(0, n, (NSIM, n)); samp = v[idx]
    means = samp.mean(1); med = np.median(samp, 1)
    eqs = samp.cumsum(1); mdd = (eqs - np.maximum.accumulate(eqs, 1)).min(1)
    sharpe = means / samp.std(1) * np.sqrt(n / YEARS)                # 매매당 → 연환산 (연 {n/YEARS:.0f}건)
    boot[lab] = {"평균$": float(v.mean()), "평균 95%": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                 "중앙값$": float(np.median(v)), "중앙값 95%": [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
                 "연환산 손익$": float(v.mean() * n / YEARS), "연환산 95%": [float(np.percentile(means, 2.5) * n / YEARS), float(np.percentile(means, 97.5) * n / YEARS)],
                 "P(평균≤0)%": float((means <= 0).mean() * 100), "샤프(연환산)": float(v.mean() / v.std() * np.sqrt(n / YEARS)),
                 "샤프 95%": [float(np.percentile(sharpe, 2.5)), float(np.percentile(sharpe, 97.5))],
                 "최대낙폭 중앙값$": float(np.median(mdd)), "최대낙폭 5% 최악$": float(np.percentile(mdd, 5))}
R["bootstrap"] = boot
tail = {}
for k in (0, 1, 3, 5, 10, 20):
    q = base.drop(base.nlargest(k).index) if k else base
    tail[f"상위 {k}건 제거" if k else "전체"] = {"총손익$": float(q.sum()), "건당$": float(q.mean()), "제거된 날": [f"{d:%Y-%m-%d}" for d in base.nlargest(k).index]}
R["tail"] = tail

# ── ⑨ 국면 구간 ──
daily = pd.read_csv(OUT / "daily_features.csv", parse_dates=["date"]).set_index("date")
Z = pd.DataFrame({"pnl": base}).join(pd.DataFrame({
    "10시 IV": daily["iv_10:00"], "IV 전일 대비": daily["iv_10:00"] - daily["iv_10:00"].shift(1),
    "밤사이 갭%": daily.gap, "개장 30분 상승폭%": X.r30 * 100, "전일 실현변동성": daily.rv_day.shift(1)}), how="left")
reg = {}
for col in ["10시 IV", "IV 전일 대비", "밤사이 갭%", "개장 30분 상승폭%", "전일 실현변동성"]:
    z = Z.dropna(subset=[col])
    b = pd.qcut(z[col], 3, labels=["낮음", "중간", "높음"])
    g = z.groupby(b, observed=False).pnl
    reg[col] = {str(k): {"매수": int(g.size()[k]), "건당$": float(g.mean()[k]), "총손익$": float(g.sum()[k]),
                         "구간": f"{z[col][b == k].min():.2f} ~ {z[col][b == k].max():.2f}"} for k in ["낮음", "중간", "높음"]}
    zx = z.drop(ODD, errors="ignore"); bx = pd.qcut(zx[col], 3, labels=["낮음", "중간", "높음"])
    for k, v in zx.groupby(bx, observed=False).pnl.mean().items():
        reg[col][str(k)]["건당$ (4/9 제외)"] = float(v)
R["regime"] = reg
json.dump(R, open(D / "critique.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

# 차트: 모멘텀 대조 누적
fig, ax = plt.subplots(figsize=(10, 4))
for (k, m), c in zip(MOM.items(), ("#0F6E5A", "#3A86A8", "#C77B2B", "#7B2CBF", "#B3261E", "#6B7570", "#BBBBBB")):
    p, _ = summary(up10 & m)
    ax.plot(p.index, p.cumsum() / 1000, color=c, lw=2 if "현재" in k else 1.1, label=f"{k} ({len(p)}건)")
ax.axhline(0, color="k", lw=.8); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25); ax.set_ylabel("누적 손익 (천 달러)")
ax.set_title("SPX 30분 상승 + 두 번째 조건별 (10:00 ATM 콜)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(D / "momentum.png", dpi=150); plt.close(fig)

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 20)
print("① 타이밍\n", pd.DataFrame(R["timing"]).T.round(1).to_string())
print("\n③ MNQ 조건 대조\n", pd.DataFrame(R["momentum"]).T.round(1).to_string())
print("구름 신호와 겹치는 비율%:", {k: round(v) for k, v in R["momentum_overlap_with_cloud%"].items()})
print("\n⑦ 부트스트랩\n", json.dumps(boot, ensure_ascii=False, indent=1, default=lambda x: round(float(x), 1)))
print("\n⑧ 꼬리 제거\n", pd.DataFrame(tail).T[["총손익$", "건당$"]].round(0).to_string())
print("\n⑨ 국면\n", json.dumps(reg, ensure_ascii=False, indent=1, default=float))
