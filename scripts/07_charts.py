"""
스트래들 결과 + 06 피처로 차트 (notebook_cells/A·C에서 옮김). 몇 초.
  python scripts/07_charts.py [--fee 0.08] [--exercise 0.12]
결과: output/A_straddle.png, C1_iv_curve.png, C2_delta_hedge.png, C3_entry_heatmap.png, C3_entry_heatmap.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT
from spx0dte.features import ENTRIES, DH_ENTRIES

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
ap = argparse.ArgumentParser()
ap.add_argument("--fee", type=float, default=0.025, help="계약당 매매 수수료 (pt, $2.5 = 0.025)")
ap.add_argument("--exercise", type=float, default=0.025, help="만기 행사 수수료 (내가격 레그당 pt)")
a = ap.parse_args()
SPLIT = "2025-01-01"

src = OUT / "straddle_trades.csv"
t = pd.read_csv(src if src.exists() else "data/straddle_trades.csv", parse_dates=["date"])
# 수수료 반영: 매수 2레그 + 정산 시 내가격 레그 1개 행사 (정산액 0이면 행사 없음)
t["fees"] = 2 * a.fee + np.where(t.payoff > 0, a.exercise, 0)
t["pnl_net"] = t.pnl_ask - t.fees


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig)


# ═══ A) 스트래들 결과 4종 ═══
T = t.sort_values("date").copy()
T["배수"] = T.payoff / T.cost_mid
fig, ax = plt.subplots(2, 2, figsize=(16, 10))
x = ax[0, 0]
for e in ["10:00", "14:00", "15:30"]:
    s = T[T.entry == e].set_index("date")
    x.plot(s.index, s.payoff.rolling(60).sum() / s.cost_mid.rolling(60).sum(), label=e, lw=1.2)
x.axhline(1, color="k", ls="--", lw=1); x.legend()
x.set_title("① 60거래일 이동: 만기정산액 ÷ 스트래들 가격 (1 위 = 롱이 이기는 구간)")
x = ax[0, 1]
H = T.groupby([T.date.dt.year, "entry"]).apply(
    lambda g: g.pnl_net.sum() / g.cost_ask.sum() * 100, include_groups=False).unstack()[ENTRIES]
N = T.groupby(T.date.dt.year).date.nunique()
x.imshow(H.values, cmap="RdYlGn", vmin=-40, vmax=40, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        x.text(j, i, f"{H.values[i, j]:.0f}", ha="center", va="center", fontsize=10)
x.set_xticks(range(len(ENTRIES)), ENTRIES); x.set_yticks(range(len(H)), [f"{y} ({N[y]}일)" for y in H.index])
x.set_title("② 연도 × 진입시각 수익률 % (ask 매수 + 수수료)")
x = ax[1, 0]
for e in ["10:00", "15:00", "15:45"]:
    v = T[T.entry == e]["배수"].clip(upper=5)
    x.hist(v, bins=50, histtype="step", lw=1.5, density=True, label=f"{e}  (1 넘는 날 {(v > 1).mean():.0%})")
x.axvline(1, color="k", ls="--", lw=1); x.legend(); x.set_xlabel("만기정산액 ÷ 가격 (5 이상은 5로 묶음)")
x.set_title("③ 하루 배수 분포")
x = ax[1, 1]
for e in ["10:00", "15:00", "15:45"]:
    p = T[T.entry == e].pnl_net.sort_values(ascending=False).values
    x.plot(np.arange(1, len(p) + 1) / len(p) * 100, p.cumsum(), label=e)
x.axhline(0, color="k", lw=1); x.legend()
x.set_xlabel("좋은 날부터 정렬한 날짜 비율 %"); x.set_ylabel("누적 손익 (지수 포인트)")
x.set_title("④ 좋은 날 순으로 누적 → 꼭짓점 이후는 전부 손실일")
save(fig, "A_straddle.png")

# ═══ C) 06 결과 차트 ═══
daily = pd.read_csv(OUT / "daily_features.csv", parse_dates=["date"])
dh = pd.read_csv(OUT / "delta_hedge.csv", parse_dates=["date"])
IVC = pd.read_pickle(OUT / "atm_iv_curve.pkl")

# 1) 장중 ATM IV 곡선 (연도별 중앙값)
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
cv = IVC.loc[:, "09:35":"15:55"]
ticks = [j for j, c in enumerate(cv.columns) if c.endswith(":00") or c.endswith(":30")]
for y, g in cv.groupby(cv.index.year):
    ax[0].plot(g.median().values, label=f"{y} ({len(g)}일)")
    ax[1].plot(g.div(g["10:00"], axis=0).median().values, label=str(y))
for x in ax:
    x.set_xticks(ticks, cv.columns[ticks], rotation=45); x.legend(); x.grid(alpha=.3)
ax[0].set_title("장중 ATM IV (거래시간 기준 연율 %, 연도별 중앙값)")
ax[1].set_title("10:00 IV = 1로 맞춘 모양"); ax[1].axhline(1, color="k", lw=1)
save(fig, "C1_iv_curve.png")

# 2) 델타헤지 스트래들 = 순수 감마 vs 세타
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for e in DH_ENTRIES:
    s = dh[dh.entry == e].sort_values("date")
    ax[0].plot(s.date, s.dh_pnl.cumsum(), label=f"{e} 델타헤지")
    ax[0].plot(s.date, s.straddle_pnl.cumsum(), ls=":", label=f"{e} 헤지 없음")
    ax[1].scatter(s.rv_win / s.iv0, s.dh_pnl / s.cost * 100, s=6, alpha=.4, label=e)
ax[0].axhline(0, color="k", lw=1); ax[0].legend(); ax[0].set_title("누적 손익 (포인트, mid, 15:59 청산)")
ax[1].axhline(0, color="k", lw=1); ax[1].axvline(1, color="k", lw=1); ax[1].set_xlim(0, 3); ax[1].set_ylim(-100, 200)
ax[1].set_xlabel("진입 후 실현변동성 ÷ 진입 IV"); ax[1].set_ylabel("델타헤지 수익률 %"); ax[1].legend()
ax[1].set_title("실현 > 내재(오른쪽)일 때만 이익")
yr = dh.groupby([dh.date.dt.year, "entry"]).apply(
    lambda g: pd.Series({"수익률%": g.dh_pnl.sum() / g.cost.sum() * 100,
                         "실현>내재%": (g.rv_win > g.iv0).mean() * 100}), include_groups=False)
yr["수익률%"].unstack().plot.bar(ax=ax[2]); ax[2].axhline(0, color="k", lw=1)
ax[2].set_title("연도별 델타헤지 수익률 %"); ax[2].set_xlabel("")
save(fig, "C2_delta_hedge.png")
yr.round(1).unstack().to_csv(OUT / "C2_delta_hedge_by_year.csv", encoding="utf-8-sig")
print("델타헤지 연도별:\n", yr.round(1).unstack().to_string())

# 3) 진입조건 히트맵: 학습 구간 5분위 경계 → 검증 구간에 그대로 적용
D = daily.set_index("date")
D["rv5"] = D.rv_day.rolling(5).mean().shift(1)     # 직전 5일 평균 실현변동성
D["abs_gap"] = D.gap.abs()


def heat(fn):
    out = {"학습": {}, "검증": {}}
    for e in ENTRIES:
        s = t[t.entry == e].merge(D.reset_index(), on="date")
        s["x"] = fn(s, e); s = s.replace([np.inf, -np.inf], np.nan).dropna(subset=["x"])
        _, edges = pd.qcut(s[s.date < SPLIT].x, 5, retbins=True, duplicates="drop")
        edges[0], edges[-1] = -np.inf, np.inf
        s["q"] = pd.cut(s.x, edges, labels=[f"Q{j}" for j in range(1, len(edges))])
        for part, h in (("학습", s[s.date < SPLIT]), ("검증", s[s.date >= SPLIT])):
            g = h.groupby("q", observed=False)
            out[part][e] = g.pnl_net.sum() / g.cost_ask.sum() * 100
    return {k: pd.DataFrame(v) for k, v in out.items()}


FEATS = {"IV ÷ 직전5일 RV (낮을수록 쌈)": lambda s, e: s[f"iv_{e}"] / s.rv5,
         "|시가 갭| %": lambda s, e: s.abs_gap,
         "개장 30분 RV": lambda s, e: s.rv_am}
fig, ax = plt.subplots(len(FEATS), 2, figsize=(15, 4.2 * len(FEATS)))
rows = []
for r, (name, fn) in enumerate(FEATS.items()):
    hm = heat(fn)
    for c, part in enumerate(("학습", "검증")):
        x, H = ax[r, c], hm[part]
        x.imshow(H.values, cmap="RdYlGn", vmin=-50, vmax=50, aspect="auto")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                x.text(j, i, f"{H.values[i, j]:.0f}", ha="center", va="center", fontsize=9)
        x.set_xticks(range(H.shape[1]), H.columns); x.set_yticks(range(H.shape[0]), H.index)
        x.set_title(f"{name} | {part} (수익률 %, ask+수수료)")
        rows.append(H.assign(feature=name, part=part))
save(fig, "C3_entry_heatmap.png")
pd.concat(rows).to_csv(OUT / "C3_entry_heatmap.csv", encoding="utf-8-sig")
print(f"\n차트 저장: {OUT.resolve()}")
