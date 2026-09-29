"""
롱 ATM 스트래들 손절·익절 규칙 테스트 (08 결과 사용, 몇 초).
  - 진입: 매도호가(ask)로 매수
  - 매분 '지금 팔면 받는 값'(bid 합)을 보고, 손절선 이하 or 익절선 이상이면 그 bid로 청산
  - 둘 다 안 걸리면 만기(16:00 SPX 종가) 정산
  - 수수료: 매수 2계약 + 중도청산 2계약(각 --fee), 만기 때 내가격 레그 행사 --exercise
  python scripts/09_exit_rules.py
결과: output/E1_exit_heatmap.png, E1_exit_grid.csv, E2_stop_regret.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT
from spx0dte.features import ENTRIES

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
ap = argparse.ArgumentParser()
ap.add_argument("--fee", type=float, default=0.025)
ap.add_argument("--exercise", type=float, default=0.025)
a = ap.parse_args()
SPLIT = pd.Timestamp("2025-01-01")
SL = [None, 0.3, 0.5, 0.7]          # 손절: 가치가 매수가 대비 30/50/70% 빠지면
TP = [None, 1.5, 2.0, 3.0]          # 익절: 가치가 매수가의 1.5/2/3배가 되면
sl_name = lambda s: "손절 없음" if s is None else f"-{s:.0%} 손절"
tp_name = lambda t: "익절 없음" if t is None else f"{t:g}배 익절"

P = pd.read_pickle(OUT / "straddle_paths.pkl")
P = P[P.entry.isin(ENTRIES)]


def apply_rule(row, sl, tp):
    """(청산 손익 pt, 청산 사유, 청산 시각 인덱스)"""
    c, path = row.cost_ask, row.bid_path
    hit = np.zeros(len(path), bool)
    if sl is not None:
        hit |= path <= c * (1 - sl)
    if tp is not None:
        hit |= path >= c * tp
    hit &= np.isfinite(path)
    if hit.any():
        i = int(hit.argmax())
        return path[i] - c - 4 * a.fee, ("손절" if path[i] < c else "익절"), i
    return row.payoff - c - 2 * a.fee - (a.exercise if row.payoff > 0 else 0), "만기", -1


res = []
for sl in SL:
    for tp in TP:
        for r in P.itertuples():
            pnl, why, _ = apply_rule(r, sl, tp)
            res.append((sl_name(sl), tp_name(tp), r.entry, r.date, r.cost_ask, pnl, why))
R = pd.DataFrame(res, columns=["sl", "tp", "entry", "date", "cost", "pnl", "why"])
R["part"] = np.where(R.date < SPLIT, "학습", "검증")

g = R.groupby(["part", "entry", "sl", "tp"])
grid = pd.DataFrame({"수익률%": g.pnl.sum() / g.cost.sum() * 100,
                     "승률%": g.pnl.apply(lambda x: (x > 0).mean() * 100),
                     "손절비율%": g.why.apply(lambda x: (x == "손절").mean() * 100),
                     "건수": g.size()}).round(1)
grid.to_csv(OUT / "E1_exit_grid.csv", encoding="utf-8-sig")

# ── 히트맵: 진입시각별 (행=손절, 열=익절), 학습 | 검증 ──
rows_, cols_ = [sl_name(s) for s in SL], [tp_name(t) for t in TP]
fig, ax = plt.subplots(len(ENTRIES), 2, figsize=(12, 3.0 * len(ENTRIES)))
for i, e in enumerate(ENTRIES):
    for j, part in enumerate(("학습", "검증")):
        H = grid.loc[(part, e), "수익률%"].unstack().reindex(index=rows_, columns=cols_)
        x = ax[i, j]
        x.imshow(H.values, cmap="RdYlGn", vmin=-20, vmax=20, aspect="auto")
        for m in range(H.shape[0]):
            for n in range(H.shape[1]):
                x.text(n, m, f"{H.values[m, n]:.1f}", ha="center", va="center", fontsize=9,
                       fontweight="bold" if (m, n) == (0, 0) else "normal")
        x.set_xticks(range(len(cols_)), cols_, fontsize=8); x.set_yticks(range(len(rows_)), rows_, fontsize=8)
        x.set_title(f"{e} 진입 | {part} (수익률 %, 굵은 칸 = 규칙 없음)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "E1_exit_heatmap.png", dpi=120); plt.close(fig)

# ── 손절 후회: 손절한 거래가 만기까지 들고 있었으면? ──
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
reg = []
for e in ENTRIES:
    for s in SL[1:]:
        sub = P[P.entry == e]
        out = []
        for r in sub.itertuples():
            pnl, why, _ = apply_rule(r, s, None)
            if why == "손절":
                hold = r.payoff - r.cost_ask - 2 * a.fee - (a.exercise if r.payoff > 0 else 0)
                out.append((pnl, hold))
        o = np.array(out) if out else np.zeros((0, 2))
        reg.append({"entry": e, "sl": sl_name(s), "손절 건수 비율%": len(o) / len(sub) * 100,
                    "손절 후 만기면 더 나았던 비율%": (o[:, 1] > o[:, 0]).mean() * 100 if len(o) else np.nan,
                    "손절 후 만기면 이익났던 비율%": (o[:, 1] > 0).mean() * 100 if len(o) else np.nan})
reg = pd.DataFrame(reg)
for s, gg in reg.groupby("sl", sort=False):
    ax[0].plot(gg.entry, gg["손절 건수 비율%"], marker="o", label=s)
    ax[1].plot(gg.entry, gg["손절 후 만기면 이익났던 비율%"], marker="o", label=s)
ax[0].set_title("손절이 걸린 날 비율 %"); ax[1].set_title("손절당한 거래 중 만기까지 들고 있었으면 이익이던 비율 %")
for x in ax:
    x.legend(); x.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "E2_stop_regret.png", dpi=120); plt.close(fig)

base = grid.xs(("손절 없음", "익절 없음"), level=("sl", "tp"))["수익률%"].unstack(0)[["학습", "검증"]]
best = grid.reset_index().loc[lambda d: d.part == "학습"].sort_values("수익률%", ascending=False) \
    .groupby("entry").head(1).set_index("entry")[["sl", "tp", "수익률%"]]
best["검증 수익률%"] = [grid.loc[("검증", e, r.sl, r.tp), "수익률%"] for e, r in best.iterrows()]
print("규칙 없음 (기준):\n", base.reindex(ENTRIES).to_string())
print("\n학습 구간에서 제일 좋았던 규칙 → 검증에 그대로 적용:\n", best.reindex(ENTRIES).to_string())
print("\n손절 후회:\n", reg.round(1).to_string(index=False))
print(f"\n시도한 조합: 손절 {len(SL)} × 익절 {len(TP)} × 진입시각 {len(ENTRIES)} = {len(SL) * len(TP) * len(ENTRIES)}개")

# ── 달러 요약: 매일 그 시각에 1세트(콜 1 + 풋 1)씩 샀다면 (SPX 1pt = $100) ──
usd = []
for (sl, tp) in [("손절 없음", "익절 없음"), ("-30% 손절", "익절 없음")]:
    for e in ENTRIES:
        s = R[(R.sl == sl) & (R.tp == tp) & (R.entry == e)].sort_values("date")
        eq = (s.pnl * 100).cumsum()
        usd.append({"규칙": sl, "진입": e, "거래일": len(s), "평균 프리미엄$": s.cost.mean() * 100,
                    "총손익$": eq.iloc[-1], "연평균손익$": eq.iloc[-1] / (len(s) / 252),
                    "최대낙폭$": (eq - eq.cummax()).min(), "최악의 날$": s.pnl.min() * 100,
                    "최고의 날$": s.pnl.max() * 100, "승률%": (s.pnl > 0).mean() * 100})
usd = pd.DataFrame(usd).round(0)
usd.to_csv(OUT / "E3_dollar_summary.csv", index=False, encoding="utf-8-sig")
print("\n달러 요약 (매일 1세트):\n", usd.to_string(index=False))
