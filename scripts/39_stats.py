"""
포트폴리오 통계 검증 (기준 전략: 10:00 ATM 콜 1계약, 만기, 중간가).
  1 기준선: 매일 / 조건1만(SPX 30분 상승) / 조건2만(MNQ 5분 구름 위) / 둘 다(현재 규칙)
  2 무작위 대조: 신호일과 같은 수만큼 날짜를 무작위로 골라 × 10,000회 → 총손익 분포, 현재 규칙 백분위 (4/9 포함·제외)
  3 부트스트랩: 건당 손익 평균 95% 신뢰구간 (4/9 포함·제외)
  4 225개 조합 수익률 분포 + 현재 규칙 순위 (34_optimize.py 결과)
  5 수수료 민감도: 편도 $2.5 / $8 × 만기 정산 수수료 $0 / $8 / $12
결과: output/stats/*.png, stats.json
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
D = OUT / "stats"; D.mkdir(exist_ok=True)
ODD = pd.Timestamp("2025-04-09"); RNG = np.random.default_rng(20260926); NSIM = 10_000
INK, ACC, LOSS, MUTED = "#17201C", "#0F6E5A", "#B3261E", "#9AA39E"

FP = pd.read_pickle(OUT / "f_paths.pkl")
X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), load_spx_ohlc(SPX_CSV))
L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date")
L = L[(L.leg == "C") & (L.offset == 0)].join(X, how="inner").sort_index()
L["gross"] = L.pay - L.mid                                            # 수수료 빼기 전 (pt)


def net(g, fee=0.025, ex=0.025):
    return (g.gross - fee - np.where(g.pay > 0, ex, 0)) * 100        # 달러


L["pnl$"] = net(L)
out = {}

# ═══ 1 기준선 ═══
BASE = {"매일 매수": L.index == L.index, "조건1만 (SPX 30분 상승)": L.r30 > 0,
        "조건2만 (MNQ 5분 구름 위)": L.c5 == 1, "둘 다 (현재 규칙)": L.signal}
rows = []
for k, m in BASE.items():
    g = L[np.asarray(m, bool)]
    y = g["pnl$"].groupby(g.index.year).sum()
    rows.append({"기준": k, "매수 건수": len(g), "총손익$": g["pnl$"].sum(), "건당 평균$": g["pnl$"].mean(),
                 "수익률%": g["pnl$"].sum() / (g.mid.sum() * 100) * 100, **{str(yy): v for yy, v in y.items()}})
B = pd.DataFrame(rows); out["baseline"] = B.round(1).to_dict("records")
fig, ax = plt.subplots(figsize=(9, 3.2))
for (k, m), c in zip(BASE.items(), (MUTED, "#3A86A8", "#C77B2B", ACC)):
    g = L[np.asarray(m, bool)]
    ax.plot(g.index, g["pnl$"].cumsum() / 1000, color=c, lw=2 if "현재" in k else 1.2, label=f"{k} ({len(g)}건)")
ax.axhline(0, color=INK, lw=.8); ax.legend(frameon=False, fontsize=8); ax.set_ylabel("누적 손익 (천 달러)"); ax.grid(alpha=.25)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(D / "baseline.png", dpi=170); plt.close(fig)

# ═══ 2 무작위 대조 ═══
sig = L[L.signal]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
rand = {}
for j, (lab, pool, act) in enumerate((("4/9 포함", L, sig), ("4/9 제외", L.drop(ODD), sig.drop(ODD)))):
    v = pool["pnl$"].values; n = len(act)
    sims = np.array([v[RNG.choice(len(v), n, replace=False)].sum() for _ in range(NSIM)])
    real = act["pnl$"].sum(); pct = (sims < real).mean() * 100
    rand[lab] = {"실제": real, "무작위 중앙값": float(np.median(sims)), "무작위 95% 상단": float(np.percentile(sims, 95)),
                 "백분위": pct, "무작위가 실제 이상인 비율%": 100 - pct, "뽑은 날 수": n, "후보 날 수": len(v)}
    ax[j].hist(sims / 1000, bins=60, color=MUTED)
    ax[j].axvline(real / 1000, color=ACC, lw=2); ax[j].axvline(0, color=INK, lw=.6)
    ax[j].text(real / 1000, ax[j].get_ylim()[1] * .9, f" 현재 규칙\n 상위 {100 - pct:.1f}%", color=ACC, fontsize=8)
    ax[j].set_title(f"{lab}: 무작위 {n}일 × {NSIM:,}회", fontsize=9); ax[j].set_xlabel("총손익 (천 달러)", fontsize=8)
    for s in ("top", "right"): ax[j].spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(D / "random.png", dpi=170); plt.close(fig)
out["random"] = rand

# ═══ 3 부트스트랩 ═══
boot = {}
fig, ax = plt.subplots(figsize=(9, 2.6))
for lab, g, c in (("4/9 포함", sig, ACC), ("4/9 제외", sig.drop(ODD), "#3A86A8")):
    v = g["pnl$"].values
    ms = np.array([v[RNG.integers(0, len(v), len(v))].mean() for _ in range(NSIM)])
    lo, hi = np.percentile(ms, [2.5, 97.5])
    boot[lab] = {"건당 평균$": float(v.mean()), "95% 하한$": float(lo), "95% 상한$": float(hi), "평균>0 확률%": float((ms > 0).mean() * 100)}
    ax.hist(ms, bins=60, histtype="step", lw=1.6, color=c, label=f"{lab}: 평균 ${v.mean():,.0f}, 95% 구간 ${lo:,.0f} ~ ${hi:,.0f}")
ax.axvline(0, color=INK, lw=.8); ax.legend(frameon=False, fontsize=8); ax.set_xlabel("건당 평균 손익 ($), 재표본 10,000회")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(D / "bootstrap.png", dpi=170); plt.close(fig)
out["bootstrap"] = boot

# ═══ 4 225개 조합 ═══
G = pd.read_csv(OUT / "O_grid.csv")
G = G.dropna(subset=["수익률%"])
cur = G[(G.진입 == "10:00") & (G.행사가 == "ATM") & (G.물타기 == "없음") & (G.신호 == "구름")].iloc[0]
rank_ret = int((G["수익률%"] > cur["수익률%"]).sum()) + 1
rank_sc = int((G["점수(손익÷낙폭)"] > cur["점수(손익÷낙폭)"]).sum()) + 1
out["grid"] = {"조합 수": len(G), "플러스 조합%": float((G["수익률%"] > 0).mean() * 100),
               "플러스 조합 (4/9 빼고)%": float((G["4/9 빼고$"] > 0).mean() * 100),
               "현재 수익률%": float(cur["수익률%"]), "수익률 순위": rank_ret, "현재 점수": float(cur["점수(손익÷낙폭)"]), "점수 순위": rank_sc,
               "수익률 중앙값%": float(G["수익률%"].median())}
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.hist(G["수익률%"], bins=40, color=MUTED)
ax.axvline(cur["수익률%"], color=ACC, lw=2); ax.axvline(0, color=INK, lw=.8)
ax.text(cur["수익률%"], ax.get_ylim()[1] * .85, f" 현재 규칙 {cur['수익률%']:.1f}%\n 수익률 {rank_ret}위 / 점수 {rank_sc}위", color=ACC, fontsize=8)
ax.set_xlabel("조합별 수익률 % (진입 5 × 행사가 5 × 물타기 3 × 신호 3)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(D / "grid.png", dpi=170); plt.close(fig)

# ═══ 5 수수료 민감도 ═══
fees = []
for fee in (2.5, 8.0):
    for ex in (0.0, 8.0, 12.0):
        p = net(sig, fee / 100, ex / 100)
        eq = p.cumsum()
        fees.append({"편도 수수료$": fee, "만기 정산 수수료$": ex, "총손익$": p.sum(), "건당 평균$": p.mean(),
                     "수익률%": p.sum() / (sig.mid.sum() * 100 + len(sig) * fee) * 100, "최대낙폭$": (eq - eq.cummax()).min(),
                     "4/9 빼고$": p.drop(ODD).sum()})
out["fees"] = pd.DataFrame(fees).round(1).to_dict("records")
json.dump(out, open(D / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(B.round(0).to_string(index=False))
print(json.dumps({k: out[k] for k in ("random", "bootstrap", "grid")}, ensure_ascii=False, indent=1, default=float))
print(pd.DataFrame(fees).round(1).to_string(index=False))
