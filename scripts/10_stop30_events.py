"""
① 롱 ATM 스트래들 −30% 손절 + 익절 없음(만기 정산) 백테스트, 진입시각 7개 각각 매일 1세트
② 이벤트일(FOMC·CPI·고용) vs 나머지 날 비교 (학습 < 2025-01-01 ≤ 검증)
08 결과(output/straddle_paths.pkl, straddle_trades.csv) 사용, 몇 초.
결과: output/R1_stop30.png, R2_event_heatmap.png, R3_event_pricing.png, R_*.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT
from spx0dte.features import ENTRIES
from spx0dte.exits import run_rule
from spx0dte.events import label_events

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
SPLIT = pd.Timestamp("2025-01-01")
LABELS = ["FOMC", "CPI", "고용", "이벤트 없음"]

P = pd.read_pickle(OUT / "straddle_paths.pkl")
P = P[P.entry.isin(ENTRIES)]
RULES = {"-30% 손절": run_rule(P, sl=0.3), "만기 보유": run_rule(P)}
for t in RULES.values():
    t["part"] = np.where(t.date < SPLIT, "학습", "검증")
    t["event"] = label_events(t.date)
    t["mins_held"] = (pd.to_datetime(t.exit_time, format="%H:%M") - pd.to_datetime(t.entry, format="%H:%M")).dt.total_seconds() / 60
ret = lambda g: g.pnl.sum() / g.cost_ask.sum() * 100

# ═══ ① −30% 손절 백테스트 요약 ═══
S, B = RULES["-30% 손절"], RULES["만기 보유"]
rows = []
for e in ENTRIES:
    s, b = S[S.entry == e].sort_values("date"), B[B.entry == e].sort_values("date")
    eq = (s.pnl * 100).cumsum()
    rows.append({"진입": e, "거래일": len(s), "평균 프리미엄$": s.cost_ask.mean() * 100,
                 "손절 걸린 날%": (s.why == "손절").mean() * 100,
                 "손절까지 중앙값(분)": s[s.why == "손절"].mins_held.median(),
                 "승률%": (s.pnl > 0).mean() * 100,
                 "수익률% 학습": ret(s[s.part == "학습"]), "수익률% 검증": ret(s[s.part == "검증"]),
                 "(만기보유) 학습": ret(b[b.part == "학습"]), "(만기보유) 검증": ret(b[b.part == "검증"]),
                 "총손익$": eq.iloc[-1], "연평균$": eq.iloc[-1] / (len(s) / 252),
                 "최대낙폭$": (eq - eq.cummax()).min(), "최악의 날$": s.pnl.min() * 100, "최고의 날$": s.pnl.max() * 100})
summ = pd.DataFrame(rows).round(1)
summ.to_csv(OUT / "R1_stop30_summary.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
for e in ENTRIES:
    s = S[S.entry == e].sort_values("date")
    ax[0].plot(s.date, (s.pnl * 100).cumsum() / 1000, label=e, lw=1.3)
ax[0].axhline(0, color="k", lw=1); ax[0].axvline(SPLIT, color="grey", ls="--", lw=1)
ax[0].text(SPLIT, ax[0].get_ylim()[0] * 0.95, " 검증 구간 →", color="grey")
ax[0].legend(ncol=4, fontsize=9); ax[0].set_ylabel("누적 손익 (천 달러)")
ax[0].set_title("-30% 손절 + 만기 정산: 매일 1세트 누적 손익 (수수료 포함)")
H = S.groupby([S.date.dt.year, "entry"]).apply(ret, include_groups=False).unstack()[ENTRIES]
ax[1].imshow(H.values, cmap="RdYlGn", vmin=-30, vmax=30, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax[1].text(j, i, f"{H.values[i, j]:.0f}", ha="center", va="center", fontsize=10)
ax[1].set_xticks(range(len(ENTRIES)), ENTRIES); ax[1].set_yticks(range(len(H)), H.index)
ax[1].set_title("연도 × 진입시각 수익률 % (-30% 손절)")
fig.tight_layout(); fig.savefig(OUT / "R1_stop30.png", dpi=120); plt.close(fig)

# ═══ ② 이벤트일 히트맵 ═══
fig, ax = plt.subplots(2, 2, figsize=(15, 8.5))
tabs = []
for r, (rule, t) in enumerate(RULES.items()):
    for c, part in enumerate(("학습", "검증")):
        h = t[t.part == part]
        H = h.groupby(["event", "entry"]).apply(ret, include_groups=False).unstack().reindex(index=LABELS, columns=ENTRIES)
        n = h[h.entry == "10:00"].event.value_counts().reindex(LABELS).fillna(0).astype(int)
        x = ax[r, c]
        x.imshow(H.values, cmap="RdYlGn", vmin=-40, vmax=40, aspect="auto")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                x.text(j, i, f"{H.values[i, j]:.0f}", ha="center", va="center", fontsize=10)
        x.set_xticks(range(len(ENTRIES)), ENTRIES); x.set_yticks(range(len(LABELS)), [f"{l} ({n[l]}일)" for l in LABELS])
        x.set_title(f"{rule} | {part} (수익률 %)")
        tabs.append(H.assign(rule=rule, part=part))
fig.tight_layout(); fig.savefig(OUT / "R2_event_heatmap.png", dpi=120); plt.close(fig)
pd.concat(tabs).round(1).to_csv(OUT / "R2_event_returns.csv", encoding="utf-8-sig")

# ═══ ③ 이벤트일 가격: 옵션이 예상한 움직임 vs 실제 움직임 ═══
T = pd.read_csv(OUT / "straddle_trades.csv", parse_dates=["date"])
T = T[T.entry.isin(ENTRIES)]
T["event"] = label_events(T.date)
pr = T.groupby(["event", "entry"]).apply(
    lambda g: pd.Series({"예상(스트래들 가격)": g.cost_mid.mean(), "실제(|종가−진입가|)": g.realized_move.mean()}),
    include_groups=False)
pr["예상÷실제"] = pr.iloc[:, 0] / pr.iloc[:, 1]
pr.round(2).to_csv(OUT / "R3_event_pricing.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
Q = pr["예상÷실제"].unstack().reindex(index=LABELS, columns=ENTRIES)
Q.T.plot.bar(ax=ax[0], rot=0); ax[0].axhline(1, color="k", lw=1)
ax[0].set_title("옵션이 예상한 움직임 ÷ 실제 움직임 (1보다 크면 비싸게 산 것)"); ax[0].set_ylim(0.6, 1.5)
M = pr["예상(스트래들 가격)"].unstack().reindex(index=LABELS, columns=ENTRIES)
R = pr["실제(|종가−진입가|)"].unstack().reindex(index=LABELS, columns=ENTRIES)
for k, col in zip(LABELS, ("C0", "C1", "C2", "C3")):
    ax[1].plot(ENTRIES, M.loc[k], color=col, marker="o", label=f"{k} 예상")
    ax[1].plot(ENTRIES, R.loc[k], color=col, ls=":", marker="x", label=f"{k} 실제")
ax[1].set_ylabel("포인트"); ax[1].legend(ncol=2, fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title("스트래들 가격(실선) vs 실제 움직임(점선)")
fig.tight_layout(); fig.savefig(OUT / "R3_event_pricing.png", dpi=120); plt.close(fig)

pd.set_option("display.width", 250)
print("① −30% 손절 + 만기 정산 요약:\n", summ.to_string(index=False))
print("\n② 이벤트일 수익률 %:\n", pd.concat(tabs).round(1).to_string())
print("\n③ 예상÷실제:\n", Q.round(2).to_string())
print(f"\n시도: 이벤트 3종 × 진입시각 7 × 규칙 2 = 42칸 (+ 이벤트 없음 기준)")
