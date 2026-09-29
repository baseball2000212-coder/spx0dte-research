"""
방향성 단일 레그 매수 (콜만 or 풋만, ATM 1계약). 12 결과 사용, 몇 초~1분.
신호
  - 개장 추세: 진입 시각(09:45·10:00·10:30·11:00) F가 개장(09:31) F보다 높으면 콜, 낮으면 풋
  - 장중 모멘텀(15:30 진입): ① 전일종가 → 10:00 방향  ② 15:00 → 15:30 방향
청산 규칙
  - 만기 / -30% 손절 / 추세 이탈(SPX 1분이 30분 이동평균을 거꾸로 뚫으면) / 추세 이탈 + -30% 손절
검증: 신호 강도(|움직임|) 5분위 경계는 학습(< 2025-01-01)으로만 정함. 반대 방향 매수도 같이 계산.
결과: output/D1_directional.png, D2_same_vs_opposite.png, D3_examples.png, D_*.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.exits import FEE, EXERCISE

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
SPLIT = pd.Timestamp("2025-01-01")
MA_N, SL = 30, 0.3
RULES = ["만기", "-30% 손절", "추세 이탈", "추세 이탈 + -30% 손절"]

L = pd.read_pickle(OUT / "leg_paths.pkl")
FP = pd.read_pickle(OUT / "f_paths.pkl")
COLS = list(FP.columns)
MA = FP.T.rolling(MA_N, min_periods=10).mean().T
prev = load_spx_ohlc(SPX_CSV).close.shift(1)
f_open = FP[COLS[:5]].bfill(axis=1).iloc[:, 0]                     # 09:31(없으면 09:35까지 첫 값)
at = lambda d, hhmm: FP.at[d, hhmm] if d in FP.index else np.nan

# ── 신호 정의: (이름, 진입 시각, 신호 계산) ──
SIGNALS = [(f"{e} 개장추세", e, lambda r: r.F0 / f_open.get(r.date, np.nan) - 1) for e in ("09:45", "10:00", "10:30", "11:00")]
SIGNALS += [("15:30 (전일종가→10시)", "15:30", lambda r: at(r.date, "10:00") / prev.get(r.date, np.nan) - 1),
            ("15:30 (15시→15:30)", "15:30", lambda r: r.F0 / at(r.date, "15:00") - 1)]


def trade(r, direction, rule):
    """(손익 pt, 비용 pt, 사유, 청산 분 인덱스)"""
    cost, path, pay = (r.call_ask, r.call_bid, r.call_pay) if direction > 0 else (r.put_ask, r.put_bid, r.put_pay)
    s = COLS.index(r.entry) + 1                                       # path[0]이 가리키는 분
    hit = np.zeros(len(path), bool)
    if "손절" in rule:
        hit |= path <= cost * (1 - SL)
    if "추세" in rule:
        f, m = FP.loc[r.date].values[s:s + len(path)], MA.loc[r.date].values[s:s + len(path)]
        hit |= (f < m) if direction > 0 else (f > m)
    hit &= np.isfinite(path)
    if hit.any():
        i = int(hit.argmax())
        return path[i] - cost - 2 * FEE, cost, ("손절" if path[i] <= cost * (1 - SL) else "추세 이탈"), s + i
    return pay - cost - FEE - (EXERCISE if pay > 0 else 0), cost, "만기", -1


rows = []
for name, e, fn in SIGNALS:
    sub = L[L.entry == e]
    for r in sub.itertuples():
        x = fn(r)
        if not np.isfinite(x) or x == 0:
            continue
        dr = 1 if x > 0 else -1
        for rule in RULES:
            pnl, cost, why, j = trade(r, dr, rule)
            rows.append((name, e, r.date, x * 100, rule, "같은 방향", pnl, cost, why, j))
        pnl, cost, why, j = trade(r, -dr, "만기")
        rows.append((name, e, r.date, x * 100, "만기", "반대 방향", pnl, cost, why, j))
T = pd.DataFrame(rows, columns=["signal", "entry", "date", "move%", "rule", "side", "pnl", "cost", "why", "exit_idx"])
T["part"] = np.where(T.date < SPLIT, "학습", "검증")
T["absmove"] = T["move%"].abs()

# 5분위 경계: 신호별 학습 구간 |움직임|
T["q"] = ""
for name in T.signal.unique():
    m = T.signal == name
    _, edges = pd.qcut(T.loc[m & (T.part == "학습"), "absmove"], 5, retbins=True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf
    T.loc[m, "q"] = pd.cut(T.loc[m, "absmove"], edges, labels=[f"Q{j}" for j in range(1, len(edges))]).astype(str)
T.to_pickle(OUT / "D_trades.pkl")

ret = lambda g: g.pnl.sum() / g.cost.sum() * 100
SIG = [s[0] for s in SIGNALS]
QS = ["Q1", "Q2", "Q3", "Q4", "Q5", "전체"]


def table(side, rule, part):
    h = T[(T.side == side) & (T.rule == rule) & (T.part == part)]
    H = h.groupby(["q", "signal"]).apply(ret, include_groups=False).unstack()
    H.loc["전체"] = h.groupby("signal").apply(ret, include_groups=False)
    return H.reindex(index=QS, columns=SIG)


# ═══ D1: 같은 방향 매수, 규칙 × 구간 ═══
fig, ax = plt.subplots(len(RULES), 2, figsize=(17, 4.3 * len(RULES)))
out = []
for i, rule in enumerate(RULES):
    for j, part in enumerate(("학습", "검증")):
        H = table("같은 방향", rule, part); x = ax[i, j]
        x.imshow(H.values, cmap="RdYlGn", vmin=-40, vmax=40, aspect="auto")
        for a in range(H.shape[0]):
            for b in range(H.shape[1]):
                x.text(b, a, f"{H.values[a, b]:.0f}", ha="center", va="center", fontsize=9)
        x.set_xticks(range(len(SIG)), [s.replace(" (", "\n(") for s in SIG], fontsize=8)
        x.set_yticks(range(len(QS)), ["Q1 (약)", "Q2", "Q3", "Q4", "Q5 (강)", "전체"])
        x.set_title(f"{rule} | {part} (수익률 %, 신호 방향으로 매수)")
        out.append(H.assign(rule=rule, part=part))
fig.tight_layout(); fig.savefig(OUT / "D1_directional.png", dpi=110); plt.close(fig)
pd.concat(out).round(1).to_csv(OUT / "D1_directional.csv", encoding="utf-8-sig")

# ═══ D2: 같은 방향 vs 반대 방향 (만기 보유, 전체 기간) ═══
fig, ax = plt.subplots(1, len(SIG), figsize=(20, 4.2), sharey=True)
for k, name in enumerate(SIG):
    h = T[(T.signal == name) & (T.rule == "만기")]
    for side, col in (("같은 방향", "C2"), ("반대 방향", "C3")):
        g = h[h.side == side].groupby("q").apply(ret, include_groups=False).reindex(QS[:5])
        ax[k].plot(QS[:5], g.values, marker="o", color=col, label=side)
    ax[k].axhline(0, color="k", lw=1); ax[k].set_title(name, fontsize=10); ax[k].grid(alpha=.3)
ax[0].set_ylabel("수익률 % (만기 보유)"); ax[0].legend()
fig.suptitle("신호 방향으로 산 것 vs 반대로 산 것 (Q1 약한 움직임 → Q5 강한 움직임)")
fig.tight_layout(); fig.savefig(OUT / "D2_same_vs_opposite.png", dpi=110); plt.close(fig)

# ═══ D3: 예시 날 차트 (10:00 개장추세, 추세 이탈 + -30% 손절) ═══
ex = T[(T.signal == "10:00 개장추세") & (T.rule == "추세 이탈 + -30% 손절") & (T.side == "같은 방향")]
pick = pd.concat([ex.nlargest(2, "pnl"), ex.nsmallest(1, "pnl"), ex[ex.date >= "2026-09-01"].tail(3)]).drop_duplicates("date")
fig, ax = plt.subplots(2, 3, figsize=(18, 8.5))
for x, r in zip(ax.flat, pick.itertuples()):
    f, m = FP.loc[r.date], MA.loc[r.date]
    x.plot(range(len(COLS)), f.values, color="black", lw=1, label="SPX (1분)")
    x.plot(range(len(COLS)), m.values, color="C0", lw=1, ls="--", label=f"{MA_N}분 이동평균")
    s = COLS.index("10:00"); x.axvline(s, color="C2", lw=1.5)
    x.scatter([s], [f.values[s]], color="C2", zorder=5, s=50, label="매수")
    if r.exit_idx >= 0:
        x.scatter([r.exit_idx], [f.values[r.exit_idx]], color="C3", zorder=5, s=50, marker="X", label=f"매도 ({r.why})")
    tick = [k for k, c in enumerate(COLS) if c.endswith(":00")]
    x.set_xticks(tick, [COLS[k] for k in tick], fontsize=8)
    side = "콜" if r._4 > 0 else "풋"
    x.set_title(f"{r.date:%Y-%m-%d} {side} 매수  손익 ${r.pnl * 100:,.0f}", fontsize=10)
    x.legend(fontsize=7); x.grid(alpha=.3)
fig.suptitle("10:00 개장 추세 따라 매수 → 추세 이탈 or -30% 손절 (예시)")
fig.tight_layout(); fig.savefig(OUT / "D3_examples.png", dpi=110); plt.close(fig)

# ═══ 달러 요약 ═══
S = []
for name in SIG:
    for rule in RULES:
        h = T[(T.signal == name) & (T.rule == rule) & (T.side == "같은 방향")].sort_values("date")
        eq = (h.pnl * 100).cumsum()
        S.append({"신호": name, "청산": rule, "거래": len(h), "평균 프리미엄$": h.cost.mean() * 100,
                  "학습%": ret(h[h.part == "학습"]), "검증%": ret(h[h.part == "검증"]),
                  "승률%": (h.pnl > 0).mean() * 100, "총손익$": eq.iloc[-1], "최대낙폭$": (eq - eq.cummax()).min(),
                  "바로 청산(5분내)%": ((h.exit_idx >= 0) & (h.exit_idx - COLS.index(h.entry.iloc[0]) <= 5)).mean() * 100})
S = pd.DataFrame(S).round(1)
S.to_csv(OUT / "D_summary.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(S.to_string(index=False))
print("\n같은 방향, 추세 이탈 + -30% 손절, 분위별:\n", table("같은 방향", "추세 이탈 + -30% 손절", "학습").round(1).to_string(),
      "\n", table("같은 방향", "추세 이탈 + -30% 손절", "검증").round(1).to_string())
print(f"\n시도한 칸: 신호 {len(SIG)} × 청산 {len(RULES)} × 분위 5(+전체) = {len(SIG) * len(RULES) * 6}")
