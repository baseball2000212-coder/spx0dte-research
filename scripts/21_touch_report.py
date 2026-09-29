"""
구름 터치 매매 결과 정리 (20 또는 22 결과 사용, 몇 초).
  python scripts/21_touch_report.py                 SPX 차트 신호 (20)
  python scripts/21_touch_report.py --src mnq       MNQ 차트 신호 (22)
  python scripts/21_touch_report.py --src mnq --id 5   그 조합으로 예시 차트
결과: output/T1_touch_grid{_mnq}.png, T2_touch_equity{_mnq}.png, T3_touch_days{_mnq}.png, T_summary{_mnq}.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT
from spx0dte.ichimoku import bars, ichimoku, draw
from spx0dte.touch import load_mnq, mnq_bars

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="spx", choices=["spx", "mnq", "mnq_strict"])
ap.add_argument("--id", type=int, default=None)
a = ap.parse_args()
SFX = "" if a.src == "spx" else "_" + a.src
SRC_NAME = {"spx": "SPX 차트", "mnq": "MNQ 차트", "mnq_strict": "MNQ 차트(관통 제외)"}[a.src]
SPLIT, ODD = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-09")
T = pd.read_pickle(OUT / f"touch_trades{SFX}.pkl")
C = pd.read_csv(OUT / f"touch_configs{SFX}.csv").set_index("id")
NDAYS = pd.read_pickle(OUT / "f_paths.pkl").shape[0]
ret = lambda g: g.pnl.sum() / g.cost.sum() * 100 if len(g) else np.nan
EXIT_NAME = {"손절30": "-30% 손절", "구름이탈": "구름 반대 이탈", "15분": "15분 본전 이하"}

rows = []
for cid, g in T.groupby("cfg"):
    g = g.sort_values(["date", "entry"])
    day = g.groupby("date").pnl.sum() * 100
    eq = day.cumsum()
    v = g[g.date >= SPLIT]
    rows.append({"id": cid, **C.loc[cid].to_dict(), "거래": len(g), "하루 평균": len(g) / NDAYS,
                 "평균 프리미엄$": g.cost.mean() * 100, "학습%": ret(g[g.date < SPLIT]), "검증%": ret(v),
                 "검증(4/9 제외)%": ret(v[v.date != ODD]), "최고 10건 빼면%": ret(g.drop(g.pnl.nlargest(10).index)),
                 "승률%": (g.pnl > 0).mean() * 100, "손절%": (g.why == "손절").mean() * 100,
                 "총손익$": eq.iloc[-1], "최대낙폭$": (eq - eq.cummax()).min(), "플러스 연도": int((g.groupby(g.date.dt.year).apply(ret, include_groups=False) > 0).sum())})
S = pd.DataFrame(rows)
S["둘 중 나쁜 쪽"] = S[["학습%", "검증(4/9 제외)%"]].min(axis=1)
S = S.sort_values("둘 중 나쁜 쪽", ascending=False).round(1)
S.to_csv(OUT / f"T_summary{SFX}.csv", index=False, encoding="utf-8-sig")

# ═══ T1: 전체 조합 히트맵 ═══
S["행"] = S.rule + " · " + S.color
S["열"] = S.tf.astype(str) + "분봉\n" + S.exit.map(EXIT_NAME)
ROWS = [f"{r} · {c}" for r in C.rule.unique() for c in ("전체", "음운만")]
COLS = [f"{tf}분봉\n{EXIT_NAME[x]}" for tf in (1, 5) for x in ("손절30", "구름이탈", "15분")]
fig, ax = plt.subplots(1, 3, figsize=(24, 9))
for j, col in enumerate(("학습%", "검증%", "검증(4/9 제외)%")):
    H = S.pivot(index="행", columns="열", values=col).reindex(index=ROWS, columns=COLS)
    N = S.pivot(index="행", columns="열", values="하루 평균").reindex(index=ROWS, columns=COLS)
    ax[j].imshow(H.values, cmap="RdYlGn", vmin=-25, vmax=25, aspect="auto")
    for p in range(H.shape[0]):
        for q in range(H.shape[1]):
            ax[j].text(q, p, f"{H.values[p, q]:.0f}\n({N.values[p, q]:.1f}/일)", ha="center", va="center", fontsize=7.5)
    ax[j].set_xticks(range(len(COLS)), COLS, fontsize=8); ax[j].set_yticks(range(len(ROWS)), ROWS, fontsize=9)
    ax[j].set_title(f"{SRC_NAME} 구름 터치 | {col.replace('%', '')} (수익률 %, 괄호 = 하루 평균 거래)")
fig.tight_layout(); fig.savefig(OUT / f"T1_touch_grid{SFX}.png", dpi=105); plt.close(fig)

# ═══ T2: 상위 조합 누적 손익 ═══
top = S[S["거래"] >= 200].head(4)
fig, ax = plt.subplots(figsize=(15, 5.5))
for _, r in top.iterrows():
    g = T[T.cfg == r.id]
    eq = (g.groupby("date").pnl.sum() * 100).cumsum() / 1000
    ax.plot(eq.index, eq.values, label=f"#{r.id} {r.tf}분봉 {r.rule}·{r.color}·{EXIT_NAME[r.exit]} ({r['하루 평균']:.1f}건/일)")
ax.axhline(0, color="k", lw=1); ax.axvline(SPLIT, color="grey", ls="--"); ax.axvline(ODD, color="orange", ls=":")
ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_title(f"{SRC_NAME} 신호 상위 조합 누적 손익 (천 달러, SPX 0DTE 1계약씩, 수수료 포함)")
fig.tight_layout(); fig.savefig(OUT / f"T2_touch_equity{SFX}.png", dpi=115); plt.close(fig)

# ═══ T3: 예시 날 장중 차트 (신호를 본 차트 위에 매수·매도 표시) ═══
cid = a.id if a.id is not None else int(top.iloc[0].id)
cfg = C.loc[cid]; tf = int(cfg.tf)
if a.src == "spx":
    B = bars(pd.read_pickle(OUT / "f_paths.pkl"), tf)
    shift = pd.Timedelta(0)                        # SPX: 봉 이름 = 봉이 포함한 분
else:
    B = mnq_bars(load_mnq()[0], tf)
    shift = pd.Timedelta(minutes=tf)               # MNQ: 매수 분 = 봉 끝난 다음 분 → 봉 시작 = 매수 분 − tf
IC = ichimoku(B)
g = T[T.cfg == cid]
dp = g.groupby("date").pnl.sum()
days = list(dp.nlargest(2).index) + list(dp.nsmallest(2).index) + list(dp[dp.index >= "2026-09-01"].index[-2:])
fig, ax = plt.subplots(3, 2, figsize=(18, 14))
for x, d in zip(ax.flat, days):
    if a.src == "spx":
        msk = B.index.normalize() == d
    else:
        msk = (B.index >= pd.Timestamp(f"{d:%Y-%m-%d} 09:30", tz=B.index.tz)) & (B.index < pd.Timestamp(f"{d:%Y-%m-%d} 16:00", tz=B.index.tz))
    b, ic = B[msk], IC[msk]
    t0 = b.index[0].normalize()
    pos = lambda hhmm: int(np.clip(np.searchsorted(b.index, t0 + pd.Timedelta(hhmm + ":00") - shift, side="right") - 1, 0, len(b) - 1))
    draw(x, b, ic, "", candles=(tf > 1))
    for r in g[g.date == d].itertuples():
        i, j = pos(r.entry), (pos(r.exit) if r.exit != "16:00" else len(b) - 1)
        c = "#2A9D8F" if r.leg == "C" else "#7B2CBF"
        x.scatter(i, b.close.iloc[i], marker="^" if r.leg == "C" else "v", color=c, s=90, zorder=6)
        x.scatter(j, b.close.iloc[j], marker="X", color="#D62828" if r.pnl < 0 else "#2A9D8F", s=70, zorder=6)
        x.annotate(f"{'콜' if r.leg == 'C' else '풋'} {r.pnl * 100:+,.0f}$", (j, b.close.iloc[j]), fontsize=7,
                   xytext=(3, 6), textcoords="offset points")
    tk = [k for k, t in enumerate(b.index) if t.minute == 0]
    x.set_xticks(tk, [b.index[k].strftime("%H:%M") for k in tk], fontsize=7)
    x.set_title(f"{d:%Y-%m-%d}  그날 합계 ${dp[d] * 100:+,.0f}  ({len(g[g.date == d])}번 매매)", fontsize=10)
fig.suptitle(f"#{cid} {SRC_NAME} {tf}분봉 · {cfg.rule} · {cfg.color} · {EXIT_NAME[cfg.exit]}   ▲콜 ▼풋 매수, X 청산 (초록 이익 / 빨강 손실)"
             f"   시간 = 뉴욕 (한국 = +13시간, 겨울 +14시간)")
fig.tight_layout(); fig.savefig(OUT / f"T3_touch_days{SFX}.png", dpi=105); plt.close(fig)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = ["id", "tf", "rule", "color", "exit", "거래", "하루 평균", "평균 프리미엄$", "학습%", "검증%", "검증(4/9 제외)%",
        "최고 10건 빼면%", "승률%", "손절%", "총손익$", "최대낙폭$", "플러스 연도"]
print(S[show].head(15).to_string(index=False))
print("\n규칙별 평균:\n", S.groupby("rule")[["학습%", "검증(4/9 제외)%"]].mean().round(1).to_string())
print("\n청산별 평균:\n", S.groupby("exit")[["학습%", "검증(4/9 제외)%"]].mean().round(1).to_string())
print("\n봉·색별 평균:\n", S.groupby(["tf", "color"])[["학습%", "검증(4/9 제외)%"]].mean().round(1).to_string())
print(f"\n두 구간(4/9 제외) 다 플러스인 조합: {int(((S['학습%'] > 0) & (S['검증(4/9 제외)%'] > 0)).sum())}개 / {len(S)}개")
print(f"예시 차트 조합: #{cid}")