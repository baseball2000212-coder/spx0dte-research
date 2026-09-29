"""
진입 시각 × 행사가 비율 × 물타기 × 신호 최적화 + 걸어가며 검증(walk-forward). 33 결과 사용.
  신호(진입 시각 t 기준): SPX가 개장가보다 위 AND
     구름  = MNQ 5분봉(마지막 완성 봉) 종가 > 일목 구름 윗선
     나스닥 주도 = 09:30부터 t까지 MNQ 수익률 > ES 수익률
     둘 다
  물타기: 없음 / -50% 1번(손절 없음) / -40%·-70% 1번씩. 추가 매수는 14:30까지. 모두 만기 보유, 중간가 체결.
  점수 = 총손익 ÷ 최대낙폭 (위험 대비 효율, 계약 수와 무관)
  걸어가며 검증: 매년 초, 그 전 해들 데이터로만 점수 1등 조합을 골라 그해에 적용.
결과: output/O2_plateau.png, O3_walkforward.png, O_grid.csv, O_walkforward.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import itertools, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT
from spx0dte.touch import load_mnq, load_es, signal_grids_ext
from spx0dte.strategy import trade_detail, trade_ladder
from spx0dte.exits import minutes

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
ENTRIES = ["09:45", "10:00", "10:15", "10:30", "11:00"]
MONEY = [0, 0.001, 0.002, 0.003, 0.005]
AVGS = ["없음", "-50% 1번", "-40%·-70%"]
SIGS = ["구름", "나스닥 주도", "둘 다"]
ODD = pd.Timestamp("2025-04-09")

FP = pd.read_pickle(OUT / "f_paths.pkl")
mnq, es = load_mnq()[0], load_es()[0]
Gm = signal_grids_ext(mnq, FP)
Ge = signal_grids_ext(es, FP)
f_open = FP[list(FP.columns[:5])].bfill(axis=1).iloc[:, 0]


def open930(m):
    o = m.open[(m.index.hour == 9) & (m.index.minute == 30)]
    o.index = o.index.tz_localize(None).normalize()
    return o.reindex(FP.index)


mo, eo = open930(mnq), open930(es)
SIG = {}
for t in ENTRIES:
    up = FP[t] > f_open
    cloud = Gm["px5"][t] > Gm["top5"][t]
    lead = (Gm["px"][t] / mo - 1) > (Ge["px"][t] / eo - 1)
    SIG[t] = {"구름": up & cloud, "나스닥 주도": up & lead, "둘 다": up & cloud & lead}

C = pd.read_pickle(OUT / "calls_grid.pkl").set_index("date")
rows, per_trade = [], {}
for t, m in itertools.product(ENTRIES, MONEY):
    base = C[(C.entry == t) & (np.isclose(C.m, m))]
    add_last = minutes("14:30") - minutes(t)
    res = {a: [trade_detail(r, averaging=False) if a == "없음" else
               trade_ladder(r, 0.5, None, add_last=add_last) if a == "-50% 1번" else
               trade_detail(r, averaging=True, add_last=add_last) for r in base.itertuples()] for a in AVGS}
    for a in AVGS:
        full = pd.DataFrame({"pnl": [x["pnl"] for x in res[a]], "cost": [x["cost"] for x in res[a]]}, index=base.index)
        for s in SIGS:
            sel = SIG[t][s].reindex(full.index).fillna(False).astype(bool)
            tt = full[sel].sort_index()
            per_trade[(t, m, a, s)] = tt
            pnl = tt.pnl * 100; eq = pnl.cumsum(); dd = (eq - eq.cummax()).min() if len(eq) else 0
            rows.append({"진입": t, "행사가": f"+{m * 100:.1f}%" if m else "ATM", "m": m, "물타기": a, "신호": s, "거래": len(tt),
                         "총손익$": pnl.sum(), "4/9 빼고$": pnl.sum() - pnl.get(ODD, 0), "최고 3건 빼고$": pnl.sum() - pnl.nlargest(3).sum(),
                         "수익률%": tt.pnl.sum() / tt.cost.sum() * 100 if len(tt) else np.nan, "최대낙폭$": dd,
                         "점수(손익÷낙폭)": pnl.sum() / -dd if dd < 0 else np.nan, "2026$": pnl[pnl.index.year == 2026].sum(),
                         "플러스 연도": int((pnl.groupby(pnl.index.year).sum() > 0).sum())})
G = pd.DataFrame(rows)
G.round(2).to_csv(OUT / "O_grid.csv", index=False, encoding="utf-8-sig")

# ═══ 고원 지도: 진입 × 행사가 (점수), 물타기 × 신호별 ═══
fig, ax = plt.subplots(len(AVGS), len(SIGS), figsize=(18, 13))
for i, a in enumerate(AVGS):
    for j, s in enumerate(SIGS):
        h = G[(G.물타기 == a) & (G.신호 == s)]
        H = h.pivot(index="진입", columns="m", values="점수(손익÷낙폭)").reindex(index=ENTRIES, columns=MONEY)
        P = h.pivot(index="진입", columns="m", values="4/9 빼고$").reindex(index=ENTRIES, columns=MONEY)
        x = ax[i, j]
        x.imshow(H.values, cmap="RdYlGn", vmin=-3, vmax=8, aspect="auto")
        for p in range(H.shape[0]):
            for q in range(H.shape[1]):
                x.text(q, p, f"{H.values[p, q]:.1f}\n{P.values[p, q] / 1000:+.0f}k", ha="center", va="center", fontsize=8)
        x.set_xticks(range(len(MONEY)), ["ATM" if m == 0 else f"+{m * 100:.1f}%" for m in MONEY])
        x.set_yticks(range(len(ENTRIES)), ENTRIES)
        x.set_title(f"물타기 {a} · 신호 {s}\n(위: 손익÷낙폭, 아래: 4/9 뺀 총손익)", fontsize=10)
fig.suptitle("전체 기간 고원 지도 — 진입 시각(세로) × 행사가(가로). 주변까지 초록인 구역이 믿을 만한 곳")
fig.tight_layout(); fig.savefig(OUT / "O2_plateau.png", dpi=105); plt.close(fig)

# ═══ 걸어가며 검증 ═══
def year_stats(key, years):
    tt = per_trade[key]; tt = tt[tt.index.year.isin(years)]
    pnl = tt.pnl * 100; eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min() if len(eq) else 0
    return pnl.sum(), (pnl.sum() / -dd if dd < 0 else (np.inf if pnl.sum() > 0 else -np.inf)), len(tt)

wf, picks = [], []
DEFAULT = ("10:00", 0.002, "없음", "구름")
for Y in (2023, 2024, 2025, 2026):
    past = list(range(2022, Y))
    best, bs = None, -np.inf
    for key in per_trade:
        tot, sc, n = year_stats(key, past)
        if n >= 40 * len(past) * 0.5 and sc > bs:
            best, bs = key, sc
    got, _, n = year_stats(best, [Y])
    dflt, _, nd = year_stats(DEFAULT, [Y])
    picks.append({"적용 연도": Y, "과거로 고른 조합": f"{best[0]} · {'ATM' if best[1] == 0 else f'+{best[1] * 100:.1f}%'} · 물타기 {best[2]} · {best[3]}",
                  "과거 점수": bs, "그해 손익$": got, "그해 거래": n, "고정 기본(10:00 +0.2% 물타기 없음 구름) 손익$": dflt})
W = pd.DataFrame(picks)
W.round(1).to_csv(OUT / "O_walkforward.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(W))
ax.bar(x - .2, W["그해 손익$"] / 1000, .4, color="#0F6E5A", label="매년 초 과거로 고른 조합")
ax.bar(x + .2, W["고정 기본(10:00 +0.2% 물타기 없음 구름) 손익$"] / 1000, .4, color="#9AA39E", label="고정: 10:00 · +0.2% · 물타기 없음 · 구름")
for i, r in W.iterrows():
    ax.text(i, ax.get_ylim()[0], r["과거로 고른 조합"].replace(" · ", "\n"), ha="center", va="bottom", fontsize=7)
ax.set_xticks(x, [f"{y}{' (9월까지)' if y == 2026 else ''}" for y in W["적용 연도"]]); ax.axhline(0, color="k", lw=.8)
ax.set_ylabel("그해 손익 (천 달러, 1계약 기준)"); ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.25)
ax.set_title("걸어가며 검증: 매년 초에 과거만 보고 골랐다면")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "O3_walkforward.png", dpi=120); plt.close(fig)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 70)
show = ["진입", "행사가", "물타기", "신호", "거래", "총손익$", "4/9 빼고$", "최고 3건 빼고$", "수익률%", "최대낙폭$", "점수(손익÷낙폭)", "2026$", "플러스 연도"]
print("점수 상위 15:\n", G.sort_values("점수(손익÷낙폭)", ascending=False)[show].head(15).round(1).to_string(index=False))
print("\n기본안 (10:00, +0.2%):\n", G[(G.진입 == "10:00") & (np.isclose(G.m, 0.002))][show].round(1).to_string(index=False))
print("\n물타기별 평균 점수:\n", G.groupby("물타기")["점수(손익÷낙폭)"].mean().round(2).to_string())
print("\n신호별 평균 점수:\n", G.groupby("신호")["점수(손익÷낙폭)"].mean().round(2).to_string())
print("\n진입 시각별 평균 점수:\n", G.groupby("진입")["점수(손익÷낙폭)"].mean().round(2).to_string())
print("\n행사가별 평균 점수:\n", G.groupby("행사가", sort=False)["점수(손익÷낙폭)"].mean().round(2).to_string())
print("\n걸어가며 검증:\n", W.round(1).to_string(index=False))
print(f"\n시도한 조합: {len(G)}개")
