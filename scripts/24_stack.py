"""
조건 겹치기 + 물타기 (10:00 진입, 개장 30분 추이 기준). 23 결과 + MNQ 신호 사용, 1~2분.
  방향: 개장(09:31) → 10:00 SPX 올랐으면 콜, 내렸으면 풋
  조건(같은 방향일 때만): MNQ 60분봉 구름 / MNQ 5분봉 구름 / 5분 전환>기준 / 밤사이 갭 방향
  행사가: ATM · +5 · +10 · +20pt 외가격
  체결: 중간가 / 매도호가 (청산은 중간가 모드면 중간가, 매도호가 모드면 매수호가)
  운용: 1계약 만기 / -30% 손절 / 물타기 (첫 매수가 대비 -40%에 1개, -70%에 1개 추가, 14:30까지만, 전부 만기)
결과: output/S1_stack.png, S2_best.png, S3_days.png, S_summary.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.exits import FEE, EXERCISE
from spx0dte.touch import load_mnq, signal_grids_ext

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
SPLIT, ODD = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-09")
ADD_LEVELS, ADD_LAST = (0.6, 0.3), 270          # 물타기: 첫 매수가의 60%·30%까지 빠지면 추가, 진입 후 270분(14:30)까지만

# ── 10:00 시점 특징 ──
FP = pd.read_pickle(OUT / "f_paths.pkl")
G = signal_grids_ext(load_mnq()[0], FP)
px = load_spx_ohlc(SPX_CSV)
f_open = FP[list(FP.columns[:5])].bfill(axis=1).iloc[:, 0]
X = pd.DataFrame({"r30": FP["10:00"] / f_open - 1,
                  "gap": f_open / px.close.shift(1).reindex(FP.index) - 1,
                  "tr60": G["tr60"]["10:00"],
                  "c5": np.sign(G["px"]["10:00"] - G["top5"]["10:00"]).where(G["px"]["10:00"] > G["top5"]["10:00"],
                        -1.0 * (G["px"]["10:00"] < G["bot5"]["10:00"])),        # 1 구름 위, -1 아래, 0 안
                  "tk5": G["tk5"]["10:00"]})
X["dir"] = np.sign(X.r30)

FILTERS = {"기본(30분 방향만)": lambda x, s: x.dir == s,
           "+60분 구름": lambda x, s: (x.dir == s) & (x.tr60 == s),
           "+5분 구름": lambda x, s: (x.dir == s) & (x.c5 == s),
           "+갭 같은 방향": lambda x, s: (x.dir == s) & (np.sign(x.gap) == s),
           "+60분+5분": lambda x, s: (x.dir == s) & (x.tr60 == s) & (x.c5 == s),
           "+60분+갭": lambda x, s: (x.dir == s) & (x.tr60 == s) & (np.sign(x.gap) == s),
           "+60분+5분+전환": lambda x, s: (x.dir == s) & (x.tr60 == s) & (x.c5 == s) & (x.tk5 == s),
           "+전부(60·5·전환·갭)": lambda x, s: (x.dir == s) & (x.tr60 == s) & (x.c5 == s) & (x.tk5 == s) & (np.sign(x.gap) == s)}
PLANS = ["1계약 만기", "-30% 손절", "물타기"]
FILLS = ["중간가", "매도호가"]


def trade(r, plan, fill):
    """(손익 pt, 투입 pt, 추가 매수 분 인덱스들)"""
    p0 = r.mid if fill == "중간가" else r.ask
    exitp = r.mid_path if fill == "중간가" else r.bid_path
    settle = lambda p: r.pay - p - FEE - (EXERCISE if r.pay > 0 else 0)
    if plan == "1계약 만기":
        return settle(p0), p0, []
    if plan == "-30% 손절":
        hit = np.isfinite(exitp) & (exitp <= p0 * 0.7)
        if hit.any():
            i = int(hit.argmax()); return exitp[i] - p0 - 2 * FEE, p0, []
        return settle(p0), p0, []
    buys, adds, start = [p0], [], 0
    for lv in ADD_LEVELS:
        hit = np.isfinite(r.mid_path[start:ADD_LAST]) & (r.mid_path[start:ADD_LAST] <= p0 * lv)
        if not hit.any():
            break
        i = start + int(hit.argmax())
        p = r.mid_path[i] if fill == "중간가" else r.ask_path[i]
        if not np.isfinite(p):
            break
        buys.append(p); adds.append(i); start = i + 1
    return sum(settle(p) for p in buys), sum(buys), adds


L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date")
L = L.join(X, how="inner")
rows, keep = [], {}
for (side, leg), fname, off, plan, fill in itertools.product(((1, "C"), (-1, "P")), FILTERS, (0, 5, 10, 20), PLANS, FILLS):
    sub = L[(L.leg == leg) & (L.offset == off)]
    sub = sub[FILTERS[fname](sub, side)]
    if not len(sub):
        continue
    out = [trade(r, plan, fill) for r in sub.itertuples()]
    t = pd.DataFrame({"pnl": [o[0] for o in out], "cost": [o[1] for o in out], "n_add": [len(o[2]) for o in out]}, index=sub.index).sort_index()
    ret = lambda g: g.pnl.sum() / g.cost.sum() * 100 if len(g) else np.nan
    eq = (t.pnl * 100).cumsum()
    v = t[t.index >= SPLIT]
    key = ("콜" if leg == "C" else "풋", fname, off, plan, fill)
    keep[key] = t
    rows.append({"종류": key[0], "조건": fname, "외가격pt": off, "운용": plan, "체결": fill, "거래": len(t),
                 "1년 평균 거래": len(t) / (len(FP) / 252), "평균 투입$": t.cost.mean() * 100,
                 "학습%": ret(t[t.index < SPLIT]), "검증%": ret(v), "검증(4/9 제외)%": ret(v[v.index != ODD]),
                 "최고 10건 빼면%": ret(t.drop(t.pnl.nlargest(10).index)), "승률%": (t.pnl > 0).mean() * 100,
                 "물타기 발생%": (t.n_add > 0).mean() * 100, "총손익$": eq.iloc[-1], "1년 평균 손익$": eq.iloc[-1] / (len(FP) / 252),
                 "최대낙폭$": (eq - eq.cummax()).min(), "최악의 날$": t.pnl.min() * 100,
                 "플러스 연도": int((t.groupby(t.index.year).apply(ret) > 0).sum()), "연도 수": t.index.year.nunique()})
S = pd.DataFrame(rows)
S["둘 중 나쁜 쪽"] = S[["학습%", "검증(4/9 제외)%"]].min(axis=1)
S.round(1).to_csv(OUT / "S_summary.csv", index=False, encoding="utf-8-sig")

# ═══ S1: 콜, 중간가 체결 — 조건 × (행사가·운용) ═══
fig, ax = plt.subplots(1, 3, figsize=(26, 7.5))
cols = [f"{'ATM' if o == 0 else f'+{o}'}\n{p}" for o in (0, 5, 10, 20) for p in PLANS]
for j, col in enumerate(("학습%", "검증(4/9 제외)%", "1년 평균 거래")):
    h = S[(S.종류 == "콜") & (S.체결 == "중간가")].copy()
    h["열"] = [f"{'ATM' if o == 0 else f'+{o}'}\n{p}" for o, p in zip(h.외가격pt, h.운용)]
    H = h.pivot(index="조건", columns="열", values=col).reindex(index=list(FILTERS), columns=cols)
    ax[j].imshow(H.values, cmap="RdYlGn" if j < 2 else "Blues", vmin=-30 if j < 2 else 0, vmax=30 if j < 2 else 150, aspect="auto")
    for p in range(H.shape[0]):
        for q in range(H.shape[1]):
            ax[j].text(q, p, f"{H.values[p, q]:.0f}", ha="center", va="center", fontsize=8)
    ax[j].set_xticks(range(len(cols)), cols, fontsize=7); ax[j].set_yticks(range(len(FILTERS)), list(FILTERS), fontsize=9)
    ax[j].set_title(f"10:00 개장 30분 상승 → 콜 (중간가 체결) | {col}")
fig.tight_layout(); fig.savefig(OUT / "S1_stack.png", dpi=100); plt.close(fig)

# ═══ S2: 상위 조합 누적 손익 ═══
cand = S[(S.체결 == "중간가") & (S.거래 >= 60)].sort_values("둘 중 나쁜 쪽", ascending=False).head(5)
fig, ax = plt.subplots(1, 2, figsize=(18, 5.5))
for _, r in cand.iterrows():
    t = keep[(r.종류, r.조건, r.외가격pt, r.운용, r.체결)]
    lab = f"{r.종류} {r.조건} {'ATM' if r.외가격pt == 0 else '+' + str(r.외가격pt)} {r.운용} ({r.거래}건)"
    ax[0].plot(t.index, (t.pnl * 100).cumsum() / 1000, label=lab)
    t2 = keep[(r.종류, r.조건, r.외가격pt, r.운용, "매도호가")]
    ax[1].plot(t2.index, (t2.pnl * 100).cumsum() / 1000, label=lab)
for x, tt in zip(ax, ("중간가 체결", "매도호가 체결 (보수적)")):
    x.axhline(0, color="k", lw=1); x.axvline(SPLIT, color="grey", ls="--"); x.axvline(ODD, color="orange", ls=":")
    x.legend(fontsize=7); x.grid(alpha=.3); x.set_title(f"상위 5개 누적 손익 (천 달러) — {tt}")
fig.tight_layout(); fig.savefig(OUT / "S2_best.png", dpi=110); plt.close(fig)

# ═══ S3: 1등 조합 예시 날 (SPX 1분 + 매수·물타기 지점) ═══
b0 = cand.iloc[0]; key = (b0.종류, b0.조건, b0.외가격pt, b0.운용, "중간가")
t = keep[key]; leg = "C" if b0.종류 == "콜" else "P"
sub = L[(L.leg == leg) & (L.offset == b0.외가격pt)]
days = list(t.pnl.nlargest(2).index) + list(t.pnl.nsmallest(2).index) + list(t.index[-2:])
fig, ax = plt.subplots(3, 2, figsize=(17, 13))
cols_ = list(FP.columns); s0 = cols_.index("10:00")
for x, d in zip(ax.flat, days):
    r = sub.loc[d]
    x.plot(range(len(cols_)), FP.loc[d].values, color="black", lw=1)
    x.axhline(r.K, color="C0", ls=":", lw=1, label=f"행사가 {r.K:.0f}")
    x.scatter([s0], [FP.loc[d].values[s0]], marker="^" if leg == "C" else "v", color="C2", s=90, zorder=5, label="매수")
    _, _, adds = trade(r, b0.운용, "중간가")
    for i in adds:
        x.scatter([s0 + 1 + i], [FP.loc[d].values[s0 + 1 + i]], marker="o", color="orange", s=60, zorder=5)
    tk = [k for k, c in enumerate(cols_) if c.endswith(":00")]
    x.set_xticks(tk, [cols_[k] for k in tk], fontsize=7); x.grid(alpha=.3); x.legend(fontsize=7)
    x.set_title(f"{d:%Y-%m-%d}  {'콜' if leg == 'C' else '풋'} {b0.운용}  손익 ${t.pnl[d] * 100:+,.0f} (물타기 {len(adds)}번)", fontsize=10)
fig.suptitle(f"1등: {b0.종류} · {b0.조건} · {'ATM' if b0.외가격pt == 0 else '+' + str(b0.외가격pt) + 'pt'} · {b0.운용} (중간가)   ▲매수 ●물타기")
fig.tight_layout(); fig.savefig(OUT / "S3_days.png", dpi=105); plt.close(fig)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
show = ["종류", "조건", "외가격pt", "운용", "체결", "거래", "1년 평균 거래", "평균 투입$", "학습%", "검증%", "검증(4/9 제외)%",
        "최고 10건 빼면%", "승률%", "물타기 발생%", "1년 평균 손익$", "최대낙폭$", "최악의 날$", "플러스 연도", "연도 수"]
print(S[(S.체결 == "중간가") & (S.거래 >= 60)].sort_values("둘 중 나쁜 쪽", ascending=False)[show].head(15).round(1).to_string(index=False))
print("\n같은 조합, 매도호가 체결이면:")
for _, r in cand.iterrows():
    q = S[(S.종류 == r.종류) & (S.조건 == r.조건) & (S.외가격pt == r.외가격pt) & (S.운용 == r.운용) & (S.체결 == "매도호가")].iloc[0]
    print(f"  {r.종류} {r.조건} +{r.외가격pt} {r.운용}: 학습 {q['학습%']:.1f} / 검증(4/9제외) {q['검증(4/9 제외)%']:.1f}")
print("\n운용 방식별 평균 (콜, 중간가):\n", S[(S.종류 == "콜") & (S.체결 == "중간가")].groupby("운용")[["학습%", "검증(4/9 제외)%", "최대낙폭$"]].mean().round(1).to_string())
print("\n조건별 평균 (콜, 중간가):\n", S[(S.종류 == "콜") & (S.체결 == "중간가")].groupby("조건", sort=False)[["거래", "학습%", "검증(4/9 제외)%"]].mean().round(1).to_string())
print("\n풋 조건별 평균 (중간가):\n", S[(S.종류 == "풋") & (S.체결 == "중간가")].groupby("조건", sort=False)[["학습%", "검증(4/9 제외)%"]].mean().round(1).to_string())
print(f"\n시도한 칸: {len(S)}개 (콜·풋 2 × 조건 8 × 행사가 4 × 운용 3 × 체결 2)")
