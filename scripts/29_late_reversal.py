"""
15시 이후 역매매: MNQ 1분·5분·60분봉이 모두 일목 구름 위 → ATM 풋 매수 / 모두 아래 → ATM 콜 매수.
  15:00~15:45 사이 첫 신호 1번만, 물타기 없음, 만기 보유. 비교용으로 같은 자리 '순매매'(위 → 콜)도 계산.
  판정: 마지막 완성 봉 종가 > 구름 윗선 (1분·5분), 60분봉 상태 = 구름 위. 봉 끝난 다음 분 SPX 호가로 매수.
결과: output/late_trades.pkl, V1_late.png, V2_late_days.png, V_late_summary.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OPT_DIR, OUT, SPX_CSV
from spx0dte.core import load_spx_close, load_day
from spx0dte.straddle import _prep, pick_atm
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.exits import FEE, EXERCISE

FIRST, LAST = "15:00", "15:45"


def run(args):
    f, close, S = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    try:
        cols = S["cols"]
        i0, i1 = cols.index(FIRST), cols.index(LAST)
        up = (S["px1"] > S["top1"]) & (S["px5"] > S["top5"]) & (S["tr60"] == 1)
        dn = (S["px1"] < S["bot1"]) & (S["px5"] < S["bot5"]) & (S["tr60"] == -1)
        hit = [i for i in range(i0, i1 + 1) if up[i] or dn[i]]
        if not hit:
            return None
        i = hit[0]
        M, bid, ask, cp, K, F = _prep(load_day(d), d)
        ts = M.index[i]
        best = pick_atm(M, ask, cp, K, S["F"][i], ts)
        if best is None:
            return None
        k, c, p = best
        out = {"date": d, "time": cols[i], "trend": "상승" if up[i] else "하락", "K": k, "F": S["F"][i], "close": close}
        for leg, col in (("C", c), ("P", p)):
            out[f"{leg}_ask"], out[f"{leg}_mid"] = ask.at[ts, col], M.at[ts, col]
            out[f"{leg}_pay"] = max(close - k, 0.0) if leg == "C" else max(k - close, 0.0)
            out[f"{leg}_midpath"] = M[col].values[i + 1:].astype("float32")
        out["Fpath"] = S["F"][i - 60: ].astype("float32")          # 14시~ SPX (차트용)
        return out
    except Exception as ex:
        return f"{f.name[:10]} 실패: {ex}"


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
    t0 = time.time()
    FP = pd.read_pickle(OUT / "f_paths.pkl")
    G = signal_grids_ext(load_mnq()[0], FP)
    close = load_spx_close(SPX_CSV)
    cols = list(FP.columns)
    jobs = []
    for f in sorted(OPT_DIR.glob("*.dbn.zst")):
        d = pd.Timestamp(f.name[:10])
        if d in close.index and d in FP.index:
            S = {k: v.loc[d].values.astype(float) for k, v in G.items()}; S["cols"] = cols
            jobs.append((f, float(close[d]), S))
    rows = []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for res in ex.map(run, jobs, chunksize=4):
            if isinstance(res, str):
                print(res)
            elif res:
                rows.append(res)
    T = pd.DataFrame(rows).set_index("date").sort_index()
    T.to_pickle(OUT / "late_trades.pkl")
    print(f"신호 난 날 {len(T)} / {len(jobs)}일 ({time.time() - t0:.0f}초)")

    # ── 손익: 역매매(상승 → 풋) vs 순매매(상승 → 콜), 중간가·매도호가 ──
    res, curves = [], {}
    for mode in ("역매매", "순매매"):
        leg = np.where((T.trend == "상승") == (mode == "역매매"), "P", "C")
        for fill in ("mid", "ask"):
            cost = np.where(leg == "C", T[f"C_{fill}"], T[f"P_{fill}"])
            pay = np.where(leg == "C", T.C_pay, T.P_pay)
            pnl = pd.Series(pay - cost - FEE - np.where(pay > 0, EXERCISE, 0), index=T.index)
            c = pd.Series(cost, index=T.index)
            eq = (pnl * 100).cumsum(); dd = (eq - eq.cummax()).min()
            y = (pnl * 100).groupby(T.index.year).sum()
            res.append({"방식": mode, "체결": "중간가" if fill == "mid" else "매도호가", "거래": len(T),
                        "평균 투입$": c.mean() * 100, "수익률%": pnl.sum() / c.sum() * 100, "승률%": (pnl > 0).mean() * 100,
                        "총손익$": pnl.sum() * 100, "최고 3건 빼고$": (pnl.sum() - pnl.nlargest(3).sum()) * 100,
                        "최대낙폭$": dd, "최악의 날$": pnl.min() * 100, "최고의 날$": pnl.max() * 100,
                        **{f"{yy}$": v for yy, v in y.items()},
                        "상승→풋/콜 수익률%": pnl[T.trend == "상승"].sum() / c[T.trend == "상승"].sum() * 100,
                        "하락→콜/풋 수익률%": pnl[T.trend == "하락"].sum() / c[T.trend == "하락"].sum() * 100})
            if fill == "mid":
                curves[mode] = eq
                T[f"{mode}_leg"], T[f"{mode}_pnl"], T[f"{mode}_cost"] = leg, pnl, c
    R = pd.DataFrame(res)
    R.round(1).to_csv(OUT / "V_late_summary.csv", index=False, encoding="utf-8-sig")
    tm = T.time.value_counts().sort_index()

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    for mode, col in (("역매매", "#0F6E5A"), ("순매매", "#B3261E")):
        ax[0].plot(curves[mode].index, curves[mode].values / 1000, color=col, label=mode)
    ax[0].axhline(0, color="k", lw=.8); ax[0].legend(); ax[0].grid(alpha=.25)
    ax[0].set_title("15시 이후 첫 신호, ATM 1계약, 만기 보유 — 누적 손익 (천 달러)")
    yy = [c for c in R.columns if c.endswith("$") and c[:4].isdigit()]
    for j, mode in enumerate(("역매매", "순매매")):
        v = R[(R.방식 == mode) & (R.체결 == "중간가")][yy].iloc[0] / 1000
        ax[1].bar(np.arange(len(yy)) + (j - .5) * .38, v.values, .38, color="#0F6E5A" if j == 0 else "#B3261E", label=mode)
    ax[1].set_xticks(range(len(yy)), [c[:4] for c in yy]); ax[1].axhline(0, color="k", lw=.8); ax[1].legend()
    ax[1].set_title("연도별 손익 (천 달러)")
    ax[2].bar(range(len(tm)), tm.values, color="#9AA39E"); ax[2].set_xticks(range(len(tm))[::5], tm.index[::5], rotation=45)
    ax[2].set_title("신호가 처음 난 시각 분포 (건수)")
    for a_ in ax:
        for s in ("top", "right"): a_.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "V1_late.png", dpi=120); plt.close(fig)

    # ── 예시 날: 좋은 날 2, 나쁜 날 2, 최근 2 (역매매) ──
    p = T["역매매_pnl"]
    days = list(p.nlargest(2).index) + list(p.nsmallest(2).index) + list(p.index[-2:])
    fig, ax = plt.subplots(3, 2, figsize=(16, 12))
    s14 = cols.index("14:00")
    for x, d in zip(ax.flat, days):
        r = T.loc[d]
        fpath = FP.loc[d].values
        x.plot(range(s14, len(cols)), fpath[s14:], color="black", lw=1)
        i = cols.index(r.time)
        x.axhline(r.K, color="C0", ls=":", lw=1)
        leg = r["역매매_leg"]
        x.scatter([i], [fpath[i]], marker="v" if leg == "P" else "^", color="#7B2CBF" if leg == "P" else "#2A9D8F", s=90, zorder=5)
        tk = [k for k in range(s14, len(cols)) if cols[k].endswith(":00") or cols[k].endswith(":30")]
        x.set_xticks(tk, [cols[k] for k in tk], fontsize=8); x.grid(alpha=.3)
        x.set_title(f"{d:%Y-%m-%d} {r.time} MNQ 3개 봉 모두 {r.trend} → {'풋' if leg == 'P' else '콜'} {r.K:.0f} @ {r[leg + '_mid']:.2f}\n"
                    f"종가 {r.close:,.2f} → 정산 {r[leg + '_pay']:.2f}, 손익 ${p[d] * 100:+,.0f}", fontsize=9)
    fig.suptitle("15시 이후 역매매 예시 (SPX 1분, 14:00~16:00, 점선 = 행사가)")
    fig.tight_layout(); fig.savefig(OUT / "V2_late_days.png", dpi=110); plt.close(fig)

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(R.round(1).to_string(index=False))
    print("\n2026년 월별 (역매매, 중간가):\n", (T["역매매_pnl"][T.index.year == 2026] * 100).groupby(T.index[T.index.year == 2026].month).agg(["count", "sum"]).round(0).to_string())
