"""
현재 규칙 v2 (2026-09-27): 10:00 조건 두 개 → ATM 콜 1계약 중간가 → −90% 손절, 아니면 만기.
체결 지연 0초(10:00:00 호가) / 60초(10:01:00 호가), 중간가 / 매도호가, 손절 유무 비교.
+ 2020-01 ~ 2022-05 표본외 (data/0dte_oos) 에도 같은 규칙 적용 (전체 / 월·수·금만).
결과: output/rule_v2/summary.json, trades.csv, R1_equity.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, os, warnings
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import numpy as np, pandas as pd, databento as db
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV, DATA
from spx0dte.core import NY, load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_rule, STOP
from spx0dte.straddle import _prep

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
R = OUT / "rule_v2"; R.mkdir(exist_ok=True)
ODD = pd.Timestamp("2025-04-09")


def stats(T):
    p = T.pnl * 100; eq = p.cumsum(); dd = eq - eq.cummax()
    under = (eq < eq.cummax()).astype(int).values; run = best = 0; end = 0
    for i, u in enumerate(under):
        run = run + 1 if u else 0
        if run > best:
            best, end = run, i
    top3 = p.drop(p.nlargest(3).index)
    return {"매수": len(p), "총손익$": p.sum(), "4/9 빼고$": p.sum() - p.get(ODD, 0), "최고3건 빼고$": top3.sum(), "건당$": p.mean(),
            "수익률%": p.sum() / (T.cost.sum() * 100) * 100, "승률%": (p > 0).mean() * 100, "평균 투입$": T.cost.mean() * 100,
            "최대낙폭$": dd.min(), "최악의 날$": p.min(), "최고의 날$": p.max(), "손절 비율%": T.stop.notna().mean() * 100,
            "가장 긴 손실 구간(거래)": best, "손실 구간": f"{p.index[end - best + 1]:%Y-%m} ~ {p.index[end]:%Y-%m}" if best else "",
            "플러스 연도": int((p.groupby(p.index.year).sum() > 0).sum()),
            "연도별$": {str(y): v for y, v in p.groupby(p.index.year).sum().items()},
            "연도별 건수": {str(y): int(v) for y, v in p.groupby(p.index.year).size().items()},
            "월별$": {f"{y}-{m:02d}": v for (y, m), v in p.groupby([p.index.year, p.index.month]).sum().items()}}


def oos_day(args):
    f, k, close = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    M, bid, ask, cp, K, F = _prep(db.DBNStore.from_file(str(f)).to_df(), d)
    col = M.columns[(K == k) & (cp == "C")][0]
    t10 = pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=NY); after = M.index > t10
    fw = F.ffill()[after].values; intr = np.maximum(fw - k, 0)
    b = bid.loc[after, col].values; m = M.loc[after, col].values; a = ask.loc[after, col].values
    return {"date": d, "mid": M.at[t10, col], "ask": ask.at[t10, col], "pay": max(close - k, 0.0),
            "mid_path": np.where(np.isfinite(m), m, intr), "bid_path": np.where(np.isfinite(b), b, intr), "ask_path": a}


if __name__ == "__main__":
    FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
    X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
    L = pd.read_pickle(OUT / "legs10_full.pkl")
    L = L[(L.leg == "C") & (L.offset == 0) & (L.entry == "10:00")].set_index("date").join(X[["signal"]], how="inner")
    L = L[L.signal].sort_index()
    out, trades = {}, {}
    for lag in (0, 1):
        for fill in ("중간가", "매도호가"):
            for stop in (STOP, None):
                nm = f"지연 {lag * 60}초 · {fill} · {'손절 −90%' if stop else '손절 없음'}"
                T = pd.DataFrame([trade_rule(r, stop=stop, fill=fill, lag=lag) for r in L.itertuples()], index=L.index)
                out[nm] = stats(T); trades[nm] = T
    main = "지연 0초 · 중간가 · 손절 −90%"
    T = trades[main].copy(); T["K"] = L.K; T["mid"] = L.mid; T["ask"] = L.ask
    T["손절 시각"] = [pd.Timestamp("2000-01-01 10:01") + pd.Timedelta(minutes=s[0]) if s else pd.NaT for s in T.stop]
    T.drop(columns=["buys"]).to_csv(R / "trades.csv", encoding="utf-8-sig")

    # 표본외 2020-01 ~ 2022-05
    O = pd.read_csv(OUT / "oos" / "days.csv", index_col=0, parse_dates=True)
    O = O[(O["신호"] == True) & O["행사가"].notna()]
    jobs = [(DATA / "0dte_oos" / f"{d:%Y-%m-%d}.dbn.zst", float(r["행사가"]), float(r["SPX 종가"])) for d, r in O.iterrows()]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        OR = [SimpleNamespace(**x) for x in ex.map(oos_day, jobs, chunksize=4)]
    oos = {}
    for lag in (0, 1):
        for stop in (STOP, None):
            T2 = pd.DataFrame([trade_rule(r, stop=stop, lag=lag) for r in OR], index=[r.date for r in OR]).sort_index()
            for tag, m in (("전체", np.ones(len(T2), bool)), ("월·수·금만", T2.index.dayofweek.isin([0, 2, 4]))):
                t = T2[m]; p = t.pnl * 100
                oos[f"지연 {lag * 60}초 · {'손절 −90%' if stop else '손절 없음'} · {tag}"] = {
                    "매수": len(t), "총손익$": p.sum(), "건당$": p.mean(), "수익률%": p.sum() / (t.cost.sum() * 100) * 100,
                    "손절 비율%": t.stop.notna().mean() * 100, "연도별$": {str(y): v for y, v in p.groupby(p.index.year).sum().items()}}
    json.dump({"backtest": out, "oos": oos}, open(R / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    pd.set_option("display.width", 250)
    cols = ["매수", "총손익$", "4/9 빼고$", "최고3건 빼고$", "건당$", "수익률%", "승률%", "최대낙폭$", "최악의 날$", "손절 비율%", "가장 긴 손실 구간(거래)", "손실 구간", "플러스 연도"]
    print(pd.DataFrame(out).T[cols].to_string())
    print(pd.DataFrame({k: v["연도별$"] for k, v in out.items()}).T.round(0).to_string())
    print("\n표본외 2020-01 ~ 2022-05")
    print(pd.DataFrame({k: {**{kk: vv for kk, vv in v.items() if kk != "연도별$"}, **v["연도별$"]} for k, v in oos.items()}).T.round(1).to_string())

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for nm, col, ls in ((main, "#0F6E5A", "-"), ("지연 0초 · 중간가 · 손절 없음", "#9AA3AD", "-"),
                        ("지연 60초 · 중간가 · 손절 −90%", "#0F6E5A", ":"), ("지연 60초 · 매도호가 · 손절 −90%", "#B3261E", ":")):
        eq = (trades[nm].pnl * 100).cumsum() / 1000
        ax.plot(eq.index, eq.values, color=col, ls=ls, lw=1.4, label=f"{nm}  {eq.iloc[-1]:+.1f}k".replace("−", "-"))
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("누적 손익 (천 달러)"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("R1 현재 규칙 (-90% 손절) — 체결 지연·체결가별 누적 손익, ATM 콜 1계약")
    fig.tight_layout(); fig.savefig(R / "R1_equity.png", dpi=120)
