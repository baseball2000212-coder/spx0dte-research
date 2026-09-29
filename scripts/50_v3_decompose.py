"""
V3(0DTE ATM IV ÷ 전날 VIX9D) 비판 점검 — 2020-01 ~ 2024-12만 (금고 안 엶).
사용자 비판: VIX9D는 전 행사가 가중(스큐 포함)·9일(밤·주말·이벤트 포함)이라 ATM 0DTE IV와 같은 잣대가 아님.
→ 비율이 실제로 뭘 재는지 분해: ATM IV 수준, 자기 과거 대비(20일 중앙값), 스큐, VIX9D, 9일 안 이벤트, 전날 VIX1D.
결과: output/holdout/V3_decompose.csv, V3_decompose.png, oos10.pkl(2020~22.5 10:00 피처 캐시)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, databento as db
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import DATA, SPX_CSV
from spx0dte.core import NY, YEAR_MIN, load_spx_ohlc, implied_vol
from spx0dte.straddle import _prep, pick_atm
from spx0dte.exits import FEE, EXERCISE
from spx0dte.holdout import DIR
from spx0dte.events import EVENTS
from spx0dte import vix

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
TU = np.sqrt(390 * 252 / 525600) * 100
E3 = 0.6401


def feat10(args):
    f, close = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    try:
        M, bid, ask, cp, K, F = _prep(db.DBNStore.from_file(str(f)).to_df(), d)
        ts = pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=NY); fv = float(F[ts])
        best = pick_atm(M, ask, cp, K, fv, ts)
        if best is None:
            return None
        k, c, p = best; T = 360 / YEAR_MIN
        iv = lambda col, kk, call: float(implied_vol(np.array([M.at[ts, col]]), np.array([fv]), np.array([kk]), np.array([T]), np.array([call]))[0])
        r = {"date": d, "c_ask": ask.at[ts, c], "p_ask": ask.at[ts, p], "c_pay": max(close - k, 0.0), "p_pay": max(k - close, 0.0),
             "iv_atm": np.nanmean([iv(c, k, True), iv(p, k, False)])}
        for leg, sgn in (("c", 1), ("p", -1)):
            ks = np.array(sorted(kk for kk, x, col in zip(K, cp, M.columns) if x == leg.upper() and np.isfinite(M.at[ts, col])))
            kk = float(ks[np.abs(ks - fv * (1 + sgn * 0.005)).argmin()])
            col = M.columns[(K == kk) & (cp == leg.upper())][0]
            r[f"{leg}5_iv"] = iv(col, kk, leg == "c")
        return r
    except Exception:
        return None


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    px = load_spx_ohlc(SPX_CSV)
    cache = DIR / "oos10.pkl"
    if cache.exists():
        O = pd.read_pickle(cache)
    else:
        files = sorted((DATA / "0dte_oos").glob("*.dbn.zst"))
        jobs = [(f, float(px.close[pd.Timestamp(f.name[:10])])) for f in files if pd.Timestamp(f.name[:10]) in px.index]
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
            O = pd.DataFrame([r for r in ex.map(feat10, jobs, chunksize=4) if r]).set_index("date")
        O.to_pickle(cache)
    dc = pd.read_pickle(DIR / "disc_chain.pkl")["chain"]
    t = dc[dc.entry == "10:00"].set_index("date")[["c_ask", "p_ask", "c_pay", "p_pay", "iv_atm", "c5_iv", "p5_iv"]]
    X = pd.concat([O, t]).sort_index(); X = X[~X.index.duplicated()]
    assert X.index.max() < pd.Timestamp("2025-01-01")

    V = vix.load(("VIX", "VIX9D", "VIX1D")); prev = V.shift(1)
    X["ivt"] = X.iv_atm * TU                                   # 10:00 ATM IV (거래시간, %)
    X["v9"] = prev.VIX9D.reindex(X.index); X["v1"] = prev.VIX1D.reindex(X.index)
    X["ratio"] = X.ivt / X.v9
    X["skew"] = (X.p5_iv - X.c5_iv) / X.iv_atm                 # 상대 스큐
    X["iv_self"] = X.ivt / X.ivt.shift(1).rolling(20, min_periods=10).median()   # 자기 과거 20회 중앙값 대비
    X["iv_v1"] = X.ivt / X.v1
    ev = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in EVENTS.values()])))
    X["event9"] = [bool(((ev > d) & (ev <= d + pd.Timedelta(days=9))).any()) for d in X.index]
    X["event_today"] = X.index.isin(ev)
    pay = X.c_pay + X.p_pay; cost = X.c_ask + X.p_ask
    X["pnl"] = pay - cost - 2 * FEE - EXERCISE * (pay > 0); X["cost"] = cost
    X["sig"] = X.ratio <= E3
    ret = lambda g: g.pnl.sum() / g.cost.sum() * 100 if len(g) else np.nan

    pd.set_option("display.width", 250)
    print(f"표본 {len(X)}일 (2020-01~2024-12, 2022-05 전은 월·수·금)")
    print("\n[1] 비율이 뭐랑 같이 움직이나 (순위상관)")
    print(X[["ratio", "ivt", "iv_self", "skew", "v9", "iv_v1", "event9"]].astype(float).corr("spearman")["ratio"].round(2).to_string())

    print("\n[2] 피처별 3분위 → 10:00 스트래들 수익률 % (연도별, 분위 경계는 연도마다 따로 = 장세 수준 차이 제거)")
    rows = []
    for feat in ("ratio", "ivt", "iv_self", "skew", "v9", "iv_v1"):
        for y in range(2020, 2025):
            g = X[(X.index.year == y) & X[feat].notna()]
            if len(g) < 30:
                continue
            q = pd.qcut(g[feat], 3, labels=["낮음", "중간", "높음"])
            rows.append({"피처": feat, "연도": y, **{k: ret(g[q == k]) for k in ("낮음", "중간", "높음")}})
    T3 = pd.DataFrame(rows); T3["낮음−높음"] = T3["낮음"] - T3["높음"]
    print(T3.round(1).to_string(index=False))
    print("\n  낮음이 높음보다 나은 해 수:", T3.groupby("피처")["낮음−높음"].apply(lambda s: f"{(s > 0).sum()}/{len(s)}").to_string())

    print("\n[3] 이중 분류: ATM IV 자기 과거 대비(iv_self) 3분위 안에서, 비율 신호일 vs 나머지")
    g = X.dropna(subset=["iv_self"]); qi = pd.qcut(g.iv_self, 3, labels=["IV 자기대비 낮음", "중간", "높음"])
    D2 = pd.DataFrame({k: {"신호일 %": ret(g[(qi == k) & g.sig]), "나머지 %": ret(g[(qi == k) & ~g.sig]),
                           "신호일 수": int(((qi == k) & g.sig).sum()), "나머지 수": int(((qi == k) & ~g.sig).sum())} for k in qi.cat.categories}).T
    print(D2.round(1).to_string())
    print("\n[4] 이중 분류: 스큐 3분위 안에서, 비율 신호일 vs 나머지")
    qs = pd.qcut(X["skew"], 3, labels=["스큐 낮음", "중간", "높음"])
    D3 = pd.DataFrame({k: {"신호일 %": ret(X[(qs == k) & X.sig]), "나머지 %": ret(X[(qs == k) & ~X.sig]),
                           "신호일 수": int(((qs == k) & X.sig).sum())} for k in qs.cat.categories}).T
    print(D3.round(1).to_string())
    print("\n[5] 이벤트 (2022-05~2024): 9일 안 이벤트 있음/없음 × 신호")
    g = X[X.index >= "2022-05-16"]
    D4 = pd.DataFrame({nm: {"신호일 %": ret(g[m & g.sig]), "나머지 %": ret(g[m & ~g.sig]), "신호일 비율": (g[m].sig).mean()}
                       for nm, m in (("9일 안 이벤트 있음", g.event9), ("없음", ~g.event9), ("오늘이 이벤트", g.event_today))}).T
    print(D4.round(2).to_string())
    T3.to_csv(DIR / "V3_decompose.csv", index=False, encoding="utf-8-sig")
    X.to_pickle(DIR / "v3_features.pkl")

    fig, axs = plt.subplots(1, 3, figsize=(19, 5))
    piv = T3.pivot(index="연도", columns="피처", values="낮음−높음")[["ratio", "iv_self", "ivt", "skew", "v9", "iv_v1"]]
    piv.plot.bar(ax=axs[0], rot=0); axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_title("피처 '낮음' − '높음' 3분위 스트래들 수익률 차 (%p)\n+ 면 낮을 때 사는 게 유리"); axs[0].legend(fontsize=8)
    D2[["신호일 %", "나머지 %"]].plot.bar(ax=axs[1], rot=0, color=["#B23A48", "#9AA3AD"]); axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_title("ATM IV 자기 과거 대비 같은 구간 안에서 V3 신호 효과")
    D3[["신호일 %", "나머지 %"]].plot.bar(ax=axs[2], rot=0, color=["#B23A48", "#9AA3AD"]); axs[2].axhline(0, color="k", lw=0.6)
    axs[2].set_title("스큐 같은 구간 안에서 V3 신호 효과")
    fig.suptitle("V3 분해 (2020~2024, 10:00 ATM 스트래들, 매도호가)"); fig.tight_layout(); fig.savefig(DIR / "V3_decompose.png", dpi=110)
