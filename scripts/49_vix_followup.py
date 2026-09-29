"""
VIX 탐색 2단계 확인 (2020-01 ~ 2022-12, 떼어둔 구간 아님). 경계는 2023-24 분위로 고정.
  V4 후보: 전날 VIX 변화율 ≤ 2023-24 20% 분위 → 10:00 ATM 콜
  V3 근접(단조 −0.8로 기준 미달, 참고): 10:00 0DTE ATM IV(거래시간) ÷ 전날 VIX9D ≤ 60% 분위 → 10:00 ATM 스트래들
  확인 자료: ES 10:00→15:59 (2020-22), 옵션 2020-01~2022-05 월·수·금(data/0dte_oos) + 2022-05~12(disc_chain)
결과: output/holdout/V2_followup.csv, V2_followup.png
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
from spx0dte import vix

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
TRADE_UNIT = np.sqrt(390 * 252 / 525600)
END = pd.Timestamp("2022-12-31")


def opt10(args):
    f, close = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    try:
        M, bid, ask, cp, K, F = _prep(db.DBNStore.from_file(str(f)).to_df(), d)
        ts = pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=NY); fv = float(F[ts])
        best = pick_atm(M, ask, cp, K, fv, ts)
        if best is None:
            return None
        k, c, p = best
        T = 360 / YEAR_MIN
        ivs = [float(implied_vol(np.array([M.at[ts, x]]), np.array([fv]), np.array([k]), np.array([T]), np.array([call]))[0]) for x, call in ((c, True), (p, False))]
        return {"date": d, "c_ask": ask.at[ts, c], "p_ask": ask.at[ts, p], "c_pay": max(close - k, 0.0), "p_pay": max(k - close, 0.0), "iv_atm": np.nanmean(ivs)}
    except Exception:
        return None


def ret(pay, cost, legs):
    pnl = pay - cost - legs * FEE - EXERCISE * (pay > 0)
    return pnl.sum() / cost.sum() * 100 if len(cost) else np.nan, pnl.sum() * 100, len(cost)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    V = vix.load(); prev = V.shift(1); chg = V.VIX.pct_change().shift(1)
    # 경계 (2023-24로 고정)
    dc = pd.read_pickle(DIR / "disc_chain.pkl")["chain"]; t = dc[dc.entry == "10:00"].set_index("date")
    t["vix_chg"] = chg.reindex(t.index); t["iv_rel"] = t.iv_atm * TRADE_UNIT * 100 / prev.VIX9D.reindex(t.index)
    d23 = t[t.index.year >= 2023]
    E4, E3 = d23.vix_chg.quantile(0.2), d23.iv_rel.quantile(0.6)
    print(f"경계: 전날 VIX 변화 ≤ {E4:+.4f}, IV÷VIX9D ≤ {E3:.4f}")

    # ES 10:00→15:59 (2020-22)
    p = pd.read_parquet(DATA / "es_1m.parquet"); p = p[p.symbol == "ES.v.0"].sort_index(); p.index = p.index.tz_convert(NY)
    p = p[p.index < pd.Timestamp("2023-01-01", tz=NY)]
    px = load_spx_ohlc(SPX_CSV); days = px.index[(px.index >= "2020-01-01") & (px.index <= END)]
    es = {}
    for d in days:
        a = p.index.searchsorted(pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=NY)); b = p.index.searchsorted(pd.Timestamp(f"{d:%Y-%m-%d} 15:59", tz=NY))
        if b > a and p.instrument_id.iloc[a] == p.instrument_id.iloc[b]:
            es[d] = p.close.iloc[b] / p.close.iloc[a] - 1
    es = pd.Series(es); sig = chg.reindex(es.index) <= E4
    esT = pd.DataFrame({y: {"신호일 평균bp": es[sig & (es.index.year == y)].mean() * 1e4, "나머지 평균bp": es[~sig & (es.index.year == y)].mean() * 1e4,
                            "신호일 상승비율": (es[sig & (es.index.year == y)] > 0).mean(), "신호일수": int((sig & (es.index.year == y)).sum())}
                        for y in (2020, 2021, 2022)}).T
    print("\nV4 ES 10:00→15:59\n", esT.round(2).to_string())

    # 옵션: 2020-01~2022-05 월·수·금 + 2022-05~12
    files = sorted((DATA / "0dte_oos").glob("*.dbn.zst"))
    jobs = [(f, float(px.close[pd.Timestamp(f.name[:10])])) for f in files if pd.Timestamp(f.name[:10]) in px.index]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        O = pd.DataFrame([r for r in ex.map(opt10, jobs, chunksize=4) if r]).set_index("date")
    O = pd.concat([O, t[t.index.year == 2022][["c_ask", "p_ask", "c_pay", "p_pay", "iv_atm"]]]).sort_index()
    O["vix_chg"] = chg.reindex(O.index); O["iv_rel"] = O.iv_atm * TRADE_UNIT * 100 / prev.VIX9D.reindex(O.index)
    rows = []
    for y in (2020, 2021, 2022, "전체"):
        o = O if y == "전체" else O[O.index.year == y]
        s4, n4 = o[o.vix_chg <= E4], o[o.vix_chg > E4]
        s3, n3 = o[o.iv_rel <= E3], o[o.iv_rel > E3]
        r = {"연도": y}
        for nm, g, kind in (("V4 신호 콜", s4, "c"), ("V4 나머지 콜", n4, "c"), ("V3 싼날 스트래들", s3, "s"), ("V3 나머지 스트래들", n3, "s")):
            if kind == "c":
                pct, usd, n = ret(g.c_pay, g.c_ask, 1)
            else:
                pct, usd, n = ret(g.c_pay + g.p_pay, g.c_ask + g.p_ask, 2)
            r[f"{nm} %"] = pct; r[f"{nm} $"] = usd; r[f"{nm} 건"] = n
        rows.append(r)
    R = pd.DataFrame(rows).set_index("연도")
    pd.set_option("display.width", 250)
    print("\n옵션 10:00 ATM (매도호가, 2020~2022.5는 월·수·금만)\n", R.round(1).to_string())
    pd.concat({"ES": esT, "옵션": R}).to_csv(DIR / "V2_followup.csv", encoding="utf-8-sig")

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    yrs = ["2020", "2021", "2022"]
    for ax, a, b, ttl in ((axs[0], "V4 신호 콜 %", "V4 나머지 콜 %", "V4 전날 VIX 크게 하락 → 10:00 ATM 콜"),
                          (axs[1], "V3 싼날 스트래들 %", "V3 나머지 스트래들 %", "V3 0DTE IV÷VIX9D 낮은 날 → 10:00 스트래들 (참고)")):
        x = np.arange(3); v = R.loc[[2020, 2021, 2022]]
        ax.bar(x - 0.2, v[a], 0.4, color="#B23A48", label="신호일"); ax.bar(x + 0.2, v[b], 0.4, color="#9AA3AD", label="나머지")
        ax.set_xticks(x, yrs); ax.axhline(0, color="k", lw=0.6); ax.set_title(ttl); ax.set_ylabel("수익률 % (매도호가)"); ax.legend()
    fig.suptitle("V2 2단계 확인: 2020~2022 (한 번도 이 아이디어로 안 본 기간)"); fig.tight_layout(); fig.savefig(DIR / "V2_followup.png", dpi=110)
