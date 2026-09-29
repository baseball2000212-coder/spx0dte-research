"""
탐색 2단계: 2023-24에서 눈에 띈 두 가지를 2020-01 ~ 2022-12(떼어둔 구간 아님)로 확인.
  ① 목요일 약세: 선물(ES) 09:45→15:59 수익률 요일별 (옵션은 2022-05 전 목요일 만기가 없어서 지수로만)
  ② 밤사이 MNQ가 적당히 오른 날(+0.18% ~ +0.53%, 2023-24 7~8분위 경계 고정): 선물 방향 + 2020-01~2022-05 월·수·금 옵션(data/0dte_oos) 09:45 ATM 콜·풋
결과: output/holdout/H2_followup.csv, H2_followup.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, databento as db
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import DATA, FUT_PARQUET, SPX_CSV
from spx0dte.core import NY, load_spx_ohlc
from spx0dte.straddle import _prep, pick_atm
from spx0dte.exits import FEE, EXERCISE
from spx0dte.holdout import DIR

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
LO, HI = 0.0018, 0.0053          # 2023-24 밤사이 MNQ 7~8분위 경계 (46 이후 고정)
END = pd.Timestamp("2022-12-31")


def raw(path, symbol):
    p = pd.read_parquet(path); p = p[p.symbol == symbol].sort_index(); p.index = p.index.tz_convert(NY)
    return p[p.index <= pd.Timestamp("2023-01-01", tz=NY)]


def daily(p, days):
    """거래일마다: 밤사이(전일 15:59→09:29), 장중(09:45→15:59) 수익률. 같은 월물 안에서만."""
    out = {}
    ix = p.index
    for i in range(1, len(days)):
        d, pd_ = days[i], days[i - 1]
        a = ix.searchsorted(pd.Timestamp(f"{pd_:%Y-%m-%d} 15:59", tz=NY)); b = ix.searchsorted(pd.Timestamp(f"{d:%Y-%m-%d} 15:59", tz=NY), side="right")
        w = p.iloc[a:b]
        if len(w) < 100 or w.instrument_id.nunique() > 1:
            continue
        c = w.close
        g = lambda hm: c.asof(pd.Timestamp(f"{d:%Y-%m-%d} {hm}", tz=NY))
        out[d] = {"on": g("09:29") / c.iloc[0] - 1, "day": g("15:59") / g("09:45") - 1}
    return pd.DataFrame(out).T


def opt_day(args):
    f, close = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    try:
        M, bid, ask, cp, K, F = _prep(db.DBNStore.from_file(str(f)).to_df(), d)
        ts = pd.Timestamp(f"{d:%Y-%m-%d} 09:45", tz=NY)
        best = pick_atm(M, ask, cp, K, float(F[ts]), ts)
        if best is None:
            return None
        k, c, p = best
        return {"date": d, "c_ask": ask.at[ts, c], "p_ask": ask.at[ts, p], "c_pay": max(close - k, 0.0), "p_pay": max(k - close, 0.0)}
    except Exception:
        return None


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    px = load_spx_ohlc(SPX_CSV)
    days = px.index[(px.index >= "2020-01-01") & (px.index <= END)]
    es = daily(raw(DATA / "es_1m.parquet", "ES.v.0"), days)
    mnq = daily(raw(FUT_PARQUET, "MNQ.v.0"), days)
    X = es.rename(columns={"day": "es_day", "on": "es_on"}).join(mnq.rename(columns={"on": "mnq_on", "day": "mnq_day"}), how="inner")
    X["year"] = X.index.year; X["dow"] = X.index.dayofweek

    # ① 요일별 ES 장중 (09:45→15:59), 연도별 평균 bp
    dow = X.pivot_table(index="dow", columns="year", values="es_day", aggfunc="mean") * 1e4
    dow.index = list("월화수목금"); dow["2020-22 평균"] = X.groupby("dow").es_day.mean().values * 1e4
    dow["하락 비율"] = X.groupby("dow").es_day.apply(lambda s: (s < 0).mean()).values
    print("① ES 09:45→15:59 평균 (bp), 요일별\n", dow.round(1).to_string())

    # ② 밤사이 MNQ 구간별 ES 장중
    X["밤사이"] = pd.cut(X.mnq_on, [-1, -0.002, LO, HI, 0.0082, 1], labels=["하락 큼", "보합", "적당히 상승", "상승", "크게 상승"])
    on = X.pivot_table(index="밤사이", columns="year", values="es_day", aggfunc="mean", observed=False) * 1e4
    on["건수"] = X.groupby("밤사이", observed=False).size()
    print("\n② 밤사이 MNQ 구간별 ES 09:45→15:59 평균 (bp)\n", on.round(1).to_string())

    # ② 옵션: 2020-01 ~ 2022-05 월·수·금
    files = sorted((DATA / "0dte_oos").glob("*.dbn.zst"))
    jobs = [(f, float(px.close[pd.Timestamp(f.name[:10])])) for f in files if pd.Timestamp(f.name[:10]) in px.index]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        O = pd.DataFrame([r for r in ex.map(opt_day, jobs, chunksize=4) if r]).set_index("date")
    O = O.join(X[["mnq_on", "밤사이"]], how="inner")
    rows = []
    for g, t in O.groupby("밤사이", observed=True):
        for leg in ("c", "p"):
            pay, a = t[f"{leg}_pay"], t[f"{leg}_ask"]
            pnl = pay - a - FEE - EXERCISE * (pay > 0)
            rows.append({"밤사이": g, "구조": "콜" if leg == "c" else "풋", "건수": len(t), "수익률%": pnl.sum() / a.sum() * 100,
                         "2020%": pnl[t.index.year == 2020].sum() / a[t.index.year == 2020].sum() * 100,
                         "2021%": pnl[t.index.year == 2021].sum() / a[t.index.year == 2021].sum() * 100,
                         "2022%": pnl[t.index.year == 2022].sum() / a[t.index.year == 2022].sum() * 100,
                         "$총": pnl.sum() * 100})
    OP = pd.DataFrame(rows)
    print("\n② 옵션 2020-01~2022-05 월·수·금 09:45 ATM (매도호가)\n", OP.round(1).to_string(index=False))
    pd.concat({"요일": dow.reset_index(), "밤사이": on.reset_index(), "옵션": OP}, names=["표"]).to_csv(DIR / "H2_followup.csv", encoding="utf-8-sig")

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, T, ttl in ((axs[0], dow.drop(columns=["2020-22 평균", "하락 비율"]), "① 요일별 ES 장중 (09:45→15:59, bp)"),
                       (axs[1], on.drop(columns=["건수"]), "② 밤사이 MNQ 구간별 ES 장중 (bp)")):
        T.plot.bar(ax=ax, rot=0, width=0.8); ax.axhline(0, color="k", lw=0.6); ax.set_title(ttl); ax.set_xlabel("")
    piv = OP.pivot(index="밤사이", columns="구조", values="수익률%").reindex(on.index)
    piv.plot.bar(ax=axs[2], rot=0, color=["#B23A48", "#3D5A80"]); axs[2].axhline(0, color="k", lw=0.6)
    axs[2].set_title("② 옵션 2020~22.5 월·수·금 09:45 ATM 수익률 %"); axs[2].set_xlabel("")
    fig.tight_layout(); fig.savefig(DIR / "H2_followup.png", dpi=110)
