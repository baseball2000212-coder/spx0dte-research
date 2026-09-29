"""
새 아이디어 탐색용 한 바퀴 (찾는 구간만: 2022-05-16 ~ 2024-12-31, holdout.discovery로 자름 — 2025~26 파일은 열지도 않음).
진입 시각마다 ATM 콜·풋, ±0.5% 외가격 콜·풋의 매도호가·중간가·정산액 + IV(스큐용).
+ 일별 피처: 전날 SPX 수익률, 밤사이 MNQ·금 수익률, 요일, 월말·월초, 월물 만기일, 휴일 전날.
결과: output/holdout/disc_chain.pkl, disc_days.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from spx0dte.config import OPT_DIR, SPX_CSV, FUT_PARQUET
from spx0dte.core import NY, YEAR_MIN, load_day, load_spx_ohlc, implied_vol
from spx0dte.straddle import _prep, pick_atm
from spx0dte.holdout import discovery, DIR

ENTRIES = ("09:35", "09:45", "10:00", "10:30", "11:00", "12:00", "13:00", "14:00")
OTM = (0.005, 0.01)


def near(ks, x):
    return float(ks[np.abs(ks - x).argmin()]) if len(ks) else np.nan


def one_day(args):
    d, close = args
    warnings.filterwarnings("ignore")
    try:
        M, bid, ask, cp, K, F = _prep(load_day(d), d)
    except Exception as ex:
        return [], None
    col = {(x, k): c for c, x, k in zip(M.columns, cp, K)}
    rows = []
    for e in ENTRIES:
        ts = pd.Timestamp(f"{d:%Y-%m-%d} {e}", tz=NY)
        if ts not in M.index or not np.isfinite(F.get(ts, np.nan)):
            continue
        f = float(F[ts]); best = pick_atm(M, ask, cp, K, f, ts)
        if best is None:
            continue
        k0, c0, p0 = best
        T = (pd.Timestamp(f"{d:%Y-%m-%d} 16:00", tz=NY) - ts).total_seconds() / 60 / YEAR_MIN
        r = {"date": d, "entry": e, "F": f, "K": k0, "close": close,
             "c_ask": ask.at[ts, c0], "c_mid": M.at[ts, c0], "p_ask": ask.at[ts, p0], "p_mid": M.at[ts, p0],
             "c_pay": max(close - k0, 0.0), "p_pay": max(k0 - close, 0.0)}
        iv = lambda c, k, call: float(implied_vol(np.array([M.at[ts, c]]), np.array([f]), np.array([k]), np.array([T]), np.array([call]))[0])
        r["iv_atm"] = np.nanmean([iv(c0, k0, True), iv(p0, k0, False)])
        for m in OTM:
            for leg, sgn in (("C", 1), ("P", -1)):
                ks = np.array(sorted(k for (x, k) in col if x == leg and np.isfinite(M.at[ts, col[(x, k)]])
                                     and np.isfinite(ask.at[ts, col[(x, k)]])))
                k = near(ks, f * (1 + sgn * m))
                tag = f"{leg.lower()}{int(m * 1000)}"
                if not np.isfinite(k) or abs(k - f * (1 + sgn * m)) > 5:
                    continue
                c = col[(leg, k)]
                r.update({f"{tag}_K": k, f"{tag}_ask": ask.at[ts, c], f"{tag}_mid": M.at[ts, c],
                          f"{tag}_pay": max(close - k, 0.0) if leg == "C" else max(k - close, 0.0),
                          f"{tag}_iv": iv(c, k, leg == "C")})
        rows.append(r)
    fp = pd.Series(F.ffill().values.astype("float32"), index=F.index.strftime("%H:%M"), name=d)
    return rows, fp


def fut_daily(symbol, days):
    """밤사이 수익률: 전 거래일 15:59 종가 → 당일 09:29 종가, 밤사이 범위 (원가격, 월물 교체일은 NaN)."""
    p = pd.read_parquet(FUT_PARQUET); p = p[p.symbol == symbol].sort_index(); p.index = p.index.tz_convert(NY)
    out = {}
    for i, d in enumerate(days[1:], 1):
        a = pd.Timestamp(f"{days[i - 1]:%Y-%m-%d} 15:59", tz=NY); b = pd.Timestamp(f"{d:%Y-%m-%d} 09:29", tz=NY)
        w = p[(p.index >= a) & (p.index <= b)]
        if len(w) < 30 or w.instrument_id.nunique() > 1:
            continue
        out[d] = {"ret": w.close.iloc[-1] / w.close.iloc[0] - 1, "rng": (w.high.max() - w.low.min()) / w.close.iloc[0]}
    return pd.DataFrame(out).T


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    px = load_spx_ohlc(SPX_CSV)
    files = pd.DatetimeIndex(sorted(pd.Timestamp(f.name[:10]) for f in OPT_DIR.glob("*.dbn.zst")))
    days = discovery(files, disc_start=pd.Timestamp("2022-05-16"))
    days = days[days.isin(px.index)]
    assert days.max() < pd.Timestamp("2025-01-01")
    print(f"찾는 구간 {len(days)}일: {days.min():%Y-%m-%d} ~ {days.max():%Y-%m-%d}", flush=True)
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        res = list(ex.map(one_day, [(d, float(px.close[d])) for d in days], chunksize=4))
    ch = pd.DataFrame([r for rows, _ in res for r in rows])
    fp = pd.DataFrame([s for _, s in res if s is not None])
    pd.to_pickle({"chain": ch, "fpath": fp}, DIR / "disc_chain.pkl")

    # ── 일별 피처 (진입 전에 알 수 있는 것만) ──
    allpx = px[px.index <= pd.Timestamp("2024-12-31")]
    D = pd.DataFrame(index=days)
    D["prev_ret"] = allpx.close.pct_change().shift(1).reindex(days)
    D["gap"] = (fp["09:31"] / allpx.close.shift(1).reindex(fp.index) - 1).reindex(days)
    mnq = fut_daily("MNQ.v.0", allpx.index[allpx.index >= "2022-05-01"]); mgc = fut_daily("MGC.v.0", allpx.index[allpx.index >= "2022-05-01"])
    D["mnq_on"] = mnq.ret.reindex(days); D["mnq_on_rng"] = mnq.rng.reindex(days); D["mgc_on"] = mgc.ret.reindex(days)
    D["dow"] = days.dayofweek
    td = allpx.index; pos = pd.Series(np.arange(len(td)), index=td)
    mon = pd.Series(td.to_period("M"), index=td)
    first = pos.groupby(mon).transform("min"); last = pos.groupby(mon).transform("max")
    D["tom"] = (((pos - first) < 3) | (pos == last)).reindex(days).values
    D["opex"] = (days.dayofweek == 4) & (days.day >= 15) & (days.day <= 21)
    nxt = pd.Series(td[1:].append(pd.DatetimeIndex([td[-1] + pd.offsets.BDay(1)])), index=td)
    gapdays = np.busday_count(td.values.astype("datetime64[D]"), nxt.values.astype("datetime64[D]"))
    D["pre_hol"] = pd.Series(gapdays > 1, index=td).reindex(days).values        # 다음 거래일까지 평일 휴장 끼면 True
    D.index.name = "date"
    D.to_csv(DIR / "disc_days.csv", encoding="utf-8-sig")
    print(f"옵션 행 {len(ch)}, 일별 {len(D)}, 저장 → {DIR}")
    print(ch.groupby("entry").size().to_string())
    print(D.isna().sum().to_string())
