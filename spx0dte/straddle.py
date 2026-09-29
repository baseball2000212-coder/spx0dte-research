"""0DTE ATM 스트래들 양매수: 장중 특정 시각 진입 → 만기(16:00 SPX 종가 정산)까지 보유 손익 + 청산가 경로."""
import numpy as np
import pandas as pd

from .core import NY, parse_osi, forward_parity, mid_matrix

MAX_ATM_DIST = 10.0      # 선도가격에서 이 이상 먼 행사가밖에 없으면 (±3% 행사가 범위 밖으로 급변) 진입 안 함


def quote_matrices(df, d, max_spread=0.5):
    """
    분 × 종목 매수·매도호가 행렬 (09:31~15:59). 호가가 한쪽 비었거나(시장조성자 호가 철회),
    역전됐거나, 스프레드가 너무 넓은 분은 bid·ask 둘 다 버리고 직전 정상 호가를 최대 5분 이어씀.
    """
    grid = pd.date_range(f"{d:%Y-%m-%d} 09:31", f"{d:%Y-%m-%d} 15:59", freq="1min", tz=NY)
    b, a = df["bid_px_00"].astype(float), df["ask_px_00"].astype(float)
    good = (a > 0) & (b >= 0) & (a >= b) & ((a - b) <= np.maximum(max_spread * (a + b) / 2, 0.20))
    q = pd.DataFrame({"t": df.index.tz_convert(NY).floor("min"), "symbol": df["symbol"].values,
                      "bid": b.values, "ask": a.values})[good.values]
    return tuple(q.pivot_table(index="t", columns="symbol", values=v, aggfunc="last").reindex(grid).ffill(limit=5)
                 for v in ("bid", "ask"))


def _prep(df, d):
    M = mid_matrix(df, d)
    bid, ask = quote_matrices(df, d)
    cols = M.columns.intersection(bid.columns)
    M, bid, ask = M[cols], bid[cols], ask[cols]
    cp, K = parse_osi(cols)
    F = pd.Series(forward_parity(M, cp, K), index=M.index)
    return M, bid, ask, cp, K, F


def pick_atm(M, ask, cp, K, f, ts):
    """f에 가장 가까운 행사가 중 콜·풋 둘 다 호가 있는 것. 10pt 넘게 멀면 None."""
    for k in sorted(set(K), key=lambda k: abs(k - f)):
        if abs(k - f) > MAX_ATM_DIST:
            return None
        c = M.columns[(K == k) & (cp == "C")]; p = M.columns[(K == k) & (cp == "P")]
        if len(c) and len(p) and np.isfinite(M.at[ts, c[0]]) and np.isfinite(M.at[ts, p[0]]) \
           and np.isfinite(ask.at[ts, c[0]]) and np.isfinite(ask.at[ts, p[0]]):
            return k, c[0], p[0]
    return None


def straddle_paths_day(df, d, close, entries):
    """
    진입 시각마다 ATM 스트래들을 ask로 샀다고 치고, 이후 매분 '지금 팔면 받는 값'(콜 bid + 풋 bid) 경로 저장.
    한쪽 레그 호가가 비면 그 레그는 내재가치(선도가격 기준)로 봄.
    """
    M, bid, ask, cp, K, F = _prep(df, d)
    Ff = F.ffill()
    rows = []
    for e in entries:
        ts = pd.Timestamp(f"{d:%Y-%m-%d} {e}", tz=NY)
        if ts not in M.index or not np.isfinite(F.get(ts, np.nan)):
            continue
        f = F[ts]
        best = pick_atm(M, ask, cp, K, f, ts)
        if best is None:
            continue
        k, c, p = best
        after = bid.index > ts
        fw = Ff[after].values
        bc = np.where(np.isfinite(bid.loc[after, c].values), bid.loc[after, c].values, np.maximum(fw - k, 0))
        bp = np.where(np.isfinite(bid.loc[after, p].values), bid.loc[after, p].values, np.maximum(k - fw, 0))
        cost_mid = M.at[ts, c] + M.at[ts, p]
        rows.append({"date": d.normalize(), "entry": e, "F": f, "K": k, "close": close,
                     "cost_mid": cost_mid, "cost_ask": ask.at[ts, c] + ask.at[ts, p], "payoff": abs(close - k),
                     "bid_path": (bc + bp).astype("float32")})          # [i] = 진입 i+1분 뒤
    return rows


def leg_paths_day(df, d, close, entries):
    """
    방향성 단일 레그용: 진입 시각마다 ATM 콜·풋 각각의 ask(매수가)와 이후 매분 bid(청산가) 경로.
    + 그날 매분 선도가격 F (09:31~15:59, SPX 1분 지수 대용) → 추세 신호·추세 이탈 청산에 사용.
    """
    M, bid, ask, cp, K, F = _prep(df, d)
    Ff = F.ffill()
    rows = []
    for e in entries:
        ts = pd.Timestamp(f"{d:%Y-%m-%d} {e}", tz=NY)
        if ts not in M.index or not np.isfinite(F.get(ts, np.nan)):
            continue
        best = pick_atm(M, ask, cp, K, F[ts], ts)
        if best is None:
            continue
        k, c, p = best
        after = bid.index > ts
        fw = Ff[after].values
        bc = np.where(np.isfinite(bid.loc[after, c].values), bid.loc[after, c].values, np.maximum(fw - k, 0))
        bp = np.where(np.isfinite(bid.loc[after, p].values), bid.loc[after, p].values, np.maximum(k - fw, 0))
        rows.append({"date": d.normalize(), "entry": e, "K": k, "F0": F[ts], "close": close,
                     "call_ask": ask.at[ts, c], "put_ask": ask.at[ts, p],
                     "call_mid": M.at[ts, c], "put_mid": M.at[ts, p],
                     "call_pay": max(close - k, 0.0), "put_pay": max(k - close, 0.0),
                     "call_bid": bc.astype("float32"), "put_bid": bp.astype("float32")})   # [i] = 진입 i+1분 뒤
    fpath = pd.Series(Ff.values.astype("float32"), index=Ff.index.strftime("%H:%M"), name=d.normalize())
    return rows, fpath


OFFSETS = (0, 5, 10, 15, 20, 30)   # ATM에서 외가격 쪽으로 몇 pt (콜은 위, 풋은 아래)


def leg_strikes_day(df, d, close, entries, offsets=OFFSETS, full=False):
    """진입 시각마다 ATM 기준 외가격 콜·풋 여러 개: ask, mid, 만기 정산액, 이후 매분 bid 경로 (full이면 mid·ask 경로도)."""
    M, bid, ask, cp, K, F = _prep(df, d)
    Ff = F.ffill()
    rows = []
    for e in entries:
        ts = pd.Timestamp(f"{d:%Y-%m-%d} {e}", tz=NY)
        if ts not in M.index or not np.isfinite(F.get(ts, np.nan)):
            continue
        best = pick_atm(M, ask, cp, K, F[ts], ts)
        if best is None:
            continue
        k0 = best[0]
        after = bid.index > ts
        fw = Ff[after].values
        for leg, sgn in (("C", 1), ("P", -1)):
            for o in offsets:
                k = k0 + sgn * o
                col = M.columns[(K == k) & (cp == leg)]
                if not len(col) or not np.isfinite(ask.at[ts, col[0]]) or ask.at[ts, col[0]] <= 0:
                    continue
                c = col[0]
                intr = np.maximum(fw - k, 0) if leg == "C" else np.maximum(k - fw, 0)
                b = bid.loc[after, c].values
                row = {"date": d.normalize(), "entry": e, "leg": leg, "offset": o, "K": k, "F0": F[ts],
                       "ask": ask.at[ts, c], "mid": M.at[ts, c],
                       "pay": max(close - k, 0.0) if leg == "C" else max(k - close, 0.0),
                       "bid_path": np.where(np.isfinite(b), b, intr).astype("float32")}
                if full:
                    mp, ap = M.loc[after, c].values, ask.loc[after, c].values
                    row["mid_path"] = np.where(np.isfinite(mp), mp, intr).astype("float32")
                    row["ask_path"] = np.where(np.isfinite(ap), ap, np.nan).astype("float32")
                rows.append(row)
    return rows


def call_moneyness_day(df, d, close, entries, moneyness=(0, 0.001, 0.002, 0.003, 0.005)):
    """진입 시각마다 선도가격 × (1 + m)에 가장 가까운 행사가 콜 (m=0은 ATM): ask·mid, 이후 매분 bid·mid·ask 경로, 정산액."""
    M, bid, ask, cp, K, F = _prep(df, d)
    Ff = F.ffill()
    calls = {k: c for k, c, x in zip(K, M.columns, cp) if x == "C"}
    ks = np.array(sorted(calls))
    rows = []
    for e in entries:
        ts = pd.Timestamp(f"{d:%Y-%m-%d} {e}", tz=NY)
        if ts not in M.index or not np.isfinite(F.get(ts, np.nan)):
            continue
        best = pick_atm(M, ask, cp, K, F[ts], ts)
        if best is None:
            continue
        after = bid.index > ts
        fw = Ff[after].values
        for m in moneyness:
            k = best[0] if m == 0 else float(ks[np.abs(ks - F[ts] * (1 + m)).argmin()])
            if m > 0 and k <= best[0]:
                k = float(ks[ks > best[0]].min()) if (ks > best[0]).any() else np.nan
            if not np.isfinite(k):
                continue
            c = calls[k]
            if not np.isfinite(ask.at[ts, c]) or ask.at[ts, c] <= 0 or not np.isfinite(M.at[ts, c]):
                continue
            intr = np.maximum(fw - k, 0)
            b, mp, ap = bid.loc[after, c].values, M.loc[after, c].values, ask.loc[after, c].values
            rows.append({"date": d.normalize(), "entry": e, "m": m, "K": k, "F0": F[ts], "ask": ask.at[ts, c], "mid": M.at[ts, c],
                         "pay": max(close - k, 0.0),
                         "bid_path": np.where(np.isfinite(b), b, intr).astype("float32"),
                         "mid_path": np.where(np.isfinite(mp), mp, intr).astype("float32"),
                         "ask_path": np.where(np.isfinite(ap), ap, np.nan).astype("float32")})
    return rows


def straddle_day(df, d, close, entries=("15:00", "15:30", "15:45", "15:55")):
    """진입 시각마다 ATM 콜+풋 1개씩 매수 → 만기 정산 손익 (mid / ask 두 가지 체결 가정)."""
    t = pd.DataFrame(straddle_paths_day(df, d, close, entries))
    return trades_from_paths(t) if len(t) else t


def trades_from_paths(P):
    t = P.drop(columns=["bid_path"]).copy()
    t["pnl_mid"] = t.payoff - t.cost_mid
    t["pnl_ask"] = t.payoff - t.cost_ask
    t["implied_move"] = t.cost_mid
    t["realized_move"] = (t.close - t.F).abs()
    return t


def summarize_straddle(t):
    """진입 시각별 요약: 평균 손익, 승률, 프리미엄 대비 수익률, 내재 vs 실현 움직임."""
    g = t.groupby("entry")
    return pd.DataFrame({
        "days": g.size(),
        "avg_cost_mid": g.cost_mid.mean(),
        "avg_payoff": g.payoff.mean(),
        "avg_pnl_mid": g.pnl_mid.mean(),
        "avg_pnl_ask": g.pnl_ask.mean(),
        "win_rate_mid": g.pnl_mid.apply(lambda x: (x > 0).mean()),
        "ret_mid_%": g.apply(lambda x: x.pnl_mid.sum() / x.cost_mid.sum() * 100, include_groups=False),
        "ret_ask_%": g.apply(lambda x: x.pnl_ask.sum() / x.cost_ask.sum() * 100, include_groups=False),
        "median_pnl_mid": g.pnl_mid.median(),
        "worst_pnl_mid": g.pnl_mid.min(),
        "best_pnl_mid": g.pnl_mid.max(),
        "implied/realized": g.implied_move.mean() / g.realized_move.mean(),
    }).round(3)
