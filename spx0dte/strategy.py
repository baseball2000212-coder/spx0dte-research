"""
기준 전략 (2026-09-26 확정 후보): 10:00에 '개장 30분 상승 + MNQ 5분봉 구름 위'면 SPX 0DTE 콜 매수 → 만기.
  물타기 옵션: 첫 매수가(중간가) 대비 -40%, -70%에 1개씩 추가 (진입 후 270분 = 14:30까지만).
"""
import numpy as np
import pandas as pd

from .exits import FEE, EXERCISE

ADD_LEVELS, ADD_LAST = (0.6, 0.3), 270


def features_10(FP, G, px):
    """10:00 시점 판단 재료: 개장 30분 수익률, 밤사이 갭, MNQ 60분 추세, MNQ 5분 구름 위치(1 위 / -1 아래 / 0 안), 5분 전환>기준."""
    f_open = FP[list(FP.columns[:5])].bfill(axis=1).iloc[:, 0]
    p, top, bot = G["px"]["10:00"], G["top5"]["10:00"], G["bot5"]["10:00"]
    X = pd.DataFrame({"f_open": f_open, "f10": FP["10:00"], "r30": FP["10:00"] / f_open - 1,
                      "gap": f_open / px.close.shift(1).reindex(FP.index) - 1,
                      "tr60": G["tr60"]["10:00"], "tk5": G["tk5"]["10:00"],
                      "c5": np.where(p > top, 1.0, np.where(p < bot, -1.0, 0.0)),
                      "mnq": p, "mnq_top5": top, "mnq_bot5": bot})
    X["signal"] = (X.r30 > 0) & (X.c5 == 1)
    return X


def trade_ladder(r, add_at=0.5, stop_at=0.375, fill="중간가", add_last=ADD_LAST):
    """
    사용자 규칙: 1개 매수 → 첫 매수가의 add_at(50%)까지 빠지면 1개 추가 (평단 75%)
                → 추가 후 stop_at(첫 매수가의 37.5% = 평단의 반)까지 빠지면 전부 손절, 아니면 만기.
    stop_at=None이면 손절 없음. 추가는 진입 후 add_last분(14:30)까지만. 손절 가격: 중간가 모드 = 중간가, 매도호가 모드 = 매수호가.
    """
    p0 = r.mid if fill == "중간가" else r.ask
    exitp = r.mid_path if fill == "중간가" else r.bid_path
    buys, stop = [(-1, float(p0))], None
    seg = r.mid_path[:add_last]
    hit = np.isfinite(seg) & (seg <= p0 * add_at)
    if hit.any():
        i = int(hit.argmax())
        p = r.mid_path[i] if fill == "중간가" else r.ask_path[i]
        if np.isfinite(p):
            buys.append((i, float(p)))
            if stop_at is not None:
                after = exitp[i + 1:]
                h2 = np.isfinite(after) & (after <= p0 * stop_at)
                if h2.any():
                    j = i + 1 + int(h2.argmax()); stop = (j, float(exitp[j]))
    cost = sum(p for _, p in buys)
    if stop:
        pnl = sum(stop[1] - p - 2 * FEE for _, p in buys)
    else:
        pnl = sum(r.pay - p - FEE - (EXERCISE if r.pay > 0 else 0) for _, p in buys)
    return {"buys": buys, "n": len(buys), "cost": cost, "avg": cost / len(buys), "pay": r.pay, "stop": stop,
            "pnl": pnl, "ret": pnl / cost * 100}


def trade_detail(r, averaging=True, fill="중간가", add_last=ADD_LAST):
    """한 거래 상세: 매수 목록 [(진입 후 분 인덱스 -1 = 10:00, 가격)], 정산액, 손익."""
    p0 = r.mid if fill == "중간가" else r.ask
    buys, start = [(-1, float(p0))], 0
    if averaging:
        for lv in ADD_LEVELS:
            seg = r.mid_path[start:add_last]
            hit = np.isfinite(seg) & (seg <= p0 * lv)
            if not hit.any():
                break
            i = start + int(hit.argmax())
            p = r.mid_path[i] if fill == "중간가" else r.ask_path[i]
            if not np.isfinite(p):
                break
            buys.append((i, float(p))); start = i + 1
    per = lambda p: r.pay - p - FEE - (EXERCISE if r.pay > 0 else 0)
    pnl = sum(per(p) for _, p in buys)
    cost = sum(p for _, p in buys)
    return {"buys": buys, "n": len(buys), "cost": cost, "avg": cost / len(buys), "pay": r.pay,
            "pnl": pnl, "ret": pnl / cost * 100, "pnl_single": per(p0)}


STOP = 0.9          # 2026-09-27 사용자 확정: 매수가 대비 −90%에서 손절


def trade_rule(r, stop=STOP, fill="중간가", lag=0):
    """
    현재 규칙 한 거래 (ATM 콜 1계약, 물타기 없음, −90% 손절, 아니면 만기).
    r: legs10_full 행 (mid·ask = 10:00:00 호가, mid_path·bid_path·ask_path[i] = 10:0(i+1):00 호가).
    lag: 체결 지연(분). 0 = 10:00:00 호가로 체결, 1 = 10:01:00 호가로 체결 (실제 지연은 이 둘 사이).
    손절: 매분 중간가가 매수가 × (1 − stop) 이하 → 그 분 매수호가(bid)로 매도. stop=None이면 만기 보유.
    """
    if lag == 0:
        p0 = r.mid if fill == "중간가" else r.ask
    else:
        p0 = r.mid_path[lag - 1] if fill == "중간가" else r.ask_path[lag - 1]
    p0 = float(p0)
    stop_at = None
    if stop is not None and np.isfinite(p0):
        mp = np.asarray(r.mid_path[lag:], float)
        hit = np.flatnonzero(np.isfinite(mp) & (mp <= p0 * (1 - stop) + 1e-6))          # 경계값(정확히 −90%)도 손절
        if len(hit):
            j = lag + int(hit[0]); stop_at = (j, float(r.bid_path[j]))
    if stop_at:
        pnl = stop_at[1] - p0 - 2 * FEE
    else:
        pnl = r.pay - p0 - FEE - (EXERCISE if r.pay > 0 else 0)
    return {"buys": [(lag - 1, p0)], "n": 1, "cost": p0, "avg": p0, "pay": r.pay, "stop": stop_at,
            "pnl": pnl, "ret": pnl / p0 * 100}


def oos_call_path(args):
    """표본외(data/0dte_oos) 하루: 행사가 k 콜의 10:00:00 중간가·매도호가, 이후 매분 중간가·매수호가·매도호가 경로, 정산액."""
    import databento as db
    from .core import NY
    from .straddle import _prep
    f, k, close = args
    d = pd.Timestamp(f.name[:10])
    M, bid, ask, cp, K, F = _prep(db.DBNStore.from_file(str(f)).to_df(), d)
    col = M.columns[(K == k) & (cp == "C")][0]
    t10 = pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=NY); after = M.index > t10
    intr = np.maximum(F.ffill()[after].values - k, 0)
    b = bid.loc[after, col].values; m = M.loc[after, col].values; a = ask.loc[after, col].values
    return {"date": d, "mid": M.at[t10, col], "ask": ask.at[t10, col], "pay": max(close - k, 0.0),
            "mid_path": np.where(np.isfinite(m), m, intr), "bid_path": np.where(np.isfinite(b), b, intr), "ask_path": a}


def load_oos_paths():
    """2020-01~2022-05 표본외 신호일(현재 규칙 신호)의 ATM 콜 경로 목록 (output/rule_v2/oos_paths.pkl 캐시)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from types import SimpleNamespace
    from .config import OUT, DATA
    cache = OUT / "rule_v2" / "oos_paths.pkl"
    if cache.exists():
        return [SimpleNamespace(**x) for x in pd.read_pickle(cache)]
    O = pd.read_csv(OUT / "oos" / "days.csv", index_col=0, parse_dates=True)
    O = O[(O["신호"] == True) & O["행사가"].notna()]
    jobs = [(DATA / "0dte_oos" / f"{d:%Y-%m-%d}.dbn.zst", float(r["행사가"]), float(r["SPX 종가"])) for d, r in O.iterrows()]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = sorted(ex.map(oos_call_path, jobs, chunksize=4), key=lambda x: x["date"])
    cache.parent.mkdir(exist_ok=True); pd.to_pickle(rows, cache)
    return [SimpleNamespace(**x) for x in rows]
