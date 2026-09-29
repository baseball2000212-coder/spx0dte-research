"""
일목 구름 터치 매매 시뮬레이터 — 하루 여러 번, 한 번에 1포지션, 손절 후 다음 터치에서 재진입.
  신호: SPX(F) n분봉이 구름 밖 → 구름에 닿거나 뚫는 봉이 완성되는 순간 (1 = 위에서, 2 = 아래에서)
  매수: 그 분의 ATM 콜 or 풋 ask / 청산: 그 분의 bid / 안 걸리면 16:00 정산
"""
import numpy as np
import pandas as pd

from .core import NY
from .straddle import _prep, pick_atm
from .ichimoku import ichimoku, OPEN_MIN
from .exits import FEE, EXERCISE


def _stack(FP):
    fp = FP.copy()
    half = fp.loc[:, "13:05":"15:59"].std(axis=1) < 1e-6              # 조기폐장일 13:00 이후 제거
    fp.loc[half, "13:01":"15:59"] = np.nan
    s = fp.stack().rename("px").reset_index()
    s.columns = ["date", "hhmm", "px"]
    s["mins"] = s.hhmm.str[:2].astype(int) * 60 + s.hhmm.str[3:].astype(int)
    return s


def _grid(FP, df, col, fill):
    """봉 단위 값(완성 분에 기록) → 날짜 × 분 격자. fill='ffill'이면 다음 봉 완성 전까지 이어씀 (날짜 넘어서도)."""
    g = df.pivot_table(index="date", columns="hhmm", values=col, aggfunc="last").reindex(index=FP.index, columns=FP.columns)
    if fill == "ffill":
        v = pd.Series(g.values.ravel()).ffill().values.reshape(g.shape)
        return pd.DataFrame(v, index=g.index, columns=g.columns)
    return g.fillna(fill)


def _events(b, strict=False):
    """
    봉(open/high/low/close) → 일목 + 터치 이벤트 (1 = 위에서 닿음, 2 = 아래에서), 상태, 구름색, 전환>기준.
    strict=True: 구름을 통째로 관통한 봉(종가가 반대편 밖)은 터치 아님 — 위에서 왔으면 종가 ≥ 구름 아래,
                 아래에서 왔으면 종가 ≤ 구름 위일 때만.
    """
    ic = ichimoku(b)
    st = np.where(b.low > ic["구름위"], 1.0, np.where(b.high < ic["구름아래"], -1.0, 0.0))
    st[ic["구름위"].isna().values] = np.nan                            # 1 구름 위, -1 아래, 0 닿음
    prev = pd.Series(st).shift(1).values
    from_above, from_below = (prev == 1) & (st <= 0), (prev == -1) & (st >= 0)
    if strict:
        from_above &= (b.close >= ic["구름아래"]).values
        from_below &= (b.close <= ic["구름위"]).values
    ev = np.where(from_above, 1, np.where(from_below, 2, 0))
    return pd.DataFrame({"ev": ev, "state": st, "px": b.close.values,
                         "col": (ic["선행스팬2"] > ic["선행스팬1"]).astype(int).values,       # 1 = 음운
                         "tk": np.where(ic["전환선"] > ic["기준선"], 1, -1),
                         "top": ic["구름위"].values, "bot": ic["구름아래"].values}, index=b.index)


def signal_grids(FP):
    """SPX(F) 정규장 봉 기준: 1분·5분 터치 이벤트/구름색/경계/전환>기준, 60분 추세 → {이름: 날짜 × 분 DataFrame}"""
    s = _stack(FP)
    out = {"F": FP.T.ffill().T}
    out["px"] = out["F"]
    for n in (1, 5, 60):
        t = s.copy()
        t["bin"] = (t.mins - OPEN_MIN) // n
        g = t.dropna(subset=["px"]).groupby(["date", "bin"])
        b = pd.DataFrame({"open": g.px.first(), "high": g.px.max(), "low": g.px.min(), "close": g.px.last(),
                          "last": g.hhmm.last()}).reset_index()
        bars_ = _events(b).assign(date=b.date.values, hhmm=b["last"].values)
        if n == 60:
            out["tr60"] = _grid(FP, bars_, "state", "ffill")
            continue
        out[f"ev{n}"] = _grid(FP, bars_, "ev", 0).astype(int)
        for k in ("col", "top", "bot", "tk"):
            out[f"{k}{n}"] = _grid(FP, bars_, k, "ffill")
    return out


# ---------- MNQ (사용자가 실제로 보는 차트) ----------
def load_mnq(path=None):
    """MNQ 연결선물 1분봉 (24시간, 뉴욕 시간). 월물 교체 지점 가격 차이만큼 과거를 역조정 (가짜 구름 터치 방지)."""
    from .config import FUT_PARQUET
    return load_fut("MNQ.v.0", path or FUT_PARQUET)


def load_es():
    """ES(S&P500 선물) 연결선물 1분봉, 역조정. data/es_1m.parquet (scripts/01b_download_es.py)."""
    from .config import DATA
    return load_fut("ES.v.0", DATA / "es_1m.parquet")


def load_fut(symbol, path):
    p = pd.read_parquet(path)
    p = p[p.symbol == symbol].sort_index()
    roll = np.flatnonzero(p.instrument_id.values[1:] != p.instrument_id.values[:-1]) + 1
    adj = np.zeros(len(p))
    for k in roll:
        adj[:k] += p.open.values[k] - p.close.values[k - 1]
    out = p[["open", "high", "low", "close"]].add(adj, axis=0)
    out.index = out.index.tz_convert(NY)
    return out, len(roll)


def mnq_bars(m, n):
    if n == 1:
        return m
    return m.resample(f"{n}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def signal_grids_ext(m, FP, tfs=(1, 5), strict=False):
    """
    외부 차트(MNQ 24시간) 기준 신호 → SPX 격자(날짜 × 09:31~15:59). 봉이 끝난 '다음 분' SPX 호가로 행동
    (예: MNQ 10:30 1분봉 → SPX 10:31 호가, 5분봉 10:25~10:29 → SPX 10:30 호가).
    """
    mins = np.array([int(c[:2]) * 60 + int(c[3:]) for c in FP.columns], dtype="timedelta64[m]")
    days = pd.DatetimeIndex(FP.index).tz_localize(NY).tz_convert("UTC").tz_localize(None).values
    grid = pd.DatetimeIndex((days[:, None] + mins[None, :]).ravel()).tz_localize("UTC").tz_convert(NY)
    shape = FP.shape
    out = {"F": FP.T.ffill().T}

    def put(name, s, fill):
        s = s[~s.index.duplicated(keep="last")].sort_index()
        v = s.reindex(grid).fillna(0).values if fill == 0 else s.reindex(grid, method="ffill").values
        out[name] = pd.DataFrame(v.reshape(shape), index=FP.index, columns=FP.columns)

    for n in tuple(tfs) + (60,):
        b = mnq_bars(m, n)
        e = _events(b, strict=strict)
        e.index = b.index + pd.Timedelta(minutes=n)                    # 봉 끝난 시각 = 행동 가능 시각
        if n == 60:
            put("tr60", e.state, "ffill"); continue
        put(f"ev{n}", e.ev, 0)
        for k in ("col", "top", "bot", "tk"):
            put(f"{k}{n}", e[k], "ffill")
        put(f"px{n}", e.px, "ffill")                                    # 그 봉 종가 (마지막 완성 봉)
        if n == 1:
            put("px", e.px, "ffill")                                    # 구름 이탈 판정은 MNQ 가격으로
    out["ev1"] = out["ev1"].astype(int)
    if "ev5" in out:
        out["ev5"] = out["ev5"].astype(int)
    return out


def decide(rule, side, tr, tk=0):
    """터치 방향(1 위에서, 2 아래에서) + 60분 추세 + 전환>기준 → 'C' / 'P' / None"""
    bounce = "C" if side == 1 else "P"
    brk = "P" if side == 1 else "C"
    trend = "C" if tr == 1 else ("P" if tr == -1 else None)
    return {"반등": bounce, "돌파": brk, "콜만": "C", "60분추세": trend,
            "반등+60분": bounce if bounce == trend else None,
            "돌파+60분": brk if brk == trend else None,
            "전환>기준": "C" if tk == 1 else "P"}[rule]


def simulate_day(df, d, close, S, configs, first="09:45", last="15:00", max_trades=6, sl=0.3, tstop=15):
    """S: 그날 신호 배열 dict (길이 389, 09:31~15:59). configs: [{'id','tf','rule','color','exit'}]"""
    M, bid, ask, cp, K, F = _prep(df, d)
    cols = list(M.index.strftime("%H:%M"))
    i0, i1 = cols.index(first), cols.index(last)
    Ff = S["F"]
    cache, rows = {}, []

    def leg_at(i, leg):
        if (i, leg) in cache:
            return cache[(i, leg)]
        ts, f = M.index[i], Ff[i]
        best = pick_atm(M, ask, cp, K, f, ts) if np.isfinite(f) else None
        res = None
        if best is not None:
            k, c, p = best
            col = c if leg == "C" else p
            cost = ask.at[ts, col]
            fw = Ff[i + 1:]
            b = bid[col].values[i + 1:]
            intr = np.maximum(fw - k, 0) if leg == "C" else np.maximum(k - fw, 0)
            pay = max(close - k, 0.0) if leg == "C" else max(k - close, 0.0)
            if np.isfinite(cost) and cost > 0:
                res = (k, cost, np.where(np.isfinite(b), b, intr), pay)
        cache[(i, leg)] = res
        return res

    px = S.get("px", Ff)                                                # 신호 차트 가격 (SPX면 F, MNQ면 MNQ)
    for cfg in configs:
        tf = cfg["tf"]
        ev, colr, top, bot, tr = S[f"ev{tf}"], S[f"col{tf}"], S[f"top{tf}"], S[f"bot{tf}"], S["tr60"]
        tk = S.get(f"tk{tf}", np.zeros(len(ev)))
        i, n = i0, 0
        while i <= i1 and n < max_trades:
            if ev[i] == 0 or (cfg["color"] == "음운만" and colr[i] != 1):
                i += 1; continue
            leg = decide(cfg["rule"], ev[i], tr[i], tk[i])
            got = leg_at(i, leg) if leg else None
            if got is None:
                i += 1; continue
            k, cost, path, pay = got
            hit = np.isfinite(path) & (path <= cost * (1 - sl))
            if cfg["exit"] == "구름이탈":
                fw, bw, tw = px[i + 1:], bot[i + 1:], top[i + 1:]
                with np.errstate(invalid="ignore"):
                    hit |= (fw < bw) if leg == "C" else (fw > tw)
            elif cfg["exit"] == "15분" and len(path) > tstop - 1:
                hit[tstop - 1] |= path[tstop - 1] <= cost
            if hit.any():
                j = int(hit.argmax())
                pnl, why, xi = path[j] - cost - 2 * FEE, ("손절" if path[j] <= cost * (1 - sl) else cfg["exit"]), i + 1 + j
            else:
                pnl, why, xi = pay - cost - FEE - (EXERCISE if pay > 0 else 0), "만기", -1
            rows.append((cfg["id"], d.normalize(), cols[i], cols[xi] if xi >= 0 else "16:00", leg, k, cost, pnl, why,
                         int(ev[i]), int(colr[i]) if np.isfinite(colr[i]) else -1, tr[i]))
            n += 1
            if xi < 0:
                break                                                   # 만기까지 보유 → 그날 끝
            i = xi + 1                                                  # 청산 다음 분부터 다시 터치 대기
    return rows
