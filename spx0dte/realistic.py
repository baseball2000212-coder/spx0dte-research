"""
실제 체결 기준 손익 (사용자 지시 2026-09-29: 기본값 = 매도호가 매수 + 체결 지연 5초·10초, −90% 손절, 수수료 $5).
  - 틱 있는 날(2023-03-28~): output/ticks/N_days.csv의 5·10초 뒤 실측 매도호가
  - 그 전(2022-05~2023-03)·틱 없는 날·앞으로 기록: 지연 없이 10:00:00 매도호가 (사용자 지시 2026-09-29, 평균 비율 추정 안 씀)
  - 손절: 매분 중간가 ≤ 매수가 × 0.1 → 그 분 매수호가 매도 (경로[0] = 10:01:00), 아니면 16:00 정산
CME 시세료(월 $228.80)는 net_cum()에서 월 단위로 뺌.
"""
import numpy as np
import pandas as pd

from .config import OUT, SPX_CSV
from .exits import FEE, EXERCISE
from .strategy import STOP

DELAYS = (5, 10)
CME = 228.80


def load_legs(signal_only=True):
    """10:00 ATM 콜 경로 (legs10_full) + 신호. signal_only=False면 모든 날 (매일 매수 기준선용)."""
    from .core import load_spx_ohlc
    from .touch import load_mnq, signal_grids_ext
    from .strategy import features_10
    FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
    X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
    L = pd.read_pickle(OUT / "legs10_full.pkl")
    L = L[(L.leg == "C") & (L.offset == 0) & (L.entry == "10:00")].set_index("date").join(X, how="inner").sort_index()
    return L[L.signal] if signal_only else L


def load_ticks():
    return pd.read_csv(OUT / "ticks" / "N_days.csv", index_col=0, parse_dates=True)


def ratios(TK):
    return {k: float((TK[f"ask_{k}"] / TK["ask_0"]).mean()) for k in DELAYS}


def rule_at(r, p0, stop=STOP):
    """진입가 p0 → (손익$, 손절 여부)."""
    if stop is not None:
        mp = np.asarray(r.mid_path, float)
        hit = np.flatnonzero(np.isfinite(mp) & (mp <= p0 * (1 - stop) + 1e-6))
        if len(hit):
            return (float(r.bid_path[hit[0]]) - p0 - 2 * FEE) * 100, True
    return (r.pay - p0 - FEE - (EXERCISE if r.pay > 0 else 0)) * 100, False


def entry(d, r, sec, TK, R):
    if sec is None:
        return float(r.mid)
    if sec == 0:
        return float(r.ask)
    if d in TK.index:
        return float(TK.at[d, f"ask_{sec}"])
    return float(r.ask)


def trades(L, sec, TK=None, R=None, stop=STOP):
    """sec = None(10:00 중간가) / 0(10:00:00 매도호가) / 5 / 10 → 거래 표 [K, 진입가, 정산, 손절, 손익$, 틱실측]."""
    TK = load_ticks() if TK is None else TK; R = ratios(TK) if R is None else R
    rows = []
    for d, r in zip(L.index, L.itertuples()):
        p0 = entry(d, r, sec, TK, R); pnl, st = rule_at(r, p0, stop)
        rows.append({"date": d, "K": r.K, "진입가": p0, "정산": r.pay, "손절": st, "손익$": pnl, "틱실측": d in TK.index})
    return pd.DataFrame(rows).set_index("date")


def forward(sec, R):
    """앞으로 기록 매수일 (10:00:00 매도호가, 손절은 기록값 없으면 만기로)."""
    L = pd.read_csv(OUT / "forward_log.csv", parse_dates=["date"]).set_index("date")
    L = L[(L["신호"] == "매수") & L["정산"].notna()]
    pay, ask = L["정산"].astype(float), L["콜 매도호가"].astype(float)
    return pd.DataFrame({"K": L["행사가"], "진입가": ask, "정산": pay, "손절": False,
                         "손익$": (pay - ask - FEE - np.where(pay > 0, EXERCISE, 0)) * 100, "틱실측": False, "앞으로 기록": True})


def net_cum(pnl, first=None):
    """누적 손익 − 경과 개월 × CME 시세료 (거래일마다 그 달까지 개월 수만큼)."""
    first = pnl.index.min() if first is None else first
    mon = np.array([(t.year - first.year) * 12 + t.month - first.month + 1 for t in pnl.index])
    return pnl.cumsum() - mon * CME


def max_dd(c):
    dd = c - c.cummax(); i = dd.idxmin()
    return dd.min(), c[:i].idxmax(), i


def boot_dd(pnl, months, n=5000, seed=0):
    """거래 순서 재표본 최대낙폭 [중앙값, 최악 5%, 최악 1%] (CME는 거래당 평균으로 나눠 뺌)."""
    rng = np.random.default_rng(seed); v = pnl.values - CME * months / len(pnl)
    out = np.empty(n)
    for k in range(n):
        c = np.cumsum(rng.choice(v, len(v), replace=True)); out[k] = (c - np.maximum.accumulate(np.maximum(c, 0))).min()
    return np.percentile(out, [50, 5, 1])
