"""
10시 구름 콜 — 실시간 판단 엔진 (증권사와 무관한 부분).
  신호: ① SPX(선도가격) 10:00 > 개장(09:31)  ② MNQ 5분봉(09:55 봉) 종가 > 일목 구름 윗선 (9·26·52, 26칸 선행)
  주문 (2026-09-28 사용자 결정): ATM 콜 1계약, 10:00:00 매도호가 지정가 → 미체결이면 1초마다 현재 매도호가로 정정 (상한·시간 제한)
  청산: 중간가가 매수가의 10% 이하(−90%)면 매수호가로 매도 (마감 직전 15:50 이후 판정 안 함), 아니면 만기 보유
증권사 연결은 broker.py의 Broker를 구현 (NH선물 REST API 문서 받은 뒤 작성, 엔드포인트 추측 금지).
"""
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

NY = "America/New_York"


# ---------- 일목균형표 (백테스트와 같은 식) ----------
def ichimoku(b, t=9, k=26, s=52):
    mid = lambda w: (b.high.rolling(w).max() + b.low.rolling(w).min()) / 2
    tenkan, kijun = mid(t), mid(k)
    span1 = ((tenkan + kijun) / 2).shift(k)
    span2 = mid(s).shift(k)
    return pd.DataFrame({"top": pd.concat([span1, span2], axis=1).max(axis=1, skipna=False),
                         "bot": pd.concat([span1, span2], axis=1).min(axis=1, skipna=False)}, index=b.index)


def to_5min(bars_1m):
    """1분봉(open/high/low/close, 인덱스 = 봉 시작 시각, 뉴욕 시간) → 5분봉 (정시 기준, 거래 있는 봉만)."""
    return bars_1m.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


# ---------- 신호 ----------
@dataclass
class Decision:
    day: str
    spx_open: float
    spx_10: float
    spx_up: bool
    mnq_close_0955: float
    cloud_top: float
    mnq_above_cloud: bool
    buy: bool
    reason: str = ""


def decide(day, spx_open, spx_10, mnq_1m):
    """
    day: 'YYYY-MM-DD'. spx_open: 09:31 선도가격(또는 SPX 시가), spx_10: 10:00 선도가격.
    mnq_1m: MNQ 1분봉 DataFrame (최소 전날 저녁부터 오늘 09:59 봉까지, 뉴욕 시간 인덱스) — 5분봉 78개 이상 필요.
    """
    t0955 = pd.Timestamp(f"{day} 09:55", tz=NY)
    b5 = to_5min(mnq_1m[mnq_1m.index < pd.Timestamp(f"{day} 10:00", tz=NY)])
    if t0955 not in b5.index or len(b5) < 80:
        return Decision(day, spx_open, spx_10, False, np.nan, np.nan, False, False, "MNQ 5분봉 부족 (09:55 봉 없음 또는 과거 78봉 미만)")
    ic = ichimoku(b5)
    c, top = float(b5.close[t0955]), float(ic.top[t0955])
    up, above = bool(spx_10 > spx_open), bool(np.isfinite(top) and c > top)
    why = ("매수" if up and above else
           "쉼: " + ", ".join(x for x, ok in (("SPX 개장 대비 하락", not up), ("MNQ 5분봉 구름 위 아님", not above)) if ok))
    return Decision(day, spx_open, spx_10, up, c, top, above, up and above, why)


# ---------- 선도가격 (풋콜패리티, 백테스트와 같은 방식) ----------
def forward_from_chain(chain, n_near=5):
    """chain: DataFrame [strike, call_bid, call_ask, put_bid, put_ask]. |C−P|가 가장 작은 행사가 n개의 K + C − P 중앙값."""
    c = (chain.call_bid + chain.call_ask) / 2
    p = (chain.put_bid + chain.put_ask) / 2
    ok = (chain.call_bid > 0) & (chain.put_bid > 0) & np.isfinite(c) & np.isfinite(p)
    g = pd.DataFrame({"K": chain.strike, "d": (c - p).abs(), "syn": chain.strike + c - p})[ok]
    return float(g.nsmallest(n_near, "d").syn.median()) if len(g) else np.nan


def pick_atm(chain, forward, max_dist=10.0):
    """선도가격에 가장 가까운 행사가 중 콜·풋 둘 다 호가 있는 것. 10pt 넘게 멀면 None."""
    g = chain[(chain.call_ask > 0) & (chain.put_ask > 0)].copy()
    g["dist"] = (g.strike - forward).abs()
    g = g[g.dist <= max_dist].sort_values("dist")
    return None if g.empty else g.iloc[0]


# ---------- 주문: 매도호가 추격 ----------
def tick_up(p):
    """SPX 옵션 호가 단위로 올림: 3.00 미만 0.05, 이상 0.10."""
    t = 0.05 if p < 3 else 0.10
    return round(np.ceil(p / t - 1e-9) * t, 2)


@dataclass
class ChasePlan:
    """
    10:00:00 매도호가로 지정가 → 안 되면 매초 현재 매도호가로 정정.
    예전 LimitPlan(중간가 → 20초마다 25%씩, 10:00 매도호가까지, 2분 뒤 포기)은 급등일을 놓침:
    틱 재현 2023-03~ 319일 중 44일 못 삼, 2025-04-09 5센트 차이로 놓쳐 −$37k (output/ticks/engine_ladder_sim.csv).
    cap: 10:00:00 매도호가 대비 최대 몇 %까지 따라갈지 (안전장치), max_wait_sec: 이 시간 지나면 포기.
    """
    cap: float = 0.20
    max_wait_sec: int = 60
    reprice_sec: int = 1

    def price(self, ask_now, ask0, elapsed_sec):
        """지금 낼 지정가. None이면 포기 (시간 초과 or 매도호가가 상한 위)."""
        if elapsed_sec > self.max_wait_sec or not np.isfinite(ask_now) or ask_now <= 0:
            return None
        if ask_now > ask0 * (1 + self.cap) + 1e-9:
            return None if elapsed_sec >= self.max_wait_sec else np.nan     # nan = 이번 초는 주문 안 냄(기다림)
        return tick_up(ask_now)


# ---------- 손절 판정 ----------
@dataclass
class StopRule:
    """중간가 ≤ 매수가 × (1 − stop)이면 매도 (매수호가 지정가). no_check_after 이후는 판정 안 함 (마감 직전 호가 붕괴 방지)."""
    stop: float = 0.9
    no_check_after: str = "15:50"

    def hit(self, mid_now, cost, now_hhmm):
        if now_hhmm >= self.no_check_after or not np.isfinite(mid_now):
            return False
        return mid_now <= cost * (1 - self.stop) + 1e-6
