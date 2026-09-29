"""스트래들 청산 규칙: 매분 bid 합 경로로 손절·익절 판정, 안 걸리면 만기 정산."""
import numpy as np
import pandas as pd

FEE = 0.05         # 계약당 편도 $5 = 0.05pt (NH SPXW 수수료, 사용자 확인 2026-09-29. 그 전 결과는 $2.5로 계산됨)
EXERCISE = 0.05    # 만기 정산 비용 (내가격 레그 1개, 미확인 → 편도와 같게 가정)


def apply_rule(cost, path, payoff, sl=None, tp=None, fee=FEE, exercise=EXERCISE):
    """(손익 pt, 사유 '손절'/'익절'/'만기', 청산 인덱스 or -1)"""
    hit = np.zeros(len(path), bool)
    if sl is not None:
        hit |= path <= cost * (1 - sl)
    if tp is not None:
        hit |= path >= cost * tp
    hit &= np.isfinite(path)
    if hit.any():
        i = int(hit.argmax())
        return path[i] - cost - 4 * fee, ("손절" if path[i] < cost else "익절"), i
    return payoff - cost - 2 * fee - (exercise if payoff > 0 else 0), "만기", -1


def minutes(hhmm):
    h, m = map(int, str(hhmm).split(":"))
    return h * 60 + m


def run_rule(P, sl=None, tp=None, exit_at=None, **kw):
    """
    straddle_paths.pkl 테이블 → 거래별 손익 (pnl pt, why, exit_time).
    exit_at="12:00"이면 그 시각까지만 보고 그 시각 bid로 청산 (시간 청산). 진입보다 이르면 제외.
    """
    rows = []
    for r in P.itertuples():
        path, n = r.bid_path, None
        if exit_at is not None:
            n = minutes(exit_at) - minutes(r.entry)             # bid_path[n-1] = exit_at 분
            if n <= 0 or n > len(path):
                continue
            path = path[:n]
        pnl, why, i = apply_rule(r.cost_ask, path, r.payoff, sl, tp, **kw)
        if why == "만기" and n is not None:                    # 시간 청산
            v = path[-1]
            if not np.isfinite(v):
                continue
            pnl, why, i = v - r.cost_ask - 4 * kw.get("fee", FEE), "시간청산", n - 1
        m = minutes(r.entry) + i + 1 if i >= 0 else 16 * 60
        rows.append((r.date, r.entry, r.K, r.cost_ask, r.payoff, pnl, why, f"{m // 60:02d}:{m % 60:02d}"))
    return pd.DataFrame(rows, columns=["date", "entry", "K", "cost_ask", "payoff", "pnl", "why", "exit_time"])
