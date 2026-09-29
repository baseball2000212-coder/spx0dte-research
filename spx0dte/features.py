"""하루치 파일 → 장중 ATM IV 곡선, 일별 변동성 피처, 델타헤지 스트래들 (notebook_cells/B에서 옮김)."""
import numpy as np
import pandas as pd

from .core import NY, YEAR_MIN, implied_vol, greeks, mid_matrix, parse_osi, forward_parity

TR = np.sqrt(390 * 252 / YEAR_MIN)       # 캘린더시간 IV → 거래시간 IV (RV·VIX와 같은 단위)
ANN = np.sqrt(390 * 252) * 100           # 1분 수익률 표준편차 → 연율 %
ENTRIES = ["10:00", "12:00", "14:00", "15:00", "15:30", "15:45", "15:55"]
PATH_ENTRIES = ["09:35", "09:45", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30",
                "14:00", "14:30", "15:00", "15:30", "15:45", "15:55"]   # 08 경로 저장용 (매수·매도 시각 조합)
DH_ENTRIES = ["10:00", "14:00"]          # 델타헤지 스트래들 진입 시각 (15:59 중간가 청산)


def leg_matrices(M, cp, K):
    Ks = np.unique(K); pos = np.searchsorted(Ks, K)
    C = np.full((len(M), len(Ks)), np.nan); P = C.copy()
    for j in range(M.shape[1]):
        (C if cp[j] == "C" else P)[:, pos[j]] = M.values[:, j]
    return Ks, C, P


def atm_iv(M, cp, K, F, tau):
    """매분 F에 가장 가까운 행사가의 콜·풋 IV 평균 (거래시간 연율 %)"""
    Ks, C, P = leg_matrices(M, cp, K)
    Fv = F.values; n = len(Fv); rows = np.arange(n)
    i = np.abs(Ks[None, :] - np.nan_to_num(Fv, nan=Ks[0])[:, None]).argmin(1)
    ivc = implied_vol(C[rows, i], Fv, Ks[i], tau, np.ones(n, bool))
    ivp = implied_vol(P[rows, i], Fv, Ks[i], tau, np.zeros(n, bool))
    with np.errstate(all="ignore"):
        return pd.Series(np.nanmean([ivc, ivp], axis=0) * TR * 100, index=M.index)


def delta_hedge(M, cp, K, F, tau, d, e):
    """e에 ATM 스트래들 매수 + 매분 델타만큼 선물 반대 포지션 → 15:59 중간가 청산"""
    ts = pd.Timestamp(f"{d:%Y-%m-%d} {e}", tz=NY)
    if ts not in M.index or not np.isfinite(F[ts]):
        return None
    leg = None
    for k in sorted(set(K), key=lambda k: abs(k - F[ts])):
        c = M.columns[(K == k) & (cp == "C")]; p = M.columns[(K == k) & (cp == "P")]
        if len(c) and len(p) and np.isfinite(M.at[ts, c[0]]) and np.isfinite(M.at[ts, p[0]]):
            leg = (k, c[0], p[0]); break
    if leg is None:
        return None
    k, c, p = leg
    w = M.index >= ts; n = int(w.sum())
    Fw, tw = F.values[w], tau[w]
    Cm, Pm = M.loc[w, c].values, M.loc[w, p].values
    ivc = implied_vol(Cm, Fw, np.full(n, k), tw, np.ones(n, bool))
    ivp = implied_vol(Pm, Fw, np.full(n, k), tw, np.zeros(n, bool))
    dc = greeks(Fw, k, tw, ivc, True)[0]; dp = greeks(Fw, k, tw, ivp, False)[0]
    dc = np.where(np.isfinite(dc), dc, (Fw > k).astype(float))      # IV 못 구하면 내가격 1 / 외가격 0
    dp = np.where(np.isfinite(dp), dp, -(Fw < k).astype(float))
    hedge = -np.nansum((dc + dp)[:-1] * np.diff(Fw))                  # 헤지 포지션 손익
    end = (Cm[-1] if np.isfinite(Cm[-1]) else max(Fw[-1] - k, 0)) + \
          (Pm[-1] if np.isfinite(Pm[-1]) else max(k - Fw[-1], 0))
    v0 = Cm[0] + Pm[0]
    with np.errstate(all="ignore"):
        return dict(date=d, entry=e, K=k, cost=v0, straddle_pnl=end - v0, hedge_pnl=hedge,
                    dh_pnl=end - v0 + hedge, iv0=np.nanmean([ivc[0], ivp[0]]) * TR * 100,
                    rv_win=np.nanstd(np.diff(np.log(Fw))) * ANN)


def features_day(df, d, ref_close):
    """하루치 원본 → (일별 피처 dict, IV 곡선 Series, 델타헤지 결과 list)"""
    M = mid_matrix(df, d); cp, K = parse_osi(M.columns)
    F = pd.Series(forward_parity(M, cp, K), index=M.index).ffill()
    tau = ((pd.Timestamp(f"{d:%Y-%m-%d} 16:00", tz=NY) - M.index).total_seconds() / 60 / YEAR_MIN).values
    iv = atm_iv(M, cp, K, F, tau)
    r = np.log(F).diff()
    row = dict(date=d,
               rv_day=r.between_time("09:35", "15:59").std() * ANN,     # 당일 실현변동성
               rv_am=r.between_time("09:35", "10:00").std() * ANN,      # 개장 30분 실현변동성
               gap=(F.dropna().iloc[0] / ref_close - 1) * 100)          # 전일 종가 대비 시가 갭 %
    for e in ENTRIES:
        row[f"iv_{e}"] = iv.get(pd.Timestamp(f"{d:%Y-%m-%d} {e}", tz=NY), np.nan)
        row[f"F_{e}"] = F.get(pd.Timestamp(f"{d:%Y-%m-%d} {e}", tz=NY), np.nan)
    dh = [x for e in DH_ENTRIES if (x := delta_hedge(M, cp, K, F, tau, d, e))]
    return row, pd.Series(iv.values, index=iv.index.strftime("%H:%M")), dh
