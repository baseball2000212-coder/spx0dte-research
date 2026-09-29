"""
V3 금고 열람 (한 번만): output/holdout/사전등록_V3_스트래들.md 규칙 그대로 2025-01-02 ~ 2026-09-23 평가.
결과: output/holdout/V3_holdout_days.csv, V3_holdout.png, V3_holdout_result.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import OPT_DIR, SPX_CSV
from spx0dte.core import NY, YEAR_MIN, load_day, load_spx_ohlc, implied_vol
from spx0dte.straddle import _prep, pick_atm
from spx0dte.exits import FEE, EXERCISE
from spx0dte.holdout import DIR, unlock
from spx0dte import vix

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
TU = np.sqrt(390 * 252 / 525600) * 100
E3 = 0.6401


def day10(args):
    d, close = args
    warnings.filterwarnings("ignore")
    try:
        M, bid, ask, cp, K, F = _prep(load_day(d), d)
        ts = pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=NY); fv = float(F[ts])
        best = pick_atm(M, ask, cp, K, fv, ts)
        if best is None:
            return {"date": d, "오류": "ATM 없음"}
        k, c, p = best; T = 360 / YEAR_MIN
        iv = [float(implied_vol(np.array([M.at[ts, x]]), np.array([fv]), np.array([k]), np.array([T]), np.array([call]))[0]) for x, call in ((c, True), (p, False))]
        return {"date": d, "K": k, "F": fv, "close": close, "c_ask": ask.at[ts, c], "p_ask": ask.at[ts, p], "c_mid": M.at[ts, c], "p_mid": M.at[ts, p],
                "pay": abs(close - k), "iv_atm": np.nanmean(iv)}
    except Exception as ex:
        return {"date": d, "오류": str(ex)[:60]}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    px = load_spx_ohlc(SPX_CSV)
    files = pd.DatetimeIndex(sorted(pd.Timestamp(f.name[:10]) for f in OPT_DIR.glob("*.dbn.zst")))
    days = unlock("V3_스트래들", files)
    days = days[days.isin(px.index)]
    print(f"검증 구간 {len(days)}일: {days.min():%Y-%m-%d} ~ {days.max():%Y-%m-%d}", flush=True)
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        R = pd.DataFrame(list(ex.map(day10, [(d, float(px.close[d])) for d in days], chunksize=4))).set_index("date")
    err = R["오류"].notna().sum() if "오류" in R else 0
    R = R[R["오류"].isna()] if "오류" in R else R
    V = vix.load(); prev = V.shift(1)
    R["v9"] = prev.VIX9D.reindex(R.index); R["ratio"] = R.iv_atm * TU / R.v9; R["sig"] = R.ratio <= E3
    R["cost"] = R.c_ask + R.p_ask; R["cost_mid"] = R.c_mid + R.p_mid
    x = 2 * FEE + EXERCISE * (R.pay > 0)
    R["pnl"] = R.pay - R.cost - x; R["pnl_mid"] = R.pay - R.cost_mid - x
    R.to_csv(DIR / "V3_holdout_days.csv", encoding="utf-8-sig")
    ret = lambda g, c="pnl", k="cost": g[c].sum() / g[k].sum() * 100 if len(g) else np.nan
    s, n = R[R.sig], R[~R.sig]
    s3 = s.drop(s.pnl.nlargest(3).index)
    c1, c2, c3 = s.pnl.sum() > 0, ret(s) > ret(n), ret(s3) > ret(n)
    lines = [f"검증 {len(R)}일 (오류 {err}), 신호일 {len(s)} ({len(s) / len(R):.0%})",
             f"신호일: 총 ${s.pnl.sum() * 100:,.0f}, 수익률 {ret(s):+.1f}% (중간가 {ret(s, 'pnl_mid', 'cost_mid'):+.1f}%), 승률 {(s.pnl > 0).mean():.0%}",
             f"나머지: 총 ${n.pnl.sum() * 100:,.0f}, 수익률 {ret(n):+.1f}% (중간가 {ret(n, 'pnl_mid', 'cost_mid'):+.1f}%)",
             f"신호일 최고 3일 제외: {ret(s3):+.1f}%  (빠진 날: {', '.join(f'{d:%Y-%m-%d}' for d in s.pnl.nlargest(3).index)})",
             f"기준1 신호일 총손익 > 0: {'통과' if c1 else '불통과'}", f"기준2 신호일 > 나머지: {'통과' if c2 else '불통과'}",
             f"기준3 최고3일 빼도 신호일 > 나머지: {'통과' if c3 else '불통과'}", f"최종: {'통과' if c1 and c2 and c3 else '불통과'}", ""]
    for y in (2025, 2026):
        sy, ny = s[s.index.year == y], n[n.index.year == y]
        lines.append(f"{y}: 신호 {len(sy)}일 {ret(sy):+.1f}% (${sy.pnl.sum() * 100:,.0f}) / 나머지 {len(ny)}일 {ret(ny):+.1f}% (${ny.pnl.sum() * 100:,.0f})")
    for y, h in ((2025, 1), (2025, 2), (2026, 1), (2026, 2)):
        m = (R.index.year == y) & ((R.index.month <= 6) == (h == 1))
        if m.any():
            lines.append(f"{y} {'상' if h == 1 else '하'}반기: 신호 {ret(R[m & R.sig]):+.1f}% ({(m & R.sig).sum()}일) / 나머지 {ret(R[m & ~R.sig]):+.1f}% ({(m & ~R.sig).sum()}일)")
    d49 = pd.Timestamp("2025-04-09")
    if d49 in R.index:
        r = R.loc[d49]; lines.append(f"2025-04-09 관세 유예일: 비율 {r.ratio:.3f}, 신호 {'예' if r.sig else '아니오'}, 스트래들 손익 ${r.pnl * 100:,.0f}")
    lines.append(f"신호일 제외 4/9: {ret(s.drop(d49, errors='ignore')):+.1f}%, 나머지 제외 4/9: {ret(n.drop(d49, errors='ignore')):+.1f}%")
    txt = "\n".join(lines); print(txt); (DIR / "V3_holdout_result.txt").write_text(txt, encoding="utf-8")

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    for g, nm, col in ((s, "신호일 (싼 날)", "#B23A48"), (n, "나머지", "#9AA3AD")):
        axs[0].plot(g.index, (g.pnl * 100).cumsum(), color=col, label=f"{nm} {len(g)}일")
    axs[0].axhline(0, color="k", lw=0.6); axs[0].set_title("누적 손익 $ (10:00 ATM 스트래들 1세트, 매도호가)"); axs[0].legend()
    q = pd.qcut(R.ratio, 5, labels=False) + 1
    axs[1].bar(range(1, 6), [ret(R[q == i]) for i in range(1, 6)], color="#3D5A80"); axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_title("비율 5분위별 수익률 % (1 = 가장 쌈, 참고)"); axs[1].set_xlabel("분위")
    fig.suptitle("V3 금고 검증 2025-01 ~ 2026-09 (규칙 고정, 한 번 열람)"); fig.tight_layout(); fig.savefig(DIR / "V3_holdout.png", dpi=110)
