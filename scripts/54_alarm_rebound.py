"""
사용자 아이디어 (2026-09-27): '알람 후 반등 매도'.
  콜 중간가가 매수가 대비 −A%(알람)까지 떨어지면 → 그 뒤
    · 중간가가 −X%까지 반등하면 그 분 매수호가로 매도 (반등 탈출)
    · 반대로 −90%까지 가면 그 분 매수호가로 손절
    · 둘 다 아니면 만기
  알람 안 울린 날은 기존 규칙 그대로 (−90% 손절, 아니면 만기).
시험: 알람 −60/−70/−80% × 반등 탈출 −X% (알람보다 10%p 위 ~ 본전), 기준 = −90%만. 2022-05~2026-09, 연도별.
결과: output/stops/alarm_grid.csv, S2_alarm.png (연도별 히트맵)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
from types import SimpleNamespace
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_rule
from spx0dte.exits import FEE, EXERCISE

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
R = OUT / "stops"; ODD = pd.Timestamp("2025-04-09"); EPS = 1e-6


def trade_alarm(r, alarm, rebound, stop=0.9):
    """(손익 pt, 결과 '반등매도'/'손절'/'만기(알람 후)'/'기존', 매도가 or nan)"""
    p0 = float(r.mid); m = np.asarray(r.mid_path, float); b = np.asarray(r.bid_path, float)
    a = np.flatnonzero(np.isfinite(m) & (m <= p0 * (1 - alarm) + EPS))
    if not len(a):
        t = trade_rule(r, stop=stop)
        return t["pnl"], "기존", np.nan
    ja = a[0]
    if m[ja] <= p0 * (1 - stop) + EPS:                                  # 알람 분에 이미 −90%
        return b[ja] - p0 - 2 * FEE, "손절", b[ja]
    after = m[ja + 1:]
    up = np.flatnonzero(np.isfinite(after) & (after >= p0 * (1 - rebound) - EPS))
    dn = np.flatnonzero(np.isfinite(after) & (after <= p0 * (1 - stop) + EPS))
    iu = up[0] if len(up) else 10**9; idn = dn[0] if len(dn) else 10**9
    if iu == idn == 10**9:
        return r.pay - p0 - FEE - (EXERCISE if r.pay > 0 else 0), "만기(알람 후)", np.nan
    j = ja + 1 + min(iu, idn)
    return b[j] - p0 - 2 * FEE, ("반등매도" if iu < idn else "손절"), b[j]


def run(rows, alarm, rebound):
    v = [trade_alarm(r, alarm, rebound) for r in rows]
    idx = pd.DatetimeIndex([r.date for r in rows])
    p = pd.Series([x[0] * 100 for x in v], index=idx); kind = pd.Series([x[1] for x in v], index=idx)
    pay = pd.Series([r.pay for r in rows], index=idx); cost = pd.Series([r.mid for r in rows], index=idx)
    reb = kind == "반등매도"
    hold = (pay - cost) * 100                                            # 만기까지 뒀으면 (수수료 전)
    eq = p.cumsum()
    return p, {"총손익$": p.sum(), "4/9 빼고$": p.sum() - p.get(ODD, 0), "최대낙폭$": (eq - eq.cummax()).min(),
               "알람": int((kind != "기존").sum()), "반등매도": int(reb.sum()), "알람 후 손절": int((kind == "손절").sum()),
               "알람 후 만기": int((kind == "만기(알람 후)").sum()),
               "반등매도 중 만기면 더 좋았음": int((reb & (hold > p)).sum()), "반등매도가 잘라낸 이익$": (hold - p)[reb & (hold > p)].sum(),
               **{str(y): v for y, v in p.groupby(p.index.year).sum().items()}}


if __name__ == "__main__":
    FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
    X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
    L = pd.read_pickle(OUT / "legs10_full.pkl")
    L = L[(L.leg == "C") & (L.offset == 0) & (L.entry == "10:00")].set_index("date").join(X[["signal"]], how="inner")
    L = L[L.signal].sort_index()
    rows = [SimpleNamespace(date=d, mid=r.mid, ask=r.ask, pay=r.pay, mid_path=r.mid_path, bid_path=r.bid_path, ask_path=r.ask_path) for d, r in L.iterrows()]

    # 질문 1: −90% 손절한 날 중 손절가보다 높게 정산된 날
    st = [(r, trade_rule(r)) for r in rows]
    s = [(r, t) for r, t in st if t["stop"]]
    above = [(r, t) for r, t in s if r.pay > t["stop"][1]]
    print(f"−90% 손절 {len(s)}건 중 정산액 > 손절 매도가: {len(above)}건, 만기 정산 0: {sum(r.pay <= 0 for r, _ in s)}건")
    for r, t in above:
        print(f"  {r.date:%Y-%m-%d} 매수 {r.mid:.2f} 손절 {t['stop'][1]:.2f} 정산 {r.pay:.2f} → 손절로 {(r.pay - t['stop'][1]) * 100:+,.0f}$ 손해")

    grid = [(None, None)] + [(a, x) for a in (0.6, 0.7, 0.8) for x in np.round(np.arange(a - 0.1, -0.001, -0.1), 1)]
    out = []
    for a, x in grid:
        if a is None:
            p = pd.Series([trade_rule(r)["pnl"] * 100 for r in rows], index=[r.date for r in rows]); eq = p.cumsum()
            m = {"총손익$": p.sum(), "4/9 빼고$": p.sum() - p.get(ODD, 0), "최대낙폭$": (eq - eq.cummax()).min(),
                 **{str(y): v for y, v in p.groupby(p.index.year).sum().items()}}
            nm = "기준: -90%만"
        else:
            _, m = run(rows, a, x)
            nm = f"알람 -{int(a * 100)}% → 반등 {'본전' if x == 0 else f'-{int(x * 100)}%'} 매도"
        out.append({"규칙": nm, "알람": a, "반등": x, **m})
    G = pd.DataFrame(out)
    yrs = ["2022", "2023", "2024", "2025", "2026"]
    G["기준보다 나은 해"] = [np.nan if k == 0 else int((G.loc[k, yrs] > G.loc[0, yrs] + 1e-9).sum()) for k in G.index]
    G.to_csv(R / "alarm_grid.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    cols = ["규칙"] + yrs + ["총손익$", "4/9 빼고$", "최대낙폭$", "기준보다 나은 해", "알람", "반등매도", "알람 후 손절", "알람 후 만기", "반등매도 중 만기면 더 좋았음", "반등매도가 잘라낸 이익$"]
    print(G[[c for c in cols if c in G]].round(0).to_string(index=False))

    # 연도별 히트맵 (천 달러, 괄호 = 기준 대비)
    T = G.set_index("규칙")[yrs + ["총손익$"]] / 1000; D = T - T.iloc[0].values
    fig, ax = plt.subplots(figsize=(11, 9)); lim = np.nanmax(np.abs(D.values[1:]))
    ax.imshow(D.values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{T.iloc[i, j]:+.1f}" + ("" if i == 0 else f"\n({D.iloc[i, j]:+.1f})"), ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if i == 0 else None)
    ax.set_xticks(range(len(yrs) + 1), ["2022(5월~)", "2023", "2024", "2025", "2026(9/23까지)", "합계"])
    ax.set_yticks(range(len(T)), [n if k == 0 else f"{n}  [{int(G.loc[k, '기준보다 나은 해'])}/5년 나음]" for k, n in enumerate(T.index)], fontsize=8)
    ax.set_title("S2 알람 후 반등 매도 — 연도별 손익 (천 달러, 괄호 = 기준 대비 차이)\n기준 = -90% 손절만. 초록 = 기준보다 나음, 빨강 = 나쁨", fontsize=11)
    fig.tight_layout(); fig.savefig(R / "S2_alarm.png", dpi=110)
