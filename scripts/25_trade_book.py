"""
기준 전략 전체 거래 차트북: 수익률 높은 순서로 한 장에 2건 (SPX 1분 + 콜 가격), 진입가·물타기가·총 포지션·손익 표시.
  python scripts/25_trade_book.py [--offset 10] [--no-avg]
결과: output/trade_book_+10.pdf, output/trade_log_+10.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_detail, trade_rule
from spx0dte.events import label_events

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})   # $ 기호를 수식으로 읽지 않게
ap = argparse.ArgumentParser()
ap.add_argument("--offset", type=int, default=10)
ap.add_argument("--no-avg", action="store_true")
ap.add_argument("--from", dest="start", default=None, help="이 날짜부터만 (예: 2025-01-01 = 검증 구간)")
ap.add_argument("--tag", default=None, help="파일 이름 꼬리표 (예: 2026-09)")
ap.add_argument("--order", default="수익률", choices=["수익률", "날짜"])
ap.add_argument("--stop", type=float, default=None, help="손절 비율 (예: 0.9 = 매수가 대비 −90%%, 물타기 없음일 때만)")
a = ap.parse_args()
AVG = not a.no_avg
TAG = f"{'ATM' if a.offset == 0 else f'+{a.offset}'}{'' if AVG else '_단일'}{f'_손절{int(a.stop * 100)}' if a.stop else ''}{('_' + a.tag) if a.tag else ('_검증' if a.start else '')}"
SPECIAL = {"2025-04-09": "관세 유예 폭등(+9.5%)", "2025-04-03": "관세 발표 폭락", "2025-04-04": "관세 보복 폭락",
           "2025-04-07": "관세 변동성", "2025-04-08": "관세 변동성", "2024-08-05": "엔캐리 청산 폭락"}

FP = pd.read_pickle(OUT / "f_paths.pkl")
X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), load_spx_ohlc(SPX_CSV))
L = pd.read_pickle(OUT / "legs10_full.pkl")
L = L[(L.leg == "C") & (L.offset == a.offset)].set_index("date").join(X, how="inner")
L = L[L.signal]
if a.start:
    L = L[L.index >= pd.Timestamp(a.start)]
close = load_spx_ohlc(SPX_CSV).close
ev = dict(zip(L.index, label_events(L.index)))

rows = []
for d, r in L.iterrows():
    t = trade_detail(r, averaging=AVG)
    stop = None
    if a.stop and not AVG:
        t = {**trade_rule(r, stop=a.stop), "pnl_single": t["pnl"]}; stop = t["stop"]
    rows.append({"date": d, **{k: t[k] for k in ("n", "cost", "avg", "pay", "pnl", "ret", "pnl_single")}, "stop": stop,
                 "buys": t["buys"], "K": r.K, "r30": r.r30, "f_open": r.f_open, "f10": r.f10, "close": close[d],
                 "event": ev[d], "special": SPECIAL.get(f"{d:%Y-%m-%d}", ""), "mid_path": r.mid_path})
B = pd.DataFrame(rows).sort_values("ret" if a.order == "수익률" else "date", ascending=(a.order == "날짜")).reset_index(drop=True)
B["rank"] = B.index + 1
log = B.drop(columns=["mid_path"]).copy()
log["stop"] = [f"{10 + (s[0] + 1) // 60:02d}:{(s[0] + 1) % 60:02d} @{s[1]:.2f}" if s else "" for s in log.stop]
log["buys"] = [" / ".join(f"{'10:00' if i < 0 else f'{10 + (i + 1) // 60:02d}:{(i + 1) % 60:02d}'} @{p:.2f}" for i, p in b) for b in log.buys]
log.round(3).to_csv(OUT / f"trade_log_{TAG}.csv", index=False, encoding="utf-8-sig")

cols = list(FP.columns); s0 = cols.index("10:00")
hhmm = lambda i: cols[s0 + 1 + i] if i >= 0 else "10:00"
tk = [k for k, c in enumerate(cols) if c.endswith(":00")]
N = len(B)
with PdfPages(OUT / f"trade_book_{TAG}.pdf") as pdf:
    # 표지
    fig = plt.figure(figsize=(11.7, 8.3)); fig.text(0.05, 0.9, f"기준 전략 거래 {(a.tag + ' ') if a.tag else ('검증 구간 ' if a.start else '전체 ')}({N}건, {'수익률 높은 순' if a.order == '수익률' else '날짜 순'})", fontsize=20)
    tot = B.pnl.sum() * 100
    lines = [f"규칙: 10:00에 SPX가 개장가보다 높고 + MNQ 5분봉이 일목 구름 위면 → SPX 0DTE 콜 {TAG.replace('_단일', '').replace('_검증', '').split('_')[0]} 매수, 만기 보유",
             f"물타기: {'첫 매수가 대비 -40%·-70%에 1개씩 추가 (14:30까지)' if AVG else '없음'},  손절: {f'중간가가 매수가 대비 -{int(a.stop * 100)}% 이하 → 그 분 매수호가로 매도' if a.stop and not AVG else '없음'},  체결: 중간가,  수수료 편도 $2.5",
             f"기간: {B.date.min():%Y-%m-%d} ~ {B.date.max():%Y-%m-%d}",
             f"총손익 ${tot:,.0f}  ({'손절 없었으면' if a.stop else '물타기 없었으면'} ${B.pnl_single.sum() * 100:,.0f})" + (f",  손절 {B.stop.notna().sum()}건" if a.stop else ""),
             f"이긴 거래 {(B.pnl > 0).sum()}건 / 진 거래 {(B.pnl <= 0).sum()}건,  물타기 한 거래 {(B.n > 1).sum()}건",
             "각 장: 위 = SPX 1분 (▲ 첫 매수, ● 물타기, ▼ 손절 매도, 점선 = 행사가), 아래 = 그 콜의 중간가 흐름 (빨간 점선 = 손절선)",
             "시간 = 뉴욕 (한국 = +13시간, 겨울 +14시간)"]
    for j, s in enumerate(lines):
        fig.text(0.05, 0.8 - j * 0.06, s, fontsize=12)
    pdf.savefig(fig); plt.close(fig)
    for p0 in range(0, N, 2):
        fig = plt.figure(figsize=(11.7, 8.3))
        gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 2.2, 1], hspace=0.55)
        for k, r in enumerate(B.iloc[p0:p0 + 2].itertuples()):
            a1, a2 = fig.add_subplot(gs[2 * k]), fig.add_subplot(gs[2 * k + 1])
            f = FP.loc[r.date].values
            a1.plot(range(len(cols)), f, color="black", lw=1)
            a1.axhline(r.K, color="C0", ls=":", lw=1)
            a1.axvline(s0, color="grey", lw=.6)
            mp = np.r_[np.nan, r.mid_path]
            a2.plot(range(s0, s0 + len(mp)), mp, color="C1", lw=1)
            for j, (i, p) in enumerate(r.buys):
                x = s0 if i < 0 else s0 + 1 + i
                a1.scatter([x], [f[x]], marker="^" if j == 0 else "o", color="C2" if j == 0 else "orange", s=70, zorder=5)
                a2.scatter([x], [p], marker="^" if j == 0 else "o", color="C2" if j == 0 else "orange", s=50, zorder=5)
            a2.axhline(r.avg, color="purple", ls="--", lw=.8)
            if a.stop and not AVG:
                a2.axhline(r.avg * (1 - a.stop), color="#B00020", ls=":", lw=.8)
            if r.stop:
                x = s0 + 1 + r.stop[0]
                a1.scatter([x], [f[x]], marker="v", color="#B00020", s=70, zorder=5)
                a2.scatter([x], [r.stop[1]], marker="v", color="#B00020", s=50, zorder=5)
            for ax_ in (a1, a2):
                ax_.set_xticks(tk, [cols[q] for q in tk], fontsize=7); ax_.grid(alpha=.3); ax_.set_xlim(0, len(cols))
            a2.set_ylabel("콜 가격", fontsize=8); a1.set_ylabel("SPX", fontsize=8)
            tag = " · ".join(x for x in (r.event if r.event != "이벤트 없음" else "", r.special) if x)
            a1.set_title(f"#{r.rank}/{N}  {r.date:%Y-%m-%d}  수익률 {r.ret:+.0f}%  손익 ${r.pnl * 100:+,.0f}" + (f"   [{tag}]" if tag else ""),
                         fontsize=11, loc="left", color="#B00020" if r.special else "black")
            buy_txt = "\n".join(f"{'첫 매수' if j == 0 else f'물타기{j}'}: {hhmm(i)} @ {p:.2f} (${p * 100:,.0f})" for j, (i, p) in enumerate(r.buys))
            info = (f"신호: 개장 {r.f_open:,.1f} → 10:00 {r.f10:,.1f} ({r.r30 * 100:+.2f}%), MNQ 5분 구름 위\n"
                    f"{r.K:.0f} 콜\n{buy_txt}\n"
                    f"총 포지션: {r.n}계약, 총 투입 ${r.cost * 100:,.0f}, 평균 단가 {r.avg:.2f}\n"
                    + (f"손절: {hhmm(r.stop[0])} 중간가 {r.avg * (1 - a.stop):.2f} 이하 → 매수호가 {r.stop[1]:.2f}에 매도\n" if r.stop else "")
                    + f"만기: SPX 종가 {r.close:,.2f} → 콜 정산 {r.pay:.2f}/계약 (${r.pay * 100 * r.n:,.0f})\n"
                    f"손익 ${r.pnl * 100:+,.0f}  |  {'손절 없었으면' if a.stop else '물타기 없었으면'} ${r.pnl_single * 100:+,.0f}")
            a1.text(1.01, 1.0, info, transform=a1.transAxes, fontsize=7.5, va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="#F7F7F7", ec="#CCCCCC"))
        fig.subplots_adjust(left=0.06, right=0.70, top=0.95, bottom=0.05)
        pdf.savefig(fig); plt.close(fig)
print(f"끝: {N}건 → {OUT / f'trade_book_{TAG}.pdf'} ({(N + 1) // 2 + 1}쪽), 거래 목록 {OUT / f'trade_log_{TAG}.csv'}")
print(f"총손익 ${B.pnl.sum() * 100:,.0f}, 물타기 없이 ${B.pnl_single.sum() * 100:,.0f}, 물타기 발생 {(B.n > 1).sum()}건")
