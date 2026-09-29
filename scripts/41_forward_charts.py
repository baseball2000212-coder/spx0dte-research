"""
forward_log.csv의 날짜별 장중 차트: 위 = SPX 1분 (선도가격, 10:00 매수 ▲, 행사가 점선), 아래 = MNQ 5분봉 + 일목 구름 (09:55 봉 표시).
  python scripts/41_forward_charts.py [날짜 ...]   (없으면 기록 전체)
결과: output/forward_charts/YYYY-MM-DD.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, FUT_PARQUET
from spx0dte.core import NY, load_day
from spx0dte.straddle import _prep
from spx0dte.ichimoku import ichimoku, draw
from live.engine import to_5min

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
D = OUT / "forward_charts"; D.mkdir(exist_ok=True)
log = pd.read_csv(OUT / "forward_log.csv", encoding="utf-8-sig")
days = sys.argv[1:] or [d for d in log.date if not (D / f"{d}.png").exists()]          # 인자 없으면 아직 안 그린 날만
fut = pd.read_parquet(FUT_PARQUET)
mnq = fut[fut.symbol == "MNQ.v.0"][["open", "high", "low", "close"]]; mnq.index = mnq.index.tz_convert(NY)
b5 = to_5min(mnq); ic5 = ichimoku(b5)

for day in days:
    r = log[log.date == day].iloc[0]
    d = pd.Timestamp(day)
    M, bid, ask, cp, K, F = _prep(load_day(d), d)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1, 1.1]})
    x = ax[0]; Ff = F.ffill()
    x.plot(Ff.index.tz_localize(None), Ff.values, color="black", lw=1)
    t10 = pd.Timestamp(f"{day} 10:00")
    x.axvline(t10, color="grey", lw=.6)
    if r["신호"] == "매수":
        x.scatter([t10], [r["SPX 10:00"]], marker="^", color="#0F6E5A", s=90, zorder=5)
        x.axhline(r["행사가"], color="#3A86A8", ls=":", lw=1)
        title = (f"{day}  매수: {r['행사가']:.0f} 콜 @ {r['콜 중간가']:.2f} (중간가) → SPX 종가 {r['SPX 종가']:,.2f}, 정산 {r['정산']:.2f}"
                 f"  손익 ${r['손익$ (중간가)']:+,.0f}")
    else:
        title = f"{day}  쉼 — {r['사유'].replace('쉼: ', '')}"
    x.set_title(title, fontsize=11, loc="left", color="#0F6E5A" if r["신호"] == "매수" and r["손익$ (중간가)"] > 0 else ("#B3261E" if r["신호"] == "매수" else "#5E6A64"))
    x.set_ylabel("SPX (선도가격)"); x.grid(alpha=.3)
    x = ax[1]
    m = (b5.index >= pd.Timestamp(f"{day}", tz=NY) - pd.Timedelta(hours=6)) & (b5.index < pd.Timestamp(f"{day} 16:00", tz=NY))
    b, ic = b5[m], ichimoku(b5)[m]
    icf = ichimoku(b5)
    full = pd.DataFrame({"선행스팬1": np.nan, "선행스팬2": np.nan}, index=b.index)
    # 구름 그리기용: spx0dte.ichimoku.ichimoku 형식으로 다시 계산
    from spx0dte.ichimoku import ichimoku as ich_full
    icx = ich_full(b5)[m]
    draw(x, b, icx, "")
    i955 = list(b.index).index(pd.Timestamp(f"{day} 09:55", tz=NY)) if pd.Timestamp(f"{day} 09:55", tz=NY) in b.index else None
    if i955 is not None:
        x.scatter([i955], [b.close.iloc[i955]], s=80, facecolors="none", edgecolors="#B3261E", lw=2, zorder=6)
        x.annotate(f"09:55 봉 종가 {r['MNQ 09:55 종가']:,.2f}\n구름 윗선 {r['구름 윗선']:,.2f}", (i955, b.close.iloc[i955]),
                   xytext=(10, 20), textcoords="offset points", fontsize=8, color="#B3261E")
    tk = [k for k, t in enumerate(b.index) if t.minute == 0 and t.hour % 2 == 0]
    x.set_xticks(tk, [b.index[k].strftime("%H:%M") for k in tk], fontsize=8)
    x.set_title("MNQ 5분봉 + 일목 구름 (뉴욕 시간, 전날 밤부터)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(D / f"{day}.png", dpi=110); plt.close(fig)
    print("저장:", D / f"{day}.png")
