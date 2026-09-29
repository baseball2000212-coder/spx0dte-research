"""
날짜별 SPX 일목균형표 차트 (1분 · 5분 · 60분). 몇 초.
  python scripts/17_ichimoku_charts.py 2026-09-21 2026-09-23 ...   (날짜 없으면 기본 예시)
결과: output/I_YYYY-MM-DD.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT
from spx0dte.ichimoku import bars, ichimoku, draw

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
DAYS = sys.argv[1:] or ["2026-09-21", "2026-09-22", "2026-09-23", "2025-04-09"]

FP = pd.read_pickle(OUT / "f_paths.pkl")
B = {n: bars(FP, n) for n in (1, 5, 60)}
IC = {n: ichimoku(b) for n, b in B.items()}


def ticks(ax, idx, fmt, every):
    pos = [i for i, t in enumerate(idx) if every(t)]
    ax.set_xticks(pos, [t.strftime(fmt) for t in idx[pos]], fontsize=7, rotation=0)


for day in DAYS:
    d = pd.Timestamp(day)
    fig, ax = plt.subplots(3, 1, figsize=(15, 14))
    # 1분: 그날
    m = B[1].index.normalize() == d
    b, ic = B[1][m], IC[1][m]
    draw(ax[0], b, ic, f"{day}  SPX 1분봉 일목균형표", candles=False)
    ticks(ax[0], b.index, "%H:%M", lambda t: t.minute == 0)
    ax[0].legend(fontsize=7, ncol=5, loc="upper left")
    # 5분: 전날 + 그날
    days = B[5].index.normalize().unique()
    i = days.get_loc(d)
    m = B[5].index.normalize().isin(days[max(0, i - 1): i + 1])
    b, ic = B[5][m], IC[5][m]
    draw(ax[1], b, ic, f"SPX 5분봉 (전날 + {day})")
    ticks(ax[1], b.index, "%m-%d %H:%M", lambda t: t.minute == 0 and t.hour % 2 == 0)
    # 60분: 최근 10거래일
    days = B[60].index.normalize().unique()
    i = days.get_loc(d)
    m = B[60].index.normalize().isin(days[max(0, i - 9): i + 1])
    b, ic = B[60][m], IC[60][m]
    draw(ax[2], b, ic, f"SPX 60분봉 (최근 10거래일, 마지막 = {day})")
    ticks(ax[2], b.index, "%m-%d", lambda t: t.hour == 9)
    fig.tight_layout(); fig.savefig(OUT / f"I_{day}.png", dpi=110); plt.close(fig)
    print("저장:", OUT / f"I_{day}.png")
