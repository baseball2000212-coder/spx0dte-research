"""
0DTE ATM 스트래들 양매수 백테스트: 진입 시각별 → 16:00 SPX 종가 정산.
  python scripts/05_straddle.py --entries 15:00 15:30 15:45 15:55
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, warnings
import pandas as pd, databento as db
from spx0dte.config import OPT_DIR, OUT, SPX_CSV
from spx0dte.core import load_spx_close
from spx0dte.straddle import straddle_day, summarize_straddle

warnings.filterwarnings("ignore", category=RuntimeWarning)
ap = argparse.ArgumentParser()
ap.add_argument("--entries", nargs="+", default=["10:00", "12:00", "14:00", "15:00", "15:30", "15:45", "15:55"])
a = ap.parse_args()

close = load_spx_close(SPX_CSV)
OUT.mkdir(exist_ok=True); rows = []
files = sorted(OPT_DIR.glob("*.dbn.zst"))
for i, f in enumerate(files, 1):
    d = pd.Timestamp(f.name[:10])
    if d not in close.index:
        continue
    try:
        rows.append(straddle_day(db.DBNStore.from_file(str(f)).to_df(), d, float(close[d]), a.entries))
    except Exception as ex:
        print(f"{f.name[:10]} 실패: {ex}")
    if i % 100 == 0:
        print(f"{i}/{len(files)}")
t = pd.concat(rows, ignore_index=True)
t.to_csv(OUT / "straddle_trades.csv", index=False, encoding="utf-8-sig")
summ = summarize_straddle(t)
summ.to_csv(OUT / "straddle_summary.csv", encoding="utf-8-sig")
print(summ.to_string())
t["year"] = t.date.dt.year
print("\n연도별 평균 손익(mid):\n", t.pivot_table(index="year", columns="entry", values="pnl_mid", aggfunc="mean").round(2))
