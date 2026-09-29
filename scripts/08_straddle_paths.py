"""
진입 후 매분 스트래들 청산가(bid 합) 경로 저장 → 손절·익절 규칙 테스트용 (병렬, 1~2분).
  python scripts/08_straddle_paths.py
결과: output/straddle_paths.pkl, straddle_trades.csv, straddle_summary.csv (05_straddle.py 대체)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import pandas as pd, databento as db
from spx0dte.config import OPT_DIR, OUT, SPX_CSV
from spx0dte.core import load_spx_close, load_day
from spx0dte.features import PATH_ENTRIES
from spx0dte.straddle import straddle_paths_day, trades_from_paths, summarize_straddle


def run(args):
    f, close = args
    warnings.filterwarnings("ignore")
    try:
        return straddle_paths_day(load_day(pd.Timestamp(f.name[:10])), pd.Timestamp(f.name[:10]), close, PATH_ENTRIES)
    except Exception as ex:
        return f"{f.name[:10]} 실패: {ex}"


if __name__ == "__main__":
    close = load_spx_close(SPX_CSV)
    jobs = [(f, float(close[pd.Timestamp(f.name[:10])])) for f in sorted(OPT_DIR.glob("*.dbn.zst"))
            if pd.Timestamp(f.name[:10]) in close.index]
    OUT.mkdir(exist_ok=True); t0 = time.time(); rows = []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for i, res in enumerate(ex.map(run, jobs, chunksize=4), 1):
            if isinstance(res, str):
                print(res); continue
            rows += res
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}일  ({time.time() - t0:.0f}초)", flush=True)
    P = pd.DataFrame(rows)
    P.to_pickle(OUT / "straddle_paths.pkl")
    t = trades_from_paths(P)
    t.to_csv(OUT / "straddle_trades.csv", index=False, encoding="utf-8-sig")
    summ = summarize_straddle(t)
    summ.to_csv(OUT / "straddle_summary.csv", encoding="utf-8-sig")
    print(summ.to_string())
    print(f"끝: {len(jobs)}일, {len(rows)}건  ({time.time() - t0:.0f}초)")
