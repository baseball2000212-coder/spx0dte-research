"""
진입 시각 5개 × 행사가 비율 5개 콜 경로 (최적화용). 병렬 3~4분.
결과: output/calls_grid.pkl
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from spx0dte.config import OPT_DIR, OUT, SPX_CSV
from spx0dte.core import load_spx_close, load_day
from spx0dte.straddle import call_moneyness_day

ENTRIES = ["09:45", "10:00", "10:15", "10:30", "11:00"]


def run(args):
    f, close = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    try:
        return call_moneyness_day(load_day(d), d, close, ENTRIES)
    except Exception as ex:
        return f"{f.name[:10]} 실패: {ex}"


if __name__ == "__main__":
    close = load_spx_close(SPX_CSV)
    jobs = [(f, float(close[pd.Timestamp(f.name[:10])])) for f in sorted(OPT_DIR.glob("*.dbn.zst"))
            if pd.Timestamp(f.name[:10]) in close.index]
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for res in ex.map(run, jobs, chunksize=4):
            if isinstance(res, str):
                print(res); continue
            rows += res
    pd.DataFrame(rows).to_pickle(OUT / "calls_grid.pkl")
    print(f"끝: {len(jobs)}일, {len(rows)}건 ({time.time() - t0:.0f}초)")
