"""
10:00 진입 ATM·외가격(+5·+10·+20) 콜·풋의 매분 bid·mid·ask 경로 (물타기·중간가 체결용). 병렬 2~3분.
결과: output/legs10_full.pkl
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from spx0dte.config import OPT_DIR, OUT, SPX_CSV
from spx0dte.core import load_spx_close, load_day
from spx0dte.straddle import leg_strikes_day


def run(args):
    f, close = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    try:
        return leg_strikes_day(load_day(d), d, close, ["10:00"], offsets=(0, 5, 10, 20), full=True)
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
    pd.DataFrame(rows).to_pickle(OUT / "legs10_full.pkl")
    print(f"끝: {len(jobs)}일, {len(rows)}건  ({time.time() - t0:.0f}초)")
