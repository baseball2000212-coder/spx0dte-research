"""
방향성 단일 레그(콜만 / 풋만 매수)용 경로 저장 (병렬, 2~3분).
  python scripts/12_leg_paths.py
결과: output/leg_paths.pkl (진입별 콜·풋 ask, 매분 bid), output/f_paths.pkl (날짜 × 분, SPX 1분 대용 F)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from spx0dte.config import OPT_DIR, OUT, SPX_CSV
from spx0dte.core import load_spx_close, load_day
from spx0dte.straddle import leg_paths_day

LEG_ENTRIES = ["09:45", "10:00", "10:30", "11:00", "15:30"]


def run(args):
    f, close = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    try:
        return leg_paths_day(load_day(d), d, close, LEG_ENTRIES)
    except Exception as ex:
        return f"{f.name[:10]} 실패: {ex}"


if __name__ == "__main__":
    close = load_spx_close(SPX_CSV)
    jobs = [(f, float(close[pd.Timestamp(f.name[:10])])) for f in sorted(OPT_DIR.glob("*.dbn.zst"))
            if pd.Timestamp(f.name[:10]) in close.index]
    OUT.mkdir(exist_ok=True); t0 = time.time(); rows, fp = [], []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for i, res in enumerate(ex.map(run, jobs, chunksize=4), 1):
            if isinstance(res, str):
                print(res); continue
            rows += res[0]; fp.append(res[1])
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}일  ({time.time() - t0:.0f}초)", flush=True)
    pd.DataFrame(rows).to_pickle(OUT / "leg_paths.pkl")
    pd.DataFrame(fp).sort_index().to_pickle(OUT / "f_paths.pkl")
    print(f"끝: {len(fp)}일, {len(rows)}건  ({time.time() - t0:.0f}초)")
