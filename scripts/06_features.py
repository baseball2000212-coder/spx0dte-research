"""
파일 한 바퀴로 한꺼번에: 장중 ATM IV 곡선 + 일별 변동성 피처 + 델타헤지 스트래들 (병렬, 수 분).
  python scripts/06_features.py
결과: output/daily_features.csv, output/delta_hedge.csv, output/atm_iv_curve.pkl
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, databento as db
from spx0dte.config import OPT_DIR, OUT, SPX_CSV
from spx0dte.core import load_spx_ref, load_day
from spx0dte.features import features_day


def run(args):
    f, ref_close = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    try:
        return features_day(load_day(pd.Timestamp(f.name[:10])), d, ref_close)
    except Exception as ex:
        return f"{f.name[:10]} 실패: {ex}"


if __name__ == "__main__":
    ref = load_spx_ref(SPX_CSV)
    files = sorted(OPT_DIR.glob("*.dbn.zst"))
    jobs = [(f, float(ref.get(pd.Timestamp(f.name[:10]), np.nan))) for f in files]
    OUT.mkdir(exist_ok=True); t0 = time.time()
    daily, dh, curves = [], [], {}
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for i, res in enumerate(ex.map(run, jobs, chunksize=4), 1):
            if isinstance(res, str):
                print(res); continue
            row, iv, x = res
            daily.append(row); curves[row["date"]] = iv; dh += x
            if i % 100 == 0:
                print(f"  {i}/{len(files)}일  ({time.time() - t0:.0f}초)", flush=True)
    daily = pd.DataFrame(daily).sort_values("date")
    daily.to_csv(OUT / "daily_features.csv", index=False)
    pd.DataFrame(dh).to_csv(OUT / "delta_hedge.csv", index=False)
    pd.DataFrame(curves).T.sort_index().to_pickle(OUT / "atm_iv_curve.pkl")
    print(f"끝: {len(daily)}일, 델타헤지 {len(dh)}건  ({time.time() - t0:.0f}초)")
