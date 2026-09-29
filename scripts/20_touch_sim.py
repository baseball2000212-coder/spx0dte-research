"""
일목 구름 터치 매매 시뮬레이션 (72개 조합, 병렬 수 분).
  봉 2 (1분·5분) × 콜/풋 규칙 6 × 구름색 2 (전체·음운만) × 청산 3 (-30% 손절 / 구름이탈 / 15분 시간손절, 모두 -30% 포함)
  python scripts/20_touch_sim.py
결과: output/touch_trades.pkl, touch_configs.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, time, warnings, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from spx0dte.config import OPT_DIR, OUT, SPX_CSV
from spx0dte.core import load_spx_close, load_day
from spx0dte.touch import signal_grids, simulate_day

CONFIGS = [{"id": i, "tf": tf, "rule": r, "color": c, "exit": x} for i, (tf, r, c, x) in enumerate(itertools.product(
    (1, 5), ("반등", "돌파", "콜만", "60분추세", "반등+60분", "돌파+60분"), ("전체", "음운만"), ("손절30", "구름이탈", "15분")))]


def run(args):
    f, close, S = args
    warnings.filterwarnings("ignore")
    d = pd.Timestamp(f.name[:10])
    try:
        return simulate_day(load_day(d), d, close, S, CONFIGS)
    except Exception as ex:
        return f"{f.name[:10]} 실패: {ex}"


if __name__ == "__main__":
    t0 = time.time()
    FP = pd.read_pickle(OUT / "f_paths.pkl")
    G = signal_grids(FP)
    print(f"신호 격자 완료 ({time.time() - t0:.0f}초)", flush=True)
    close = load_spx_close(SPX_CSV)
    jobs = []
    for f in sorted(OPT_DIR.glob("*.dbn.zst")):
        d = pd.Timestamp(f.name[:10])
        if d in close.index and d in FP.index:
            jobs.append((f, float(close[d]), {k: v.loc[d].values.astype(float) for k, v in G.items()}))
    rows = []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for i, res in enumerate(ex.map(run, jobs, chunksize=4), 1):
            if isinstance(res, str):
                print(res); continue
            rows += res
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}일  ({time.time() - t0:.0f}초)", flush=True)
    T = pd.DataFrame(rows, columns=["cfg", "date", "entry", "exit", "leg", "K", "cost", "pnl", "why", "side", "color", "tr60"])
    T.to_pickle(OUT / "touch_trades.pkl")
    pd.DataFrame(CONFIGS).to_csv(OUT / "touch_configs.csv", index=False, encoding="utf-8-sig")
    print(f"끝: {len(jobs)}일, 거래 {len(T)}건  ({time.time() - t0:.0f}초)")
