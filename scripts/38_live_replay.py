"""
실시간 엔진(live/engine.py)을 과거 데이터로 재생 → 백테스트 판단과 똑같은지 확인.
  실시간처럼: MNQ는 역조정 없는 원래 가격(연결선물 그대로), 10:00 이전 봉만 사용. SPX는 선도가격(옵션 패리티).
  python scripts/38_live_replay.py [--from 2026-01-01]
결과: output/live_replay.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, warnings
import numpy as np, pandas as pd
from spx0dte.config import OUT, SPX_CSV, FUT_PARQUET
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10
from live.engine import decide, NY

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--from", dest="start", default="2026-01-01"); a = ap.parse_args()

FP = pd.read_pickle(OUT / "f_paths.pkl")
X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), load_spx_ohlc(SPX_CSV))       # 백테스트 판단
raw = pd.read_parquet(FUT_PARQUET)
raw = raw[raw.symbol == "MNQ.v.0"].sort_index()
raw.index = raw.index.tz_convert(NY)
raw_bars = raw[["open", "high", "low", "close"]]
roll_days = set(raw.index[1:][raw.instrument_id.values[1:] != raw.instrument_id.values[:-1]].normalize().tz_localize(None))

rows = []
for d in FP.index[FP.index >= a.start]:
    day = f"{d:%Y-%m-%d}"
    hist = raw_bars[(raw_bars.index >= pd.Timestamp(day, tz=NY) - pd.Timedelta(days=4)) & (raw_bars.index < pd.Timestamp(f"{day} 10:00", tz=NY))]
    x = X.loc[d]
    dec = decide(day, x.f_open, x.f10, hist)
    rows.append({"date": day, "엔진": dec.buy, "백테스트": bool(x.signal), "일치": dec.buy == bool(x.signal),
                 "사유": dec.reason, "MNQ 09:55": dec.mnq_close_0955, "구름 위": dec.cloud_top,
                 "월물교체 근처": any(abs((d - r).days) <= 3 for r in roll_days)})
R = pd.DataFrame(rows)
R.to_csv(OUT / "live_replay.csv", index=False, encoding="utf-8-sig")
print(f"{len(R)}거래일, 일치 {R.일치.sum()} ({R.일치.mean() * 100:.1f}%), 엔진 매수 {R.엔진.sum()} / 백테스트 매수 {R.백테스트.sum()}")
bad = R[~R.일치]
if len(bad):
    print("불일치:\n", bad.to_string(index=False))
