"""MNQ·MGC 1분봉 연결선물 다운로드 (이미 있으면 건너뜀). 약 $17."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import databento as db
from spx0dte.config import FUT_RAW, FUT_PARQUET, FUT_START, FUT_END, client

if not FUT_RAW.exists():
    client().timeseries.get_range(dataset="GLBX.MDP3", symbols=["MNQ.v.0", "MGC.v.0"], stype_in="continuous",
                                  schema="ohlcv-1m", start=FUT_START, end=FUT_END, path=str(FUT_RAW))
fut = db.DBNStore.from_file(str(FUT_RAW)).to_df()
fut.to_parquet(FUT_PARQUET)
print(fut.groupby("symbol").size(), fut.index.min(), fut.index.max(), sep="\n")
