"""ES(S&P500 선물) 1분봉 연결선물 다운로드 (이미 있으면 건너뜀). 2020-01 ~ 2026-09-23, 약 $8.68 (2026-09-26 사용자 승인)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import databento as db
from spx0dte.config import DATA, FUT_START, FUT_END, client

RAW, PQ = DATA / "es_1m.dbn.zst", DATA / "es_1m.parquet"
if not RAW.exists():
    c = client()
    p = dict(dataset="GLBX.MDP3", symbols=["ES.v.0"], stype_in="continuous", schema="ohlcv-1m", start=FUT_START, end=FUT_END)
    cost = c.metadata.get_cost(**p)
    assert cost <= 10, f"예상 비용 ${cost:.2f}가 상한 $10 초과 — 중단"
    print(f"비용 ${cost:.2f} 다운로드 시작")
    c.timeseries.get_range(**p, path=str(RAW))
es = db.DBNStore.from_file(str(RAW)).to_df()
es.to_parquet(PQ)
print(len(es), es.index.min(), es.index.max())
