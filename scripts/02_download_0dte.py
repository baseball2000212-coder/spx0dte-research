"""SPXW 0DTE cbbo-1m 다운로드: 그날 만기·전일 종가 ±3%·정규장만, 최근 날짜부터, 이어받기. 하루 약 $0.01."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import sys, time, warnings
from databento.common.error import BentoWarning
from spx0dte.config import OPT_DIR, SPX_CSV, OPT_START, OPT_END, OPT_BAND, OPT_BUDGET, client
from spx0dte.core import load_spx_ref, day_params

warnings.filterwarnings("ignore", category=BentoWarning)       # 상장 안 된 행사가 경고
start = sys.argv[1] if len(sys.argv) > 1 else OPT_START
end = sys.argv[2] if len(sys.argv) > 2 else OPT_END
c = client()
ref = load_spx_ref(SPX_CSV)
ref = ref[(ref.index >= start) & (ref.index <= end)]
OPT_DIR.mkdir(parents=True, exist_ok=True)
have = {f.name[:10] for f in OPT_DIR.glob("*.dbn.zst")}
print(f"대상 {len(ref)}일 ({start}~{end}), 이미 받은 날 {len(have)}일")

spent, done, skip, t0 = 0.0, 0, 0, time.time()
for d, r in ref.sort_index(ascending=False).items():
    if f"{d:%Y-%m-%d}" in have:
        continue
    p = day_params(d, float(r), OPT_BAND)
    try:
        cost = c.metadata.get_cost(**p)
    except Exception:
        skip += 1; continue                                     # 휴장일 등 0DTE 없는 날
    if spent + cost > OPT_BUDGET:
        print(f"예산 ${OPT_BUDGET} 도달 → 멈춤"); break
    try:
        c.timeseries.get_range(**p, path=str(OPT_DIR / f"{d:%Y-%m-%d}.dbn.zst"))
        spent += cost; done += 1
        print(f"{d:%Y-%m-%d} 완료 ${cost:.3f} 누적 ${spent:.2f} ({time.time()-t0:.0f}s)")
    except Exception as ex:                                     # 최신일은 403 license (실시간 라이선스 필요)
        skip += 1; print(f"{d:%Y-%m-%d} 실패: {str(ex)[:60]}")
print(f"끝: {done}일 추가, 건너뜀 {skip}일, ${spent:.2f}")
