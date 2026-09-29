"""
매수한 날(현재 규칙 신호일)의 산 콜 1종목 틱 데이터 (사용자 승인 2026-09-27, 약 $10).
  2023-03-28 이후: OPRA cmbp-1 (최우선호가가 바뀔 때마다 전부) → 1초 격자 bid·ask로 변환
  그 전 (2022-05 ~ 2023-03): trades (실제 체결가, 호가 없음 — 그 전엔 초 단위 호가 데이터가 없음)
구간: 09:58 ~ 16:00 ET (10:00:00 직전 호가 상태를 알기 위해 2분 앞부터).
  python scripts/55_download_ticks.py          비용 조회만
  python scripts/55_download_ticks.py --go     다운로드 (예산 --budget, 기본 $12) + 1초 변환
결과: data\\ticks\\{cmbp-1,trades}\\날짜.dbn.zst, ticks\\sec\\날짜.parquet (cmbp-1만, 1초 bid·ask)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, time, warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, databento as db
from spx0dte.config import DATA, OUT, client
from spx0dte.core import NY

warnings.filterwarnings("ignore")
TICK = DATA / "ticks"; SEC = TICK / "sec"
CMBP_START = pd.Timestamp("2023-03-28")


def params(d, k):
    schema = "cmbp-1" if d >= CMBP_START else "trades"
    return schema, dict(dataset="OPRA.PILLAR", schema=schema, stype_in="raw_symbol", symbols=[f"SPXW  {d:%y%m%d}C{int(k * 1000):08d}"],
                        start=pd.Timestamp(f"{d:%Y-%m-%d} 09:58", tz=NY).tz_convert("UTC"),
                        end=pd.Timestamp(f"{d:%Y-%m-%d} 16:00", tz=NY).tz_convert("UTC"))


def to_sec(f, d):
    """cmbp-1 → 09:59:00~15:59:59 1초 격자 (각 초 끝 시점의 최우선 bid·ask, 그 초 안의 최저 bid·최고 ask)."""
    df = db.DBNStore.from_file(str(f)).to_df()
    df.index = df.index.tz_convert(NY)
    q = df[["bid_px_00", "ask_px_00"]].astype(float)
    q = q[(q.ask_px_00 > 0) & (q.ask_px_00 >= q.bid_px_00)]
    g = pd.date_range(f"{d:%Y-%m-%d} 09:59:00", f"{d:%Y-%m-%d} 15:59:59", freq="1s", tz=NY)
    r = q.resample("1s", label="right", closed="left")                  # 라벨 t = [t−1초, t) 동안의 마지막 상태 → t 시점 상태
    out = pd.DataFrame({"bid": r.bid_px_00.last(), "ask": r.ask_px_00.last(), "bid_min": r.bid_px_00.min(), "ask_max": r.ask_px_00.max()})
    last = out[["bid", "ask"]].reindex(g.union(out.index)).ffill().reindex(g)
    ext = out[["bid_min", "ask_max"]].reindex(g)
    return pd.concat([last, ext], axis=1).astype("float32")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--go", action="store_true"); ap.add_argument("--budget", type=float, default=12.0)
    a = ap.parse_args()
    T = pd.read_csv(OUT / "rule_v2" / "trades.csv", index_col=0, parse_dates=True)
    cl = client(); t0 = time.time()
    for s in ("cmbp-1", "trades"):
        (TICK / s).mkdir(parents=True, exist_ok=True)
    SEC.mkdir(parents=True, exist_ok=True)

    def plan(d):
        schema, p = params(d, T.loc[d, "K"])
        out = TICK / schema / f"{d:%Y-%m-%d}.dbn.zst"
        if out.exists() and out.stat().st_size > 0:
            return None
        return d, schema, p, out, cl.metadata.get_cost(**p)
    with ThreadPoolExecutor(8) as ex:
        todo = [x for x in ex.map(plan, T.index) if x]
    tot = sum(x[4] for x in todo)
    by = pd.Series({s: sum(x[4] for x in todo if x[1] == s) for s in ("cmbp-1", "trades")})
    print(f"받을 날 {len(todo)} (cmbp-1 {sum(x[1] == 'cmbp-1' for x in todo)}, trades {sum(x[1] == 'trades' for x in todo)}), "
          f"비용 ${tot:.3f} (cmbp-1 ${by['cmbp-1']:.3f}, trades ${by['trades']:.3f}) ({time.time() - t0:.0f}초)", flush=True)
    if not a.go:
        sys.exit()
    if tot > a.budget:
        sys.exit(f"예산 ${a.budget} 초과 — 중단")

    def get(x):
        d, schema, p, out, c = x
        try:
            cl.timeseries.get_range(**p, path=str(out))
        except Exception as ex:
            out.unlink(missing_ok=True); print(f"{d:%Y-%m-%d} 실패: {str(ex)[:80]}", flush=True); return 0.0
        if schema == "cmbp-1":
            to_sec(out, d).to_parquet(SEC / f"{d:%Y-%m-%d}.parquet")
        return c
    with ThreadPoolExecutor(4) as ex:
        spent = sum(ex.map(get, todo))
    # 이미 받았는데 1초 변환이 없는 날
    for f in (TICK / "cmbp-1").glob("*.dbn.zst"):
        d = pd.Timestamp(f.name[:10])
        if not (SEC / f"{d:%Y-%m-%d}.parquet").exists():
            to_sec(f, d).to_parquet(SEC / f"{d:%Y-%m-%d}.parquet")
    size = sum(f.stat().st_size for f in TICK.rglob("*") if f.is_file()) / 1e9
    print(f"끝: ${spent:.3f} 사용, cmbp-1 {len(list((TICK / 'cmbp-1').glob('*.dbn.zst')))}일, trades {len(list((TICK / 'trades').glob('*.dbn.zst')))}일, "
          f"1초 변환 {len(list(SEC.glob('*.parquet')))}일, 용량 {size:.2f} GB ({time.time() - t0:.0f}초)")
