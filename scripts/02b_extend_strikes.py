"""
±3% 밖으로 움직인 날의 빠진 행사가만 보충 다운로드 (갭·장중 급변 대응).
  필요 범위 = [min(전일종가, 시가, 저가) × (1 − margin), max(전일종가, 시가, 고가) × (1 + margin)]  (인베스팅 CSV)
  이미 받은 파일의 행사가 범위(최소~최대) 밖인 행사가만 받음 → data/0dte_ext/YYYY-MM-DD.dbn.zst
  고가·저가는 '어떤 행사가를 받을지'에만 쓰고 전략 판단엔 안 씀 → 미래 정보 누설 아님.

  python scripts/02b_extend_strikes.py              # 계획 + 비용 조회만 (돈 안 나감)
  python scripts/02b_extend_strikes.py --go         # 실제 다운로드 (이어받기, --budget 상한)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, databento as db
from spx0dte.config import OPT_DIR, EXT_DIR, OUT, SPX_CSV, client
from spx0dte.core import NY, load_spx_ohlc, parse_osi

ap = argparse.ArgumentParser()
ap.add_argument("--margin", type=float, default=0.01, help="고가·저가 바깥 여유 (0.01 = 1%)")
ap.add_argument("--budget", type=float, default=5.0, help="이번 실행 다운로드 총액 상한 ($)")
ap.add_argument("--go", action="store_true", help="실제 다운로드 (없으면 비용 조회만)")
a = ap.parse_args()


def strike_range(f):
    """이미 받은 파일의 행사가 최소·최대"""
    warnings.filterwarnings("ignore")
    try:
        _, K = parse_osi(db.DBNStore.from_file(str(f)).to_df()["symbol"].unique())
        return f.name[:10], K.min(), K.max()
    except Exception:
        return f.name[:10], np.nan, np.nan


def ext_params(d, strikes):
    s = pd.Timestamp(f"{d:%Y-%m-%d} 09:30", tz=NY).tz_convert("UTC")
    e = pd.Timestamp(f"{d:%Y-%m-%d} 16:01", tz=NY).tz_convert("UTC")
    syms = [f"SPXW  {d:%y%m%d}{cp}{int(k * 1000):08d}" for k in strikes for cp in "CP"]
    return dict(dataset="OPRA.PILLAR", schema="cbbo-1m", stype_in="raw_symbol", symbols=syms, start=s, end=e)


if __name__ == "__main__":
    # 1) 이미 받은 행사가 범위 (한 번 계산 후 캐시)
    cache = OUT / "strike_ranges.csv"
    files = sorted(OPT_DIR.glob("*.dbn.zst"))
    if cache.exists():
        have = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
            have = pd.DataFrame(list(ex.map(strike_range, files, chunksize=8)), columns=["date", "kmin", "kmax"])
        have["date"] = pd.to_datetime(have.date); have = have.set_index("date")
        OUT.mkdir(exist_ok=True); have.to_csv(cache)

    # 2) 날마다 필요한 범위 vs 이미 있는 범위 → 빠진 행사가
    px = load_spx_ohlc(SPX_CSV)
    px["prev"] = px.close.shift(1)
    plan = []
    for d, h in have.iterrows():
        if d not in px.index or not np.isfinite(h.kmin):
            continue
        r = px.loc[d]
        lo_need = min(r.prev, r.open, r.low) * (1 - a.margin)
        hi_need = max(r.prev, r.open, r.high) * (1 + a.margin)
        lo = np.floor(lo_need / 5) * 5; hi = np.ceil(hi_need / 5) * 5
        miss = [k for k in np.arange(lo, hi + 5, 5) if k < h.kmin or k > h.kmax]
        if miss:
            plan.append({"date": d, "빠진 행사가 수": len(miss), "아래쪽": sum(k < h.kmin for k in miss),
                         "위쪽": sum(k > h.kmax for k in miss), "strikes": miss,
                         "전일 대비 저가%": (r.low / r.prev - 1) * 100, "전일 대비 고가%": (r.high / r.prev - 1) * 100})
    plan = pd.DataFrame(plan)
    if plan.empty:
        print("보충할 날 없음"); sys.exit()
    plan.drop(columns="strikes").round(2).to_csv(OUT / "extend_plan.csv", index=False, encoding="utf-8-sig")
    print(f"보충이 필요한 날: {len(plan)}일 / {len(have)}일, 빠진 행사가 총 {plan['빠진 행사가 수'].sum()}개 (콜·풋 ×2 종목)")
    print(plan.drop(columns="strikes").sort_values("빠진 행사가 수", ascending=False).head(15).round(2).to_string(index=False))

    # 3) 비용 조회 (돈 안 나감) → --go면 다운로드
    try:
        cl = client()
    except AssertionError as ex:
        print(f"\n비용 조회 불가: {ex}"); sys.exit()
    EXT_DIR.mkdir(exist_ok=True)
    total, spent, done, skip, t0 = 0.0, 0.0, 0, 0, time.time()
    for p in plan.sort_values("date", ascending=False).itertuples():
        out = EXT_DIR / f"{p.date:%Y-%m-%d}.dbn.zst"
        if out.exists() and out.stat().st_size > 0:
            continue
        prm = ext_params(p.date, p.strikes)
        try:
            cost = cl.metadata.get_cost(**prm)
        except Exception as ex:
            skip += 1; print(f"{p.date:%Y-%m-%d} 비용 조회 실패: {str(ex)[:80]}"); continue
        total += cost
        if not a.go:
            continue
        if spent + cost > a.budget:
            print(f"예산 ${a.budget} 도달 → {p.date:%Y-%m-%d}부터 중단"); break
        try:
            cl.timeseries.get_range(**prm, path=str(out))
            spent += cost; done += 1
            print(f"{p.date:%Y-%m-%d} 완료  ${cost:.4f}  누적 ${spent:.3f}  ({time.time() - t0:.0f}초)")
        except Exception as ex:
            skip += 1; out.unlink(missing_ok=True); print(f"{p.date:%Y-%m-%d} 실패: {str(ex)[:80]}")
    if a.go:
        print(f"\n끝: {done}일 받음, 실패·건너뜀 {skip}일, 이번에 쓴 금액 ${spent:.3f}")
    else:
        print(f"\n예상 비용 합계 ${total:.3f} (아직 안 받음, 받으려면 --go)")
