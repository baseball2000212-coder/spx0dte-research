"""
진짜 표본외 검증: 2020-01-02 ~ 2022-05-13 SPXW 0DTE (규칙 고정, output/oos/사전등록_판정기준.md 참고).
  python scripts/43_oos.py            비용 조회만
  python scripts/43_oos.py --go       다운로드 (예산 --budget, 기본 $3) → data/0dte_oos/
  python scripts/43_oos.py --eval     평가 (다운로드된 날 전부)
결과: output/oos/days.csv, result.json, *.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, json, os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, databento as db
from spx0dte.config import DATA, OUT, SPX_CSV, FUT_PARQUET, client
from spx0dte.core import NY, load_spx_ohlc
from spx0dte.straddle import _prep, pick_atm
from spx0dte.exits import FEE, EXERCISE

OOS = DATA / "0dte_oos"; R = OUT / "oos"
START, END = "2020-01-02", "2022-05-13"


def params(d, strikes):
    s = pd.Timestamp(f"{d:%Y-%m-%d} 09:30", tz=NY).tz_convert("UTC"); e = pd.Timestamp(f"{d:%Y-%m-%d} 16:01", tz=NY).tz_convert("UTC")
    return dict(dataset="OPRA.PILLAR", schema="cbbo-1m", stype_in="raw_symbol", start=s, end=e,
                symbols=[f"SPXW  {d:%y%m%d}{cp}{int(k * 1000):08d}" for k in strikes for cp in "CP"])


def strikes_for(d, px):
    r = px.loc[d]; pc = px.close[px.index < d]; prev = pc.iloc[-1] if len(pc) else r.open
    lo = np.floor(min(prev, r.open, r.low) * 0.99 / 5) * 5; hi = np.ceil(max(prev, r.open, r.high) * 1.01 / 5) * 5
    return np.arange(lo, hi + 5, 5)


def evaluate(args):
    f, close, mnq_hist = args
    warnings.filterwarnings("ignore")
    from live.engine import decide
    d = pd.Timestamp(f.name[:10])
    try:
        M, bid, ask, cp, K, F = _prep(db.DBNStore.from_file(str(f)).to_df(), d)
        t = lambda hm: pd.Timestamp(f"{d:%Y-%m-%d} {hm}", tz=NY)
        fo = F.loc[t("09:31"): t("09:35")].dropna()
        if fo.empty or not np.isfinite(F.get(t("10:00"), np.nan)):
            return {"date": d, "오류": "선도가격 없음"}
        f_open, f10, f959 = float(fo.iloc[0]), float(F[t("10:00")]), float(F.get(t("09:59"), np.nan))
        dec = decide(f"{d:%Y-%m-%d}", f_open, f10, mnq_hist)
        best = pick_atm(M, ask, cp, K, f10, t("10:00"))
        row = {"date": d, "SPX 개장": f_open, "SPX 10:00": f10, "SPX 09:59": f959, "조건1": f10 > f_open, "조건1(09:59)": f959 > f_open,
               "조건2": dec.mnq_above_cloud, "신호": dec.buy, "사유": dec.reason, "SPX 종가": close}
        if best is None:
            row["오류"] = "ATM 없음"; return row
        k, c, _ = best
        pay = max(close - k, 0.0); cost_fee = FEE + (EXERCISE if pay > 0 else 0)
        mid, a_ = float(M.at[t("10:00"), c]), float(ask.at[t("10:00"), c])
        m1 = float(M[c].get(t("10:01"), np.nan))
        row.update({"행사가": k, "중간가": mid, "매도호가": a_, "중간가 10:01": m1, "정산": pay,
                    "손익$": (pay - mid - cost_fee) * 100, "손익$ 매도호가": (pay - a_ - cost_fee) * 100,
                    "손익$ 10:01": (pay - m1 - cost_fee) * 100 if np.isfinite(m1) else np.nan})
        return row
    except Exception as ex:
        return {"date": d, "오류": str(ex)[:80]}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(); ap.add_argument("--go", action="store_true"); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--budget", type=float, default=3.0); a = ap.parse_args()
    px = load_spx_ohlc(SPX_CSV); R.mkdir(exist_ok=True); OOS.mkdir(exist_ok=True)
    days = px.index[(px.index >= START) & (px.index <= END)]

    if not a.eval:
        from concurrent.futures import ThreadPoolExecutor
        cl = client(); t0 = time.time()
        def q(d):
            p = params(d, strikes_for(d, px))
            try:
                return (d, p, cl.metadata.get_cost(**p))
            except Exception:
                return None                                            # 그날 만기 없음 (심볼 없음)
        todo = [d for d in days if not (OOS / f"{d:%Y-%m-%d}.dbn.zst").exists()]
        with ThreadPoolExecutor(8) as ex:
            plan = [x for x in ex.map(q, todo) if x]
        total = sum(c for *_, c in plan)
        print(f"만기 있는 날 {len(plan)} / 거래일 {len(days)}, 예상 비용 ${total:.3f} ({time.time() - t0:.0f}초)", flush=True)
        if a.go:
            if total > a.budget:
                sys.exit(f"예산 ${a.budget} 초과 — 중단")
            def g(x):
                d, p, c = x; out = OOS / f"{d:%Y-%m-%d}.dbn.zst"
                try:
                    cl.timeseries.get_range(**p, path=str(out)); return c
                except Exception as ex:
                    out.unlink(missing_ok=True); print(f"{d:%Y-%m-%d} 실패: {str(ex)[:60]}"); return 0.0
            with ThreadPoolExecutor(4) as ex:
                spent = sum(ex.map(g, plan))
            print(f"다운로드 끝: {len(list(OOS.glob('*.dbn.zst')))}일, ${spent:.3f} ({time.time() - t0:.0f}초)")
        sys.exit()

    # ── 평가 ──
    raw = pd.read_parquet(FUT_PARQUET); raw = raw[raw.symbol == "MNQ.v.0"].sort_index(); raw.index = raw.index.tz_convert(NY)
    bars = raw[["open", "high", "low", "close"]]
    jobs = []
    for f in sorted(OOS.glob("*.dbn.zst")):
        d = pd.Timestamp(f.name[:10])
        if d not in px.index:
            continue
        t10 = pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=NY)
        jobs.append((f, float(px.close[d]), bars[(bars.index >= t10 - pd.Timedelta(days=4)) & (bars.index < t10)]))
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = list(ex.map(evaluate, jobs, chunksize=4))
    Dd = pd.DataFrame(rows).set_index("date").sort_index()
    Dd.to_csv(R / "days.csv", encoding="utf-8-sig")
    print(f"평가한 날 {len(Dd)}, 오류 {Dd['오류'].notna().sum() if '오류' in Dd else 0}")
