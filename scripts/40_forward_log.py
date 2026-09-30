"""
앞으로 매일: 새 날짜 데이터 받기 → 10:00 신호 판단 → 가상 체결 → output/forward_log.csv에 한 줄씩 쌓기.
  python scripts/40_forward_log.py                 비용만 조회 (돈 안 나감) + 이미 받은 날 기록
  python scripts/40_forward_log.py --go            실제 다운로드 (예산 --budget, 기본 $1) 후 기록
  받는 것: SPXW 0DTE 1분 호가 (전일 종가 ±4% 행사가), MNQ 1분봉 (마지막 저장 이후)
  판단은 live/engine.py (실시간 엔진과 같은 코드). 정산은 인베스팅 CSV 종가, 없으면 15:59 선도가격으로 추정 표시.
  매수일은 산 콜의 초 단위 호가(cmbp-1, 09:59:50~16:00, 하루 약 $0.03~0.05)를 받아 5초·10초 뒤 매도호가로 체결,
  초 단위 중간가 −90% 손절(마감 직전 포함)·수수료 $5까지 반영한 손익을 기록 (사용자 기본값, 2026-09-30).
  예산 기본값 $0.10 (사용자 승인 2026-09-30).
  Databento 최신 1일은 라이선스 문제로 403이 날 수 있음 → 다음 날 다시 실행하면 받아짐.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, warnings, datetime as dt
import numpy as np, pandas as pd, databento as db
from spx0dte.config import OPT_DIR, OUT, SPX_CSV, FUT_PARQUET, DATA, client
from spx0dte.core import NY, load_spx_close, day_params, load_day
from spx0dte.straddle import _prep, pick_atm
from spx0dte.exits import FEE, EXERCISE
from live.engine import decide
from types import SimpleNamespace
from spx0dte.strategy import trade_rule
from spx0dte.realistic import rule_at, sec_stop, DELAYS
from live.engine import tick_up

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--go", action="store_true"); ap.add_argument("--budget", type=float, default=0.10)
ap.add_argument("--band", type=float, default=0.04)
a = ap.parse_args()
LOG = OUT / "forward_log.csv"
close = load_spx_close(SPX_CSV)
# CBOE 공식 SPX 일별 종가 (무료, 로그인 없음, 2026-09-30 추가) → 인베스팅 CSV에 없는 최근 날짜를 채움
CBOE_SPX = DATA / "cboe_SPX.csv"
try:
    import requests, io
    _t = requests.get("https://cdn.cboe.com/api/global/us_indices/daily_prices/SPX_History.csv", headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    _t.raise_for_status(); CBOE_SPX.write_bytes(_t.content)
except Exception as _ex:
    print("CBOE SPX 종가 받기 실패 (저장된 파일 사용):", str(_ex)[:60])
if CBOE_SPX.exists():
    _c = pd.read_csv(CBOE_SPX); _c.columns = [c.strip().lower() for c in _c.columns]
    _c = pd.Series(_c.iloc[:, 1].astype(float).values, index=pd.to_datetime(_c.iloc[:, 0], format="%m/%d/%Y"))
    close = pd.concat([close, _c[~_c.index.isin(close.index)]]).sort_index()
MANUAL = DATA / "spx_close_manual.csv"                              # 사용자가 직접 알려준 종가 (date,close) — CSV에 없을 때 사용
if MANUAL.exists():
    man = pd.read_csv(MANUAL, parse_dates=["date"]).set_index("date")["close"]
    close = pd.concat([close, man[~man.index.isin(close.index)]]).sort_index()
have = sorted(pd.Timestamp(f.name[:10]) for f in OPT_DIR.glob("*.dbn.zst"))
today = pd.Timestamp(dt.date.today())
todo = [d for d in pd.bdate_range(have[-1] + pd.Timedelta(days=1), today) if d < today]   # 오늘 장은 끝난 뒤에

# ── 1. 비용 조회 / 다운로드 ──
cl = client()
fut = pd.read_parquet(FUT_PARQUET)
last_ts = fut[fut.symbol == "MNQ.v.0"].index.max()
spent, plan = 0.0, []
# Databento는 가장 최근 구간을 늦게 공개함 (OPRA는 약 하루) → 공개된 범위까지만 요청, 나머지는 다음 실행 때 자동으로
opra_end = pd.Timestamp(cl.metadata.get_dataset_range(dataset="OPRA.PILLAR")["end"])
glbx_end = pd.Timestamp(cl.metadata.get_dataset_range(dataset="GLBX.MDP3")["end"])
for d in todo:
    prev = close[close.index < d]
    if prev.empty:
        continue
    if pd.Timestamp(f"{d:%Y-%m-%d} 16:01", tz=NY).tz_convert("UTC") > opra_end:
        print(f"{d:%Y-%m-%d} 옵션: 아직 공개 전 (공개된 끝 {opra_end.tz_convert(NY):%m-%d %H:%M} ET) — 다음 실행 때 받음")
        continue
    p = day_params(d, float(prev.iloc[-1]), a.band)
    try:
        c = cl.metadata.get_cost(**p); plan.append((d, p, c))
    except Exception as ex:
        print(f"{d:%Y-%m-%d} 옵션 비용 조회 실패 (휴장일이거나 아직 준비 안 됨): {str(ex)[:70]}")
fend = min((todo[-1] + pd.Timedelta(days=1)).tz_localize("UTC") if todo else glbx_end, glbx_end)
fp = dict(dataset="GLBX.MDP3", symbols=["MNQ.v.0"], stype_in="continuous", schema="ohlcv-1m",
          start=last_ts + pd.Timedelta(minutes=1), end=fend)
try:
    fcost = cl.metadata.get_cost(**fp) if fp["start"] < fend else 0.0
except Exception as ex:
    fcost = None; print("MNQ 비용 조회 실패:", str(ex)[:80])
total = sum(c for *_, c in plan) + (fcost or 0)
print(f"받을 날짜: {[f'{d:%m-%d}' for d, *_ in plan]}  옵션 ${sum(c for *_, c in plan):.3f} + MNQ ${fcost or 0:.3f} = ${total:.3f}")
STATUS = OUT / "forward_status.txt"
note = lambda s: open(STATUS, "a", encoding="utf-8").write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M}  {s}\n")
over = total > a.budget
if a.go and plan and over:
    print(f"예산 ${a.budget} 초과 (${total:.3f}) — 이번엔 안 받음")
    note(f"예산 초과로 다운로드 건너뜀: {[f'{d:%m-%d}' for d, *_ in plan]} ${total:.3f} > ${a.budget}")
if a.go and not over:
    for d, p, c in plan:
        out = OPT_DIR / f"{d:%Y-%m-%d}.dbn.zst"
        try:
            cl.timeseries.get_range(**p, path=str(out)); spent += c; print(f"{d:%Y-%m-%d} 옵션 받음 ${c:.4f}")
        except Exception as ex:
            out.unlink(missing_ok=True); print(f"{d:%Y-%m-%d} 옵션 실패: {str(ex)[:80]}")
    if fcost and fp["start"] < fend:
        tmp = DATA / f"fut_1m_add_{today:%Y%m%d}.dbn.zst"
        tmp.unlink(missing_ok=True)                                   # 같은 날 다시 돌릴 때 "이미 있음" 오류 방지
        try:
            cl.timeseries.get_range(**fp, path=str(tmp)); spent += fcost
            add = db.DBNStore.from_file(str(tmp)).to_df()
            fut = pd.concat([fut, add]).reset_index().drop_duplicates(subset=["ts_event", "symbol"], keep="last").set_index("ts_event").sort_index()
            fut.to_parquet(FUT_PARQUET); print(f"MNQ {len(add)}봉 추가 ${fcost:.4f}")
        except Exception as ex:
            print("MNQ 실패:", str(ex)[:80])
    print(f"이번에 쓴 금액 ${spent:.3f}")

# ── 2. 기록 (옵션 파일 있고 아직 기록 안 된 날) ──
def save_log(df):
    """날짜는 항상 YYYY-MM-DD 문자열로 저장 (섞이면 41번 그림 파일 이름에 시간이 붙어 실패함, 2026-09-30)."""
    out = df.copy(); out["date"] = pd.to_datetime(out["date"], format="mixed").dt.strftime("%Y-%m-%d")
    out.to_csv(LOG, index=False, encoding="utf-8-sig")


log = pd.read_csv(LOG, parse_dates=["date"], encoding="utf-8-sig") if LOG.exists() else pd.DataFrame(columns=["date"])
if len(log):
    log["date"] = pd.to_datetime(log["date"], format="mixed")
done = set(log.date.dt.normalize()) if len(log) else set()
# 추정 종가로 기록했던 날 → 실제 종가가 생겼으면 고침
fixed = 0
if len(log) and "종가 출처" in log:
    for i, r in log[log["종가 출처"] == "15:59 선도가격 추정"].iterrows():
        d = pd.Timestamp(r.date)
        if d in close.index:
            pay = max(float(close[d]) - r["행사가"], 0.0)
            ex = EXERCISE if pay > 0 else 0
            log.loc[i, ["SPX 종가", "정산", "종가 출처"]] = [round(float(close[d]), 2), round(pay, 2), "실제 종가 (나중에 고침)"]
            log.loc[i, "손익$ (중간가)"] = round((pay - r["콜 중간가"] - FEE - ex) * 100, 1)
            log.loc[i, "손익$ (매도호가)"] = round((pay - r["콜 매도호가"] - FEE - ex) * 100, 1)
            for cc in [c for c in log.columns if c.startswith("손익$ ") and "초(−90%)" in c] + ["손익$ 규칙(중간가·−90%)"]:
                log.loc[i, cc] = np.nan                                       # 실제 종가로 5·10초 손익 다시 계산
            fixed += 1
    if fixed:
        save_log(log); print(f"추정 종가 {fixed}건 실제 종가로 고침")
PATHS = {}


def stop_cols(M, bid, ask, F, d, k, c, pay):
    """현재 규칙(−90% 손절, 2026-09-27~): 10:00 중간가 매수 → 매분 중간가가 10% 이하면 그 분 매수호가로 매도."""
    t10 = pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=NY); after = M.index > t10
    intr = np.maximum(F.ffill()[after].values - k, 0)
    m, b = M.loc[after, c].values, bid.loc[after, c].values
    r = SimpleNamespace(mid=float(M.at[t10, c]), ask=float(ask.at[t10, c]), pay=pay, mid_path=np.where(np.isfinite(m), m, intr),
                        bid_path=np.where(np.isfinite(b), b, intr), ask_path=ask.loc[after, c].values)
    t = trade_rule(r)
    PATHS[(f"{d:%Y-%m-%d}", float(k))] = r                          # 5·10초 체결 손익 계산에 재사용
    return {"손절(−90%)": (t10 + pd.Timedelta(minutes=t["stop"][0] + 1)).strftime("%H:%M") + f" @ {t['stop'][1]:.2f}" if t["stop"] else "없음",
            "손익$ 규칙(중간가·−90%)": round(t["pnl"] * 100, 1)}


mnq = fut[fut.symbol == "MNQ.v.0"][["open", "high", "low", "close"]].copy()
mnq.index = mnq.index.tz_convert(NY)
if len(log) and "신호" in log:
    need = log[(log["신호"] == "매수") & (log.get("손익$ 규칙(중간가·−90%)", pd.Series(np.nan, index=log.index)).isna())]
    for i, r0 in need.iterrows():
        d = pd.Timestamp(r0.date); f0 = OPT_DIR / f"{d:%Y-%m-%d}.dbn.zst"
        if f0.exists() and r0["종가 출처"] != "15:59 선도가격 추정":
            M, bid, ask, cp, K, F = _prep(load_day(d), d)
            c = M.columns[(K == r0["행사가"]) & (cp == "C")][0]
            for kk, vv in stop_cols(M, bid, ask, F, d, r0["행사가"], c, r0["정산"]).items():
                log.loc[i, kk] = vv
            print(f"{d:%Y-%m-%d} −90% 손절 칸 채움")
    save_log(log)
if len(log) and ("요일" not in log or log["요일"].isna().any()):
    log["요일"] = pd.to_datetime(log.date).dt.dayofweek.map(dict(enumerate("월화수목금")))
    save_log(log)
start = pd.Timestamp("2026-09-24")                                     # 백테스트 끝(9/23) 다음부터 = 앞으로의 기록
rows = []
for f in sorted(OPT_DIR.glob("*.dbn.zst")):
    d = pd.Timestamp(f.name[:10])
    if d < start or d in done:
        continue
    M, bid, ask, cp, K, F = _prep(load_day(d), d)
    Ff = F.ffill()
    t931, t10 = pd.Timestamp(f"{d:%Y-%m-%d} 09:31", tz=NY), pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=NY)
    f_open = F.loc[t931: t931 + pd.Timedelta(minutes=4)].dropna()
    f_open = float(f_open.iloc[0]) if len(f_open) else np.nan
    hist = mnq[(mnq.index >= t10 - pd.Timedelta(days=4)) & (mnq.index < t10)]
    dec = decide(f"{d:%Y-%m-%d}", f_open, float(F.get(t10, np.nan)), hist)
    row = {"date": f"{d:%Y-%m-%d}", "요일": "월화수목금"[d.dayofweek], "SPX 개장(09:31)": round(f_open, 2), "SPX 10:00": round(float(F.get(t10, np.nan)), 2),
           "MNQ 09:55 종가": dec.mnq_close_0955, "구름 윗선": round(dec.cloud_top, 2), "신호": "매수" if dec.buy else "쉼", "사유": dec.reason}
    if dec.buy:
        best = pick_atm(M, ask, cp, K, F[t10], t10)
        if best is not None:
            k, c, _ = best
            px_close = float(close[d]) if d in close.index else float(Ff.iloc[-1])
            pay = max(px_close - k, 0.0)
            mid, a_ = float(M.at[t10, c]), float(ask.at[t10, c])
            row.update({"행사가": k, "콜 매수호가": float(bid.at[t10, c]), "콜 매도호가": a_, "콜 중간가": mid,
                        "SPX 종가": round(px_close, 2), "종가 출처": ("사용자 입력" if MANUAL.exists() and d in pd.read_csv(MANUAL, parse_dates=["date"]).date.values else "CBOE·인베스팅 종가") if d in close.index else "15:59 선도가격 추정",
                        "정산": round(pay, 2),
                        "손익$ (중간가)": round((pay - mid - FEE - (EXERCISE if pay > 0 else 0)) * 100, 1),
                        "손익$ (매도호가)": round((pay - a_ - FEE - (EXERCISE if pay > 0 else 0)) * 100, 1)})
            row.update(stop_cols(M, bid, ask, F, d, k, c, pay))
    row["기록 시각"] = f"{pd.Timestamp.now():%Y-%m-%d %H:%M}"
    rows.append(row); print(row)
if rows:
    log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True) if len(log) else pd.DataFrame(rows)
    save_log(log)
    print(f"forward_log.csv에 {len(rows)}줄 추가 (총 {len(log)}줄)")
    note(f"기록 {len(rows)}줄 추가: " + ", ".join(f"{r['date']} {r['신호']}" + (f" ${r.get('손익$ (중간가)', 0):+,.0f}" if r['신호'] == '매수' else "") for r in rows)
         + f" (다운로드 ${spent:.3f})")
else:
    print("새로 기록할 날 없음")
    note(f"새로 기록할 날 없음 (다운로드 ${spent:.3f})")

# ── 3. 5초·10초 뒤 매도호가 체결 + 초 단위 손절 (산 콜 하나, 09:59:50~16:00 초 단위 호가) ──
TICKD = DATA / "ticks" / "fwd"; TICKD.mkdir(parents=True, exist_ok=True)


def tick_params(d, k):
    return dict(dataset="OPRA.PILLAR", schema="cmbp-1", stype_in="raw_symbol", symbols=[f"SPXW  {d:%y%m%d}C{int(k * 1000):08d}"],
                start=pd.Timestamp(f"{d:%Y-%m-%d} 09:59:50", tz=NY).tz_convert("UTC"), end=pd.Timestamp(f"{d:%Y-%m-%d} 16:00", tz=NY).tz_convert("UTC"))


def to_sec(q, d):
    """호가 변경 기록 → 10:00:00~15:59:59 1초 격자 (라벨 t = t 순간의 상태, 백테스트 55번과 같은 정의)."""
    g = pd.date_range(f"{d:%Y-%m-%d} 10:00:00", f"{d:%Y-%m-%d} 15:59:59", freq="1s", tz=NY)
    x = q[["bid_px_00", "ask_px_00"]].rename(columns={"bid_px_00": "bid", "ask_px_00": "ask"}).astype(float)
    x = x[(x.ask > 0) & (x.ask >= x.bid)]
    r = x.resample("1s", label="right", closed="left").last()
    return r.reindex(g.union(r.index)).ffill().reindex(g)


def ask_at(q, t):
    """t 순간의 매도호가 (t 이전 마지막 호가 상태)."""
    q = q[q.index < t]
    return float(q.ask_px_00.iloc[-1]) if len(q) else np.nan


log = pd.read_csv(LOG, encoding="utf-8-sig"); log["date"] = pd.to_datetime(log["date"], format="mixed")
col10 = "손익$ 10초(−90%)"
if len(log) and "손절 10초" not in log and col10 in log:                   # 예전(1분 손절·10시 앞뒤 틱만) 값은 지우고 다시 계산
    log[[c for c in log.columns if c.startswith("손익$ ") and "초(−90%)" in c]] = np.nan
todo_t = log[(log["신호"] == "매수") & (log[col10].isna() if col10 in log else True)] if len(log) else log
for i, r0 in todo_t.iterrows():
    d, k = pd.Timestamp(r0.date), float(r0["행사가"])
    f = TICKD / f"{d:%Y-%m-%d}_{int(k)}_full.dbn.zst"
    if not f.exists():
        tp = tick_params(d, k)
        try:
            c = cl.metadata.get_cost(**tp)
        except Exception as ex:
            print(f"{d:%Y-%m-%d} 틱 비용 조회 실패: {str(ex)[:70]}"); continue
        if not a.go or spent + c > a.budget:
            print(f"{d:%Y-%m-%d} 틱 ${c:.4f} — {'--go 없음' if not a.go else '예산 초과'}, 다음 실행 때"); continue
        try:
            cl.timeseries.get_range(**tp, path=str(f)); spent += c; print(f"{d:%Y-%m-%d} 틱 받음 ${c:.4f}")
        except Exception as ex:
            f.unlink(missing_ok=True); print(f"{d:%Y-%m-%d} 틱 실패: {str(ex)[:70]}"); continue
    q = db.DBNStore.from_file(str(f)).to_df(); q.index = q.index.tz_convert(NY); q = q[q.ask_px_00 > 0].sort_index()
    if q.index.max() < pd.Timestamp(f"{d:%Y-%m-%d} 15:59", tz=NY):   # 다운로드가 중간에 끊긴 파일 (2026-09-28 첫 다운로드가 10:47에서 끊겼음)
        print(f"{d:%Y-%m-%d} 틱 파일이 {q.index.max():%H:%M}에서 끊김 → 지우고 다음 실행 때 다시 받음")
        note(f"{d:%Y-%m-%d} 틱 파일 끊김({q.index.max():%H:%M}) → 삭제, 다시 받을 예정")
        f.unlink(missing_ok=True)
        for cc in [c for c in log.columns if "초(−90%)" in c or c.startswith("손절 ") and c.endswith("초")]:
            log.loc[i, cc] = np.nan
        continue
    key = (f"{d:%Y-%m-%d}", k)
    if key not in PATHS:
        M, bid, ask, cp, K, F = _prep(load_day(d), d)
        c_ = M.columns[(K == k) & (cp == "C")][0]; stop_cols(M, bid, ask, F, d, k, c_, float(r0["정산"]))
    r = PATHS[key]; t0 = pd.Timestamp(f"{d:%Y-%m-%d} 10:00:00", tz=NY)
    sgrid = to_sec(q, d); sgrid.to_parquet(TICKD / f"{d:%Y-%m-%d}_{int(k)}_sec.parquet")
    for sec in DELAYS:
        p0 = tick_up(round(ask_at(q, t0 + pd.Timedelta(seconds=sec)), 4))
        pnl, stopped, ts = sec_stop(sgrid, d, p0, sec, float(r0["정산"]))
        log.loc[i, f"매도호가 {sec}초"] = p0
        log.loc[i, f"손익$ {sec}초(−90%)"] = round(pnl, 1)
        log.loc[i, f"손절 {sec}초"] = f"{ts:%H:%M:%S} @ {sgrid.bid.asof(ts + pd.Timedelta(seconds=1)):.2f}" if stopped else "없음"
    print(f"{d:%Y-%m-%d} 5초 {log.loc[i, '매도호가 5초']} → ${log.loc[i, '손익$ 5초(−90%)']:+,.0f}, 10초 {log.loc[i, '매도호가 10초']} → ${log.loc[i, col10]:+,.0f}")
    note(f"{d:%Y-%m-%d} 매도호가 5초·10초 반영: 5초 ${log.loc[i, '손익$ 5초(−90%)']:+,.0f}, 10초 ${log.loc[i, col10]:+,.0f}")
if len(todo_t):
    save_log(log)
