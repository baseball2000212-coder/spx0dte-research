"""
세 번째 외부 검토 대응 (2026-09-27).
  ① 화·목 사전 등록 기준의 검정력: 표본 내 추정 엣지(100%·70%·50%)가 진짜일 때 106건에서 통과 확률 (t ≥ 2 / t ≥ 1 버전)
  ② 수요일 분해: FOMC·CPI·고용 날 제외 전후 요일별 건당 (표본 내·표본외)
  ③ 국면 표(조건부 − 무조건)를 표본외 2020-01 ~ 2022-05로
  ④ XSP 표: 수수료 편도 $2.5/$8 × 정산 $0/$8 × 체결 중간가/매도호가
결과: output/critique3/*.csv, critique3.json
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, warnings
from types import SimpleNamespace
import numpy as np, pandas as pd
from spx0dte.config import OUT, SPX_CSV, DATA
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_rule
from spx0dte.exits import FEE, EXERCISE
from spx0dte.events import EVENTS

warnings.filterwarnings("ignore")
D = OUT / "critique3"; D.mkdir(exist_ok=True)
ODD = pd.Timestamp("2025-04-09"); DOW = "월화수목금"; RNG = np.random.default_rng(3); NSIM = 20_000; ACCOUNT = 100_000
EV = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in EVENTS.values()])))
TU = np.sqrt(390 * 252 / 525600) * 100


def tstat(v):
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


if __name__ == "__main__":
    FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
    X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
    L = pd.read_pickle(OUT / "legs10_full.pkl")
    L = L[(L.leg == "C") & (L.offset == 0) & (L.entry == "10:00")].set_index("date").sort_index()
    rows = [SimpleNamespace(date=d, mid=r.mid, ask=r.ask, pay=r.pay, mid_path=r.mid_path, bid_path=r.bid_path, ask_path=r.ask_path) for d, r in L.iterrows()]
    sig = X.signal.reindex(L.index).fillna(False).astype(bool)
    P = pd.Series({r.date: trade_rule(r, stop=None)["pnl"] * 100 for r in rows}).sort_index()     # 무손절, 전체 날
    R0 = P[sig]
    out = {}

    # ① 검정력 시뮬레이션 (무손절, 부트스트랩: 분포 모양 유지, 평균만 이동)
    tt = R0[R0.index.dayofweek.isin([1, 3])]; mwf = R0[~R0.index.dayofweek.isin([1, 3])]
    mwf_x = mwf.drop(ODD, errors="ignore")
    N_TT = 106; N_MWF = int(round(N_TT * len(mwf) / len(tt)))
    base_tt, base_m = tt.values - tt.mean(), mwf_x.values - mwf_x.mean()
    pw = []
    for frac in (1.0, 0.7, 0.5, 0.0):
        mu = tt.mean() * frac
        s_tt = base_tt[RNG.integers(0, len(base_tt), (NSIM, N_TT))] + mu
        s_m = base_m[RNG.integers(0, len(base_m), (NSIM, N_MWF))] + mwf_x.mean()
        t = s_tt.mean(1) / (s_tt.std(1, ddof=1) / np.sqrt(N_TT)); diff = s_tt.mean(1) > s_m.mean(1)
        pw.append({"가정 화·목 엣지": f"{frac:.0%} (${mu:,.0f})", "통과 확률 t≥2": (t >= 2).mean() * 100, "통과 확률 t≥2 & 화·목>월수금": ((t >= 2) & diff).mean() * 100,
                   "통과 확률 t≥1 & 화·목>월수금": ((t >= 1) & diff).mean() * 100, "t 중앙값": np.median(t)})
    PW = pd.DataFrame(pw); PW.to_csv(D / "Q1_power_tt.csv", index=False, encoding="utf-8-sig")
    out["power_tt"] = {"n_tt": N_TT, "n_mwf": N_MWF, "tt_mean": tt.mean(), "tt_sd": tt.std(ddof=1), "mwf_ex_mean": mwf_x.mean(), "rows": PW.to_dict("records")}
    print(f"① 화·목 {len(tt)}건 건당 ${tt.mean():,.0f} SD ${tt.std(ddof=1):,.0f} / 월수금(4/9 제외) ${mwf_x.mean():,.0f}, 가정 월수금 {N_MWF}건\n", PW.round(1).to_string(index=False))

    # ② 요일 × 이벤트 제외 (무손절)
    O = pd.read_csv(OUT / "oos" / "days.csv", index_col=0, parse_dates=True); O = O[O["행사가"].notna() & O["손익$"].notna()]
    Q = O["손익$"]; osig = O["신호"] == True
    wk = []
    for lab, allv, s in (("2022-05~2026-09", P, sig), ("표본외 2020-01~2022-05", Q, osig)):
        for who, m in (("규칙", s), ("매일 매수", pd.Series(True, index=allv.index))):
            v = allv[m.reindex(allv.index).fillna(False).astype(bool)]
            for ex in ("전체", "이벤트일 제외", "이벤트일 제외 · 4/9 제외"):
                vv = v if ex == "전체" else v[~v.index.isin(EV)]
                if "4/9" in ex:
                    vv = vv.drop(ODD, errors="ignore")
                for k in range(5):
                    x = vv[vv.index.dayofweek == k]
                    if len(x):
                        wk.append({"구간": lab, "대상": who, "제외": ex, "요일": DOW[k], "건수": len(x), "건당$": x.mean(), "t값": tstat(x),
                                   "이벤트일 수": int(v[(v.index.dayofweek == k)].index.isin(EV).sum())})
    WK = pd.DataFrame(wk); WK.to_csv(D / "Q2_weekday_events.csv", index=False, encoding="utf-8-sig")
    out["weekday_events"] = WK.to_dict("records")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print("\n② 요일 × 이벤트 제외 (규칙)\n", WK[WK["대상"] == "규칙"].pivot_table(index=["구간", "요일"], columns="제외", values="건당$").round(0).to_string())
    print(WK[WK["대상"] == "규칙"].pivot_table(index=["구간", "요일"], columns="제외", values="건수").to_string())
    # 이벤트 종류별 (규칙, 수요일)
    ek = []
    for lab, allv, s in (("2022-05~2026-09", P, sig), ("표본외 2020-01~2022-05", Q, osig)):
        v = allv[s.reindex(allv.index).fillna(False).astype(bool)]
        for nm, idx in EVENTS.items():
            x = v[v.index.isin(idx)]
            ek.append({"구간": lab, "이벤트": nm, "건수": len(x), "건당$": x.mean() if len(x) else np.nan, "그중 수요일": int((x.index.dayofweek == 2).sum()),
                       "수요일 건당$": x[x.index.dayofweek == 2].mean() if (x.index.dayofweek == 2).any() else np.nan})
    EK = pd.DataFrame(ek); EK.to_csv(D / "Q2_event_types.csv", index=False, encoding="utf-8-sig"); out["event_types"] = EK.to_dict("records")
    print(EK.round(0).to_string(index=False))

    # ③ 표본외 국면 (조건부 − 무조건, 경계 = 표본외 전체 날 3분위)
    F10 = pd.read_pickle(OUT / "holdout" / "oos10.pkl")                         # 50_v3_decompose.py: 표본외 10:00 ATM IV
    es = pd.read_parquet(DATA / "es_1m.parquet"); es = es[es.symbol == "ES.v.0"].sort_index(); es.index = es.index.tz_convert("America/New_York")
    es = es[(es.index < pd.Timestamp("2022-06-01", tz="America/New_York")) & (es.index.time >= pd.Timestamp("09:30").time()) & (es.index.time < pd.Timestamp("16:00").time())]
    lr = np.log(es.close).diff(); lr[es.instrument_id.ne(es.instrument_id.shift())] = np.nan
    rv = (lr.groupby(lr.index.tz_localize(None).normalize()).std() * np.sqrt(390 * 252) * 100)
    tdays = px.index
    prev_rv = pd.Series({d: rv.get(tdays[tdays < d][-1], np.nan) if (tdays < d).any() else np.nan for d in O.index})
    prev_close = px.close.shift(1).reindex(O.index)
    ZO = pd.DataFrame({"10시 IV": (F10.iv_atm * TU).reindex(O.index), "IV 전일 대비": (F10.iv_atm * TU).reindex(O.index).diff(),
                       "밤사이 갭%": (O["SPX 개장"] / prev_close - 1) * 100, "개장 30분 상승폭%": (O["SPX 10:00"] / O["SPX 개장"] - 1) * 100,
                       "전일 실현변동성": prev_rv})
    reg = []
    for col in ZO.columns:
        z = ZO[col].dropna(); edges = z.quantile([1 / 3, 2 / 3]).values
        b = pd.Series(np.digitize(z, edges), index=z.index).map({0: "낮음", 1: "중간", 2: "높음"})
        for k in ("낮음", "중간", "높음"):
            dd = b.index[b == k]; allv = Q.reindex(dd); sv = allv[osig.reindex(dd).values]
            reg.append({"피처": col, "구간": k, "범위": f"{z[b == k].min():.2f} ~ {z[b == k].max():.2f}", "매일 매수 건수": len(allv), "매일 매수 건당$": allv.mean(),
                        "규칙 건수": len(sv), "규칙 건당$": sv.mean() if len(sv) else np.nan, "조건부−무조건$": (sv.mean() - allv.mean()) if len(sv) else np.nan})
    RG = pd.DataFrame(reg); RG.to_csv(D / "Q3_regime_oos.csv", index=False, encoding="utf-8-sig"); out["regime_oos"] = RG.to_dict("records")
    print("\n③ 표본외 국면 (무손절)\n", RG.round(0).to_string(index=False))

    # ④ XSP 수수료·체결 변형 (규칙 신호일, 1·3계약, 무손절 · −90%)
    xs = []
    for stop, slab in ((None, "무손절"), (0.9, "−90%")):
        for fill in ("중간가", "매도호가"):
            tr = [(r, trade_rule(r, stop=stop, fill=fill)) for r in rows if sig.get(r.date, False)]
            for fee in (2.5, 8.0):
                for settle in (0.0, 8.0):
                    v = []
                    for r, t in tr:
                        if t["stop"]:
                            v.append((t["stop"][1] - t["cost"]) * 10 - 2 * fee)
                        else:
                            v.append((r.pay - t["cost"]) * 10 - fee - (settle if r.pay > 0 else 0))
                    v = np.array(v); row = {"손절": slab, "체결": fill, "수수료 편도$": fee, "정산$": settle, "XSP 1계약 총손익$": v.sum(), "건당$": v.mean()}
                    for n in (1, 3):
                        vv = v * n; idx = RNG.integers(0, len(vv), (10_000, len(vv))); eq = vv[idx].cumsum(1)
                        mdd = (eq - np.maximum.accumulate(eq, 1)).min(1)
                        row[f"{n}계약 중앙값%"] = np.median(mdd) / ACCOUNT * 100; row[f"{n}계약 최악5%%"] = np.percentile(mdd, 5) / ACCOUNT * 100
                    xs.append(row)
    XS = pd.DataFrame(xs); XS.to_csv(D / "Q4_xsp_fees.csv", index=False, encoding="utf-8-sig"); out["xsp_fees"] = XS.to_dict("records")
    print("\n④ XSP 수수료·체결 변형\n", XS.round(1).to_string(index=False))
    json.dump(out, open(D / "critique3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
