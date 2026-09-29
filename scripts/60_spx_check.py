"""
옵션 풋콜패리티로 구한 SPX(선도가격 F)가 실제 SPX와 맞는지 검증.
  A. 장 마감: F 15:59:00 vs 실제 SPX 종가 (인베스팅 일봉, 1,093일) — 차이는 마지막 1분 움직임 포함
  B. 장 시작: F 09:31:00 vs 실제 SPX 시가
  C. 조건 1 판단 일치: F(10:00:00) > F(09:31:00) vs ES 선물(09:59 봉 종가 > 09:30 봉 종가), 1,093일
  D. 실제 SPX 1분 (야후 ^GSPC 차트 JSON, 최근 5일만 제공): 분마다 F vs 실제 SPX
결과: output/spx_check/*.csv, X1_spx_check.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV, DATA, OPT_DIR
from spx0dte.core import NY, load_spx_ohlc, load_day
from spx0dte.straddle import _prep

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
R = OUT / "spx_check"; R.mkdir(exist_ok=True)

FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
# 앞으로 기록 날짜(9/24~)도 F 계산
extra = {}
for f in sorted(OPT_DIR.glob("*.dbn.zst")):
    d = pd.Timestamp(f.name[:10])
    if d > FP.index.max():
        M, bid, ask, cp, K, F = _prep(load_day(d), d)
        extra[d] = pd.Series(F.ffill().values, index=F.index.strftime("%H:%M"))
if extra:
    FP = pd.concat([FP, pd.DataFrame(extra).T]).sort_index()
days = FP.index.intersection(px.index)

# A·B
A = pd.DataFrame({"F 15:59": FP.loc[days, "15:59"], "SPX 종가": px.close[days], "F 09:31": FP.loc[days, "09:31"], "SPX 시가": px.open[days]})
A["종가 차이"] = A["F 15:59"] - A["SPX 종가"]; A["시가 차이"] = A["F 09:31"] - A["SPX 시가"]
A.to_csv(R / "A_close_open.csv", encoding="utf-8-sig")
q = lambda s: f"평균 {s.mean():+.2f}, 중앙값 {s.median():+.2f}, 절댓값 중앙값 {s.abs().median():.2f}, 90% {s.abs().quantile(.9):.2f}, 최대 {s.abs().max():.2f} pt"
print(f"A. 마감: F 15:59:00 − SPX 종가 ({A['종가 차이'].notna().sum()}일): {q(A['종가 차이'])}")
print(f"B. 시작: F 09:31:00 − SPX 시가 ({A['시가 차이'].notna().sum()}일): {q(A['시가 차이'])}")

# C. ES와 조건 1 판단 일치
es = pd.read_parquet(DATA / "es_1m.parquet"); es = es[es.symbol == "ES.v.0"].sort_index(); es.index = es.index.tz_convert(NY)
def bar(hh, mm):
    s = es[(es.index.hour == hh) & (es.index.minute == mm)]
    s = s[~s.index.normalize().duplicated()]
    return pd.DataFrame({"c": s.close.values, "id": s.instrument_id.values}, index=s.index.tz_localize(None).normalize())
e930, e959 = bar(9, 30), bar(9, 59)
C = pd.DataFrame({"F 09:31": FP.loc[days, "09:31"], "F 10:00": FP.loc[days, "10:00"]}).join(e930.add_prefix("es930_")).join(e959.add_prefix("es959_")).dropna()
C = C[C.es930_id == C.es959_id]
C["F 30분 %"] = (C["F 10:00"] / C["F 09:31"] - 1) * 100; C["ES 30분 %"] = (C.es959_c / C.es930_c - 1) * 100
C["F 상승"] = C["F 30분 %"] > 0; C["ES 상승"] = C["ES 30분 %"] > 0
agree = (C["F 상승"] == C["ES 상승"]).mean()
dis = C[C["F 상승"] != C["ES 상승"]]
print(f"C. 조건 1 (09:31 → 10:00 상승?) F vs ES 판단 일치 {agree:.1%} ({len(C)}일), 수익률 상관 {C['F 30분 %'].corr(C['ES 30분 %']):.4f}")
print(f"   불일치 {len(dis)}일: F 30분 변화 절댓값 중앙값 {dis['F 30분 %'].abs().median():.3f}% (전체 {C['F 30분 %'].abs().median():.3f}%) — 거의 보합인 날")
C.to_csv(R / "C_condition1_vs_es.csv", encoding="utf-8-sig")

# D. 실제 SPX 1분 (야후, 최근 7일)
D = None
try:
    import json, urllib.request                                             # 야후 차트 JSON 직접 (yfinance는 요청 제한에 자주 걸림)
    jf = R / "yahoo_gspc_1m.json"
    if not jf.exists() or pd.Timestamp.now() - pd.Timestamp(jf.stat().st_mtime, unit="s") > pd.Timedelta(hours=12):
        req = urllib.request.Request("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?interval=1m&range=5d", headers={"User-Agent": "Mozilla/5.0"})
        jf.write_bytes(urllib.request.urlopen(req, timeout=30).read())
    r0 = json.loads(jf.read_text())["chart"]["result"][0]
    y = pd.DataFrame({"Close": r0["indicators"]["quote"][0]["close"]}, index=pd.to_datetime(r0["timestamp"], unit="s", utc=True).tz_convert(NY)).dropna()
    rows = []
    for d in sorted(set(y.index.normalize().tz_localize(None)) & set(FP.index)):
        yd = y[y.index.normalize().tz_localize(None) == d]
        for hm in FP.columns:
            t = pd.Timestamp(f"{d:%Y-%m-%d} {hm}", tz=NY) - pd.Timedelta(minutes=1)          # F HH:MM:00 = 직전 1분봉 종가 시점
            if t in yd.index and np.isfinite(FP.at[d, hm]):
                rows.append({"date": d, "시각": hm, "F": FP.at[d, hm], "SPX": float(yd.at[t, "Close"])})
    D = pd.DataFrame(rows)
    if len(D):
        D["차이"] = D.F - D.SPX
        D.to_csv(R / "D_minute_vs_spx.csv", index=False, encoding="utf-8-sig")
        g = D.groupby("date")["차이"]
        print(f"D. 실제 SPX 1분 비교 {D.date.nunique()}일 {len(D)}분: " + q(D["차이"]))
        print(g.agg(["mean", "median", lambda s: s.abs().max()]).round(2).rename(columns={"<lambda_0>": "최대|차이|"}).to_string())
        for d in D.date.unique():
            x = D[D.date == d].set_index("시각")
            if "09:31" in x.index and "10:00" in x.index:
                print(f"   {pd.Timestamp(d):%Y-%m-%d} 조건 1: F {x.at['10:00', 'F'] - x.at['09:31', 'F']:+.2f} / 실제 SPX {x.at['10:00', 'SPX'] - x.at['09:31', 'SPX']:+.2f}")
    else:
        print("D. 겹치는 날 없음 (야후 7일 범위 밖)")
except Exception as ex:
    print("D. 야후 1분 실패:", str(ex)[:100])

fig, axs = plt.subplots(1, 3, figsize=(18, 4.6))
axs[0].hist(A["종가 차이"].clip(-15, 15), bins=60, color="#3D5A80"); axs[0].set_title("A. F 15:59:00 - 실제 SPX 종가 (pt)")
axs[1].scatter(C["ES 30분 %"], C["F 30분 %"], s=6, alpha=.5, color="#B23A48"); axs[1].axhline(0, color="k", lw=.5); axs[1].axvline(0, color="k", lw=.5)
axs[1].set_xlabel("ES 09:30→09:59 %"); axs[1].set_ylabel("F 09:31→10:00 %"); axs[1].set_title(f"C. 조건 1 판단 일치 {agree:.1%}")
if D is not None and len(D):
    for d in D.date.unique():
        x = D[D.date == d]
        axs[2].plot(range(len(x)), x["차이"].values, lw=.8, label=f"{pd.Timestamp(d):%m-%d}")
    axs[2].axhline(0, color="k", lw=.5); axs[2].legend(fontsize=8); axs[2].set_title("D. 분마다 F - 실제 SPX (pt)"); axs[2].set_xlabel("09:31부터 분")
fig.tight_layout(); fig.savefig(R / "X1_spx_check.png", dpi=110)

# A·B 보정: 비교 시점 차이(마지막 1분 / 첫 1분)를 ES 1분봉 움직임으로 빼고 남는 오차
def esbar(hh, mm):
    s = es[(es.index.hour == hh) & (es.index.minute == mm)]
    s = s[~s.index.normalize().duplicated()]
    return pd.DataFrame({"o": s.open.values, "c": s.close.values}, index=s.index.tz_localize(None).normalize())
b1559, b930 = esbar(15, 59), esbar(9, 30)
AA = A.join(b1559.add_prefix("e1559_")).join(b930.add_prefix("e930_"))
ratio = AA["SPX 종가"] / AA.e1559_c
AA["종가 잔차"] = AA["F 15:59"] - (AA["SPX 종가"] - (AA.e1559_c - AA.e1559_o) * ratio)          # F(15:59:00) vs 종가 − 마지막 1분 ES 변화
AA["시가 잔차"] = AA["F 09:31"] - (AA["SPX 시가"] + (AA.e930_c - AA.e930_o) * ratio)              # F(09:31:00) vs 시가 + 첫 1분 ES 변화
print(f"A'. 마지막 1분 ES 움직임 보정 후 F − SPX 종가: {q(AA['종가 잔차'].dropna())}")
print(f"B'. 첫 1분 ES 움직임 보정 후 F − SPX 시가: {q(AA['시가 잔차'].dropna())}")
AA.to_csv(R / "A_close_open.csv", encoding="utf-8-sig")
