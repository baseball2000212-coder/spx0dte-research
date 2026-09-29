"""
1초(틱) 데이터로 현재 규칙의 체결을 다시 보기 (산 콜 1종목, 매수한 날만).
  2023-03-28 ~ : cmbp-1 (호가 변화 전부)
    ① 진입: 10:00:00 중간가(L)로 지정가, 주문 도착 = 10:00:00 + 지연(1초·5초). 대기 W초(10·30·60) 안에 매도호가가 L 이하로 오면 L에 체결,
            아니면 W초 뒤 매도호가로 추격 매수. (실제론 우리 호가가 최우선이 되어 매도호가가 안 내려와도 체결될 수 있음 → 보수적)
    ② 손절: 체결 후 중간가가 매수가의 10% 이하가 되는 첫 순간(초 단위) → 1초 뒤 매수호가로 매도. 1분 스냅샷 규칙과 비교.
  2022-05 ~ 2023-03 : trades (체결가만)
    ③ 10:00:00~10:01:00 사이 L 이하 체결이 있었나, −90% 선 아래 첫 체결 시각 vs 1분 규칙 손절 시각.
결과: output/ticks/tick_days.csv, tick_summary.csv, K1_ticks.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, databento as db
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import DATA, OUT
from spx0dte.core import NY
from spx0dte.exits import FEE, EXERCISE

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
TICK = DATA / "ticks"; R = OUT / "ticks"; R.mkdir(exist_ok=True)
STOP = 0.9; LATS = (1, 5); WAITS = (10, 30, 60)


def load_quotes(f):
    df = db.DBNStore.from_file(str(f)).to_df()
    df.index = df.index.tz_convert(NY)
    q = df[["bid_px_00", "ask_px_00"]].astype(float).rename(columns={"bid_px_00": "bid", "ask_px_00": "ask"})
    q = q[(q.ask > 0) & (q.ask >= q.bid)]
    return q[~q.index.duplicated(keep="last")]


def at(q, t):
    """t 시점 상태 (t 이전 마지막 호가)."""
    i = q.index.searchsorted(t, side="right") - 1
    return q.iloc[i] if i >= 0 else None


def cmbp_day(args):
    f, d, k, mid1m, pay, stop1m_min, pnl1m = args
    q = load_quotes(f)
    t0 = pd.Timestamp(f"{d:%Y-%m-%d} 10:00:00", tz=NY)
    s0 = at(q, t0)
    if s0 is None:
        return {"date": d, "오류": "10:00 호가 없음"}
    L = round((s0.bid + s0.ask) / 2, 2)
    out = {"date": d, "구분": "호가(cmbp-1)", "1분 중간가": mid1m, "틱 중간가 10:00:00": L, "틱 매도호가 10:00:00": s0.ask,
           "호가 변경 수(10~16시)": int(((q.index >= t0) & (q.index < t0 + pd.Timedelta(hours=6))).sum())}
    # ① 진입
    for lat in LATS:
        for w in WAITS:
            a, b = t0 + pd.Timedelta(seconds=lat), t0 + pd.Timedelta(seconds=lat + w)
            st = at(q, a); win = q[(q.index > a) & (q.index <= b)].ask
            touched = (st is not None and st.ask <= L) or (len(win) and (win <= L + 1e-9).any())
            if touched:
                px, how = L, "중간가"
            else:
                e = at(q, b); px, how = (e.ask if e is not None else np.nan), "추격"
            out[f"진입가 지연{lat}s 대기{w}s"] = px; out[f"체결 지연{lat}s 대기{w}s"] = how
    # ② 손절 (진입가 = 1분 중간가 기준, 1분 규칙과 같은 조건에서 비교)
    after = q[q.index > t0]; mid = (after.bid + after.ask) / 2
    hit = mid[mid <= mid1m * (1 - STOP) + 1e-6]
    if len(hit):
        th = hit.index[0]; s1 = at(q, th + pd.Timedelta(seconds=1))
        out.update({"틱 손절 시각": th.strftime("%H:%M:%S"), "틱 손절 매도가": s1.bid,
                    "틱 손익$": (s1.bid - mid1m - 2 * FEE) * 100})
    else:
        out.update({"틱 손절 시각": "", "틱 손절 매도가": np.nan, "틱 손익$": (pay - mid1m - FEE - (EXERCISE if pay > 0 else 0)) * 100})
    out.update({"1분 손절 시각": stop1m_min, "1분 손익$": pnl1m, "정산": pay})
    return out


def trades_day(args):
    f, d, k, mid1m, pay, stop1m_min, pnl1m = args
    df = db.DBNStore.from_file(str(f)).to_df()
    df.index = df.index.tz_convert(NY)
    t0 = pd.Timestamp(f"{d:%Y-%m-%d} 10:00:00", tz=NY)
    tr = df["price"].astype(float)
    w = tr[(tr.index >= t0) & (tr.index < t0 + pd.Timedelta(minutes=1))]
    after = tr[tr.index > t0]; low = after[after <= mid1m * (1 - STOP) + 1e-6]
    return {"date": d, "구분": "체결(trades)", "1분 중간가": mid1m, "10:00~10:01 체결 수": len(w),
            "10:00~10:01 중간가 이하 체결": int((w <= mid1m + 1e-9).sum()), "10:00~10:01 최저 체결가": w.min() if len(w) else np.nan,
            "체결 −90% 아래 첫 시각": low.index[0].strftime("%H:%M:%S") if len(low) else "", "1분 손절 시각": stop1m_min,
            "1분 손익$": pnl1m, "정산": pay, "하루 체결 수": int((tr.index >= t0).sum())}


if __name__ == "__main__":
    T = pd.read_csv(OUT / "rule_v2" / "trades.csv", index_col=0, parse_dates=True)
    jobs_c, jobs_t = [], []
    for d, r in T.iterrows():
        st = str(r["손절 시각"])[11:16] if isinstance(r["손절 시각"], str) and r["손절 시각"] else ""
        args = (None, d, r.K, r.mid, r.pay, st, r.pnl * 100)
        fc, ft = TICK / "cmbp-1" / f"{d:%Y-%m-%d}.dbn.zst", TICK / "trades" / f"{d:%Y-%m-%d}.dbn.zst"
        if fc.exists():
            jobs_c.append((fc,) + args[1:])
        elif ft.exists():
            jobs_t.append((ft,) + args[1:])
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        C = pd.DataFrame(list(ex.map(cmbp_day, jobs_c, chunksize=2))).set_index("date").sort_index()
        Tr = pd.DataFrame(list(ex.map(trades_day, jobs_t, chunksize=4))).set_index("date").sort_index() if jobs_t else pd.DataFrame()
    pd.concat([C, Tr]).sort_index().to_csv(R / "tick_days.csv", encoding="utf-8-sig")
    print(f"호가 {len(C)}일, 체결 {len(Tr)}일")

    rows = []
    # 1분 중간가 = 틱 10:00:00 중간가?
    dm = (C["틱 중간가 10:00:00"] - C["1분 중간가"]).abs()
    print(f"10:00:00 중간가 1분 vs 틱 일치: 차이 0.05 이하 {(dm <= 0.05).mean():.0%}, 최대 {dm.max():.2f}")
    base = C["1분 손익$"].sum()
    rows.append({"경우": "기준: 1분 데이터 (10:00:00 중간가 체결, 1분 손절)", "총손익$": base, "차이$": 0.0})
    for lat in LATS:
        for w in WAITS:
            px = C[f"진입가 지연{lat}s 대기{w}s"]; how = C[f"체결 지연{lat}s 대기{w}s"]
            adj = C["1분 손익$"] - (px - C["1분 중간가"]) * 100          # 진입가 차이만큼 (손절 판정은 그대로)
            rows.append({"경우": f"진입 지연 {lat}초, 중간가 대기 {w}초 → 안 되면 매도호가 추격", "총손익$": adj.sum(), "차이$": adj.sum() - base,
                         "중간가 체결 비율%": (how == "중간가").mean() * 100, "추격 시 평균 추가 비용$": ((px - C["1분 중간가"]) * 100)[how == "추격"].mean()})
    imm = C["1분 손익$"] - (C["틱 매도호가 10:00:00"] - C["1분 중간가"]) * 100
    rows.append({"경우": "10:00:00 매도호가로 즉시 매수", "총손익$": imm.sum(), "차이$": imm.sum() - base, "중간가 체결 비율%": 0.0,
                 "추격 시 평균 추가 비용$": ((C["틱 매도호가 10:00:00"] - C["1분 중간가"]) * 100).mean()})
    rows.append({"경우": "손절을 틱(초)으로 판정, 1초 뒤 매수호가로 매도", "총손익$": C["틱 손익$"].sum(), "차이$": C["틱 손익$"].sum() - base,
                 "틱 손절 수": int((C["틱 손절 시각"] != "").sum()), "1분 손절 수": int((C["1분 손절 시각"] != "").sum())})
    S = pd.DataFrame(rows); S.to_csv(R / "tick_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); print(S.round(1).to_string(index=False))

    both = C[(C["틱 손절 시각"] != "") | (C["1분 손절 시각"] != "")]
    only_tick = both[(both["틱 손절 시각"] != "") & (both["1분 손절 시각"] == "")]
    print(f"\n틱에서만 손절 (1분 스냅샷은 놓침): {len(only_tick)}일, 그날들 틱 손익 합 ${only_tick['틱 손익$'].sum():,.0f} vs 1분 ${only_tick['1분 손익$'].sum():,.0f}")
    print(only_tick[["1분 중간가", "정산", "틱 손절 시각", "틱 손절 매도가", "틱 손익$", "1분 손익$"]].round(2).to_string())
    diffc = both[(both["틱 손절 시각"] != "") & (both["1분 손절 시각"] != "")]
    print(f"둘 다 손절: {len(diffc)}일, 틱 손익 합 ${diffc['틱 손익$'].sum():,.0f} vs 1분 ${diffc['1분 손익$'].sum():,.0f}")
    if len(Tr):
        print(f"\n체결 데이터 {len(Tr)}일 (2022-05 ~ 2023-03): 10:00~10:01 체결 있음 {(Tr['10:00~10:01 체결 수'] > 0).mean():.0%}, "
              f"그중 중간가 이하 체결 있음 {(Tr.loc[Tr['10:00~10:01 체결 수'] > 0, '10:00~10:01 중간가 이하 체결'] > 0).mean():.0%}, "
              f"체결 −90% 아래 있었지만 1분 손절 없음 {((Tr['체결 −90% 아래 첫 시각'] != '') & (Tr['1분 손절 시각'] == '')).sum()}일")

    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    e = S.iloc[1:2 + len(LATS) * len(WAITS)]
    axs[0].barh(e["경우"].str.replace("진입 ", "").str.replace(" → 안 되면 매도호가 추격", ""), e["차이$"] / 1000, color=["#B23A48"] * (len(e) - 1) + ["#0F6E5A"])
    axs[0].set_title("진입 체결 현실화: 기준 대비 총손익 차이 (천 달러)"); axs[0].axvline(0, color="k", lw=.6)
    for i, (v, pct) in enumerate(zip(e["차이$"] / 1000, e["중간가 체결 비율%"])):
        axs[0].text(v, i, f"  중간가 체결 {pct:.0f}%" if pct > 0 else "  바로 매도호가", va="center", fontsize=8)
    cum1 = C["1분 손익$"].cumsum() / 1000; cumt = C["틱 손익$"].cumsum() / 1000
    axs[1].plot(cum1.index, cum1, color="#9AA3AD", label=f"1분 손절 {cum1.iloc[-1]:+.1f}k"); axs[1].plot(cumt.index, cumt, color="#0F6E5A", label=f"틱 손절 {cumt.iloc[-1]:+.1f}k")
    axs[1].set_title("손절 판정 1분 vs 틱 (2023-03-28 ~, 누적 천 달러)"); axs[1].legend(); axs[1].grid(alpha=.3)
    fig.suptitle("K1 틱 데이터로 본 체결 현실성 (매수한 날의 산 콜만)"); fig.tight_layout(); fig.savefig(R / "K1_ticks.png", dpi=110)
