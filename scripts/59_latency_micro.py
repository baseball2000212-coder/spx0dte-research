"""
틱(1초) 데이터로 체결 지연·추가 비용과 10시 전후 미시구조 (2023-03-28 ~ 매수한 319일, 산 콜 1종목).
  A. 지연 L초(0~60) 뒤 그때 매도호가로 매수 × 추가 비용(0 / +1틱 / +5% / +10%) → 총손익 (손절 판정·정산은 1분 기준 그대로, 진입가만 바꿈)
     + 새 엔진(ChasePlan: 매초 현재 매도호가, 10:00 매도호가 +20% 상한, 60초 제한)
  B. 10:00 전후 −60~+120초: 스프레드(매도−매수, 중간가 대비 %)와 매도호가 변화 (10:00:00 대비)
  C. −90% 손절 체결: 판정 순간 중간가 대비 1초 뒤 매수호가 (손절할 때 잃는 스프레드)
  D. 만기 내가격 콜: 15:59:00 매수호가로 팔기 vs 정산 (정산 수수료 0.025 포함)
결과: output/ticks/L1_latency.csv, L2_micro.csv, L1_latency.png, L2_micro.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import DATA, OUT
from spx0dte.exits import FEE, EXERCISE
from live.engine import ChasePlan, tick_up

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
R = OUT / "ticks"; SEC = DATA / "ticks" / "sec"; NY = "America/New_York"; ODD = pd.Timestamp("2025-04-09")
LATS = (0, 1, 2, 3, 5, 10, 30, 60); OFFS = (-60, -30, -10, -5, -1, 0, 1, 2, 3, 5, 10, 20, 30, 60, 120)
SLIPS = (("그대로", lambda p: p), ("+1틱", lambda p: p + (0.05 if p < 3 else 0.10)), ("+5%", lambda p: tick_up(p * 1.05)), ("+10%", lambda p: tick_up(p * 1.10)))


def one(args):
    d, mid1m, pnl1m, pay, stopped, stop_px = args
    s = pd.read_parquet(SEC / f"{d:%Y-%m-%d}.parquet")
    t0 = pd.Timestamp(f"{d:%Y-%m-%d} 10:00:00", tz=NY)
    at = lambda k: s.loc[t0 + pd.Timedelta(seconds=k)]
    ask0 = float(at(0).ask)
    out = {"date": d}
    for L in LATS:
        out[f"ask_{L}"] = float(at(L).ask)
    for k in OFFS:
        q = at(k); out[f"spr_{k}"] = float(q.ask - q.bid); out[f"sprpct_{k}"] = float((q.ask - q.bid) / ((q.ask + q.bid) / 2) * 100)
        out[f"askchg_{k}"] = float(q.ask / ask0 - 1) * 100
    plan = ChasePlan(); fill = np.nan
    for k in range(1, plan.max_wait_sec + 1):
        p = plan.price(float(at(k).ask), ask0, k)
        if p is None:
            break
        if np.isfinite(p):
            fill = p; break
    out["engine_fill"] = fill
    q = s.loc[pd.Timestamp(f"{d:%Y-%m-%d} 15:59:00", tz=NY)]
    out["bid_1559"] = float(q.bid)
    return out


if __name__ == "__main__":
    C = pd.read_csv(R / "tick_days.csv", index_col=0, parse_dates=True); C = C[C["구분"] == "호가(cmbp-1)"]
    T = pd.read_csv(OUT / "rule_v2" / "trades.csv", index_col=0, parse_dates=True)
    jobs = [(d, r["1분 중간가"], r["1분 손익$"], r["정산"], r["1분 손절 시각"] if isinstance(r["1분 손절 시각"], str) else "", np.nan) for d, r in C.iterrows()]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        M = pd.DataFrame(list(ex.map(one, jobs, chunksize=8))).set_index("date").sort_index()
    base = C["1분 손익$"]; mid = C["1분 중간가"]

    # A. 지연 × 추가 비용
    rows = []
    for L in LATS:
        for nm, f in SLIPS:
            px = M[f"ask_{L}"].map(lambda p: f(tick_up(p)))
            pnl = base - (px - mid) * 100
            rows.append({"지연(초)": L, "추가 비용": nm, "총손익$": pnl.sum(), "4/9 빼고$": pnl.drop(ODD, errors="ignore").sum(),
                         "기준 대비$": pnl.sum() - base.sum(), "중간가 대비 건당$": ((px - mid) * 100).mean()})
    ef = M["engine_fill"]; pe = (base - (ef - mid) * 100).where(ef.notna(), 0.0)
    rows.append({"지연(초)": "1~60", "추가 비용": "새 엔진 (매초 매도호가, +20% 상한)", "총손익$": pe.sum(), "4/9 빼고$": pe.drop(ODD, errors="ignore").sum(),
                 "기준 대비$": pe.sum() - base.sum(), "중간가 대비 건당$": ((ef - mid) * 100).mean(), "못 산 날": int(ef.isna().sum())})
    A = pd.DataFrame(rows); A.to_csv(R / "L1_latency.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
    print(f"기준 (1분 중간가, 319일) ${base.sum():,.0f}")
    print(A.round(0).to_string(index=False))

    # B. 10시 전후
    B = pd.DataFrame({k: {"스프레드$ 중앙값": M[f"spr_{k}"].median() * 100, "스프레드 % 중앙값": M[f"sprpct_{k}"].median(),
                          "매도호가 변화% 평균": M[f"askchg_{k}"].mean(), "매도호가 변화% 중앙값": M[f"askchg_{k}"].median(),
                          "5% 넘게 오른 날%": (M[f"askchg_{k}"] > 5).mean() * 100} for k in OFFS}).T
    B.index.name = "10:00 기준 초"; B.to_csv(R / "L2_micro.csv", encoding="utf-8-sig")
    print("\nB. 10:00 전후 (산 콜, 319일)\n", B.round(2).to_string())

    # C. 손절 체결 비용
    st = C[C["1분 손절 시각"].fillna("") != ""]
    tk = C[C["틱 손절 시각"].fillna("") != ""]
    slip = (tk["1분 중간가"] * 0.1 - tk["틱 손절 매도가"]) * 100
    print(f"\nC. 손절 {len(tk)}건: 판정선(매수가×10%) 대비 1초 뒤 매수호가가 건당 ${slip.mean():.1f} 낮음 (중앙값 ${slip.median():.1f}), 합 ${slip.sum():,.0f}")

    # D. 만기 내가격: 15:59:00 매수호가 매도 vs 정산
    itm = C[(C["정산"] > 0) & (C["1분 손절 시각"].fillna("") == "")]
    sell = (M.loc[itm.index, "bid_1559"] - FEE) - (itm["정산"] - EXERCISE)
    print(f"D. 만기 내가격 {len(itm)}건: 15:59:00 매수호가로 팔면 정산보다 건당 ${sell.mean() * 100:+.1f} (중앙값 ${sell.median() * 100:+.1f}), 합 ${sell.sum() * 100:+,.0f}"
          f" — 단 15:59→16:00 가격 변화 포함")

    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    for nm, _ in SLIPS:
        a = A[A["추가 비용"] == nm]
        axs[0].plot([str(x) for x in a["지연(초)"]], a["총손익$"] / 1000, "o-", label=f"매도호가 {nm}")
    axs[0].axhline(base.sum() / 1000, color="k", ls="--", lw=1, label="기준 (1분 중간가)")
    axs[0].axhline(pe.sum() / 1000, color="#0F6E5A", ls=":", lw=1.5, label=f"새 엔진 {pe.sum() / 1000:.1f}k")
    axs[0].set_xlabel("주문 지연 (초)"); axs[0].set_ylabel("총손익 (천 달러)"); axs[0].set_title("A. 지연 × 추가 비용 (319일)"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    x = [str(k) for k in OFFS]
    axs[1].plot(x, B["매도호가 변화% 평균"], "o-", color="#B23A48", label="매도호가 변화 % (평균)")
    axs[1].plot(x, B["매도호가 변화% 중앙값"], "o--", color="#B23A48", alpha=.5, label="매도호가 변화 % (중앙값)")
    ax2 = axs[1].twinx(); ax2.plot(x, B["스프레드 % 중앙값"], "s-", color="#3D5A80", label="스프레드 % (중앙값)"); ax2.set_ylabel("스프레드 (중간가 대비 %)")
    axs[1].axvline(x.index("0"), color="grey", lw=.6); axs[1].set_xlabel("10:00:00 기준 초"); axs[1].set_ylabel("매도호가 변화 %")
    axs[1].set_title("B. 10:00 전후 산 콜의 매도호가·스프레드"); axs[1].legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(R / "L1_latency.png", dpi=110)
