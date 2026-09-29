"""
10:00 거시지표 발표일과 1초 스파이크 (다음 할 일 3번, 2026-09-29). 2023-03-28 ~ 매수한 319일, 산 콜 1종목 틱.
  A. 발표 종류별 (spx0dte/releases.py): 10:00:00 대비 −10~+60초 매도호가 변화·스프레드, 1초 뒤 대비 L초 뒤 매도호가 차이($/계약)
  B. 매수 방식 비교 (총손익$, 1분 중간가 기준 손익에서 진입가만 바꿈 — 59번과 같은 방식):
     1초 즉시 / 발표일만 W초 대기 / 09:59:5x 선매수 / 새 엔진(ChasePlan) / 새 엔진 + 발표일 W초 대기
  C. 선매수 때 신호가 뒤집힐 위험: 10:00:00 판단 여유(조건1 SPX pt, 조건2 MNQ pt) vs 5·10초 동안 움직임 (1,093일)
  D. 매도호가 수량 (cmbp-1 원본 ask_sz_00): 10:00:00, 1초·10초 뒤 → 몇 계약까지 한 번에 살 수 있나
결과: output/ticks/N1_release_path.csv, N2_policy.csv, N3_flip.csv, N4_depth.csv, N1_release.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, databento as db
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import DATA, OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.releases import label_release, ORDER
from live.engine import ChasePlan, tick_up

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
R = OUT / "ticks"; TICK = DATA / "ticks"; SEC = TICK / "sec"; NY = "America/New_York"; ODD = pd.Timestamp("2025-04-09")
OFFS = (-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20, 30, 60)
WAITS = (5, 10, 20, 30, 60)
BIG = ("ISM 서비스업", "JOLTS", "미시간대")          # 1초 움직임이 평소보다 큰 발표 (A 결과 보고 정함 — 아래 표에 전부 있음)


def chase(ask_at, ask_ref, start):
    """ChasePlan을 start초부터 돌림 (상한 기준은 엔진과 같게 10:00:00 매도호가, 제한 60초도 start부터)."""
    plan = ChasePlan()
    for k in range(start, plan.max_wait_sec + start):
        p = plan.price(ask_at(k), ask_ref, k - start + 1)
        if p is None:
            return np.nan
        if np.isfinite(p):
            return p
    return np.nan


def one(d):
    s = pd.read_parquet(SEC / f"{d:%Y-%m-%d}.parquet")
    t0 = pd.Timestamp(f"{d:%Y-%m-%d} 10:00:00", tz=NY)
    q = lambda k: s.loc[t0 + pd.Timedelta(seconds=k)]
    ask_at = lambda k: float(q(k).ask)
    out = {"date": d}
    for k in OFFS:
        r = q(k); out[f"ask_{k}"] = float(r.ask); out[f"bid_{k}"] = float(r.bid)
    a0 = ask_at(0)
    out["engine_1"] = chase(ask_at, a0, 1)
    for w in WAITS:
        out[f"engine_{w}"] = chase(ask_at, a0, w)
    # D. 수량 (원본)
    df = db.DBNStore.from_file(str(TICK / "cmbp-1" / f"{d:%Y-%m-%d}.dbn.zst")).to_df()
    df.index = df.index.tz_convert(NY)
    df = df[(df.ask_px_00 > 0)][["ask_px_00", "ask_sz_00", "bid_sz_00"]]
    for k in (0, 1, 10):
        i = df.index.searchsorted(t0 + pd.Timedelta(seconds=k), side="left") - 1     # k초 순간 직전 상태 (sec 격자와 같은 정의)
        out[f"asksz_{k}"] = float(df.ask_sz_00.iloc[i]) if i >= 0 else np.nan
    return out


def flip_risk():
    """C. 1,093일: 10:00 판단 여유 vs 10:00 직전 5·10초 움직임 (SPX는 산 콜 중간가/델타 0.5로, MNQ는 1분 움직임 비율로 환산)."""
    from spx0dte.touch import load_mnq, signal_grids_ext
    from spx0dte.strategy import features_10
    FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
    X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
    m1 = X.f10 - X.f_open; m2 = X.mnq - X.mnq_top5
    mnq1 = load_mnq()[0]
    return X, m1, m2, mnq1, FP


if __name__ == "__main__":
    C = pd.read_csv(R / "tick_days.csv", index_col=0, parse_dates=True); C = C[C["구분"] == "호가(cmbp-1)"]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        M = pd.DataFrame(list(ex.map(one, list(C.index), chunksize=8))).set_index("date").sort_index()
    M["발표"] = [label_release(d) for d in M.index]
    base = C["1분 손익$"]; mid = C["1분 중간가"]
    pd.set_option("display.width", 240); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 30)

    # ---------- A ----------
    rows = []
    for g in ORDER + ("전체",):
        S = M if g == "전체" else M[M["발표"] == g]
        if not len(S):
            continue
        for k in OFFS:
            chg = (S[f"ask_{k}"] / S["ask_0"] - 1) * 100
            vs1 = (S[f"ask_{k}"] - S["ask_1"]) * 100
            midk = (S[f"ask_{k}"] + S[f"bid_{k}"]) / 2
            rows.append({"발표": g, "날 수": len(S), "초": k, "매도호가 변화% 평균": chg.mean(), "중앙값": chg.median(),
                         "|중간가 변화|% 평균": ((midk / ((S.ask_0 + S.bid_0) / 2) - 1).abs() * 100).mean(),
                         "스프레드% 중앙값": ((S[f"ask_{k}"] - S[f"bid_{k}"]) / midk * 100).median(),
                         "1초 대비 $ 평균": vs1.mean(), "1초 대비 $ t": vs1.mean() / (vs1.std() / np.sqrt(len(vs1))) if len(vs1) > 2 and vs1.std() > 0 else np.nan})
    A = pd.DataFrame(rows); A.to_csv(R / "N1_release_path.csv", index=False, encoding="utf-8-sig")
    print("A. 발표 종류별 날 수:", M["발표"].value_counts().to_dict())
    print(A[A["초"].isin((-5, 1, 10, 30, 60))].round(2).to_string(index=False))

    # ---------- B ----------
    def pnl_from(fill):
        return base - (fill - mid) * 100
    big = M["발표"].isin(BIG)
    pol = {"① 1초 뒤 매도호가 (기준)": M["ask_1"].map(tick_up)}
    for w in (10, 20, 30, 60):
        pol[f"② 발표일({'·'.join(BIG)})만 {w}초 대기"] = pd.Series(np.where(big, M[f"ask_{w}"], M["ask_1"]), M.index).map(tick_up)
    for k in (-5, -2):
        pol[f"③ 09:59:{60 + k}에 선매수 (신호 안 뒤집힌다고 가정)"] = M[f"ask_{k}"].map(tick_up)
        pol[f"③' 발표일만 1초 / 나머지 09:59:{60 + k} 선매수"] = pd.Series(np.where(big, M["ask_1"], M[f"ask_{k}"]), M.index).map(tick_up)
    pol["④ 새 엔진 (ChasePlan, 1초부터)"] = M["engine_1"]
    for w in (10, 30):
        pol[f"⑤ 새 엔진 + 발표일만 {w}초부터"] = pd.Series(np.where(big, M[f"engine_{w}"], M["engine_1"]), M.index)
    rows = []
    for nm, fill in pol.items():
        p = pnl_from(fill).where(fill.notna(), 0.0)
        rows.append({"방식": nm, "총손익$": p.sum(), "4/9 빼고$": p.drop(ODD, errors="ignore").sum(),
                     "발표일 손익$": p[big].sum(), "나머지 날$": p[~big].sum(), "못 산 날": int(fill.isna().sum()),
                     "중간가 대비 건당$": ((fill - mid) * 100).mean()})
    B = pd.DataFrame(rows); B["기준 대비$"] = B["총손익$"] - B["총손익$"].iloc[0]
    B.to_csv(R / "N2_policy.csv", index=False, encoding="utf-8-sig")
    print(f"\nB. 매수 방식 (319일, 발표일 {int(big.sum())}일, 1분 중간가 기준 ${base.sum():,.0f})\n", B.round(0).to_string(index=False))
    # 발표일 부트스트랩: 대기 W초가 1초보다 나은 게 우연인지
    rng = np.random.default_rng(0)
    for w in (10, 30):
        diff = ((M["ask_1"].map(tick_up) - M[f"ask_{w}"].map(tick_up)) * 100)[big].values
        bs = np.array([rng.choice(diff, len(diff)).sum() for _ in range(5000)])
        print(f"  발표일 {w}초 대기 − 1초: 합 ${diff.sum():,.0f}, 95% [{np.percentile(bs, 2.5):,.0f}, {np.percentile(bs, 97.5):,.0f}], 나아진 날 {(diff > 0).mean():.0%}")
    diff = ((M["ask_1"].map(tick_up) - M["ask_-5"].map(tick_up)) * 100)
    for nm, sel in (("발표일", big), ("나머지", ~big)):
        v = diff[sel].values; bs = np.array([rng.choice(v, len(v)).sum() for _ in range(5000)])
        print(f"  09:59:55 선매수 − 1초 ({nm}): 합 ${v.sum():,.0f}, 건당 ${v.mean():.1f}, 95% [{np.percentile(bs, 2.5):,.0f}, {np.percentile(bs, 97.5):,.0f}]")

    # ---------- C ----------
    X, m1, m2, mnq1, FP = flip_risk()
    # SPX 5초 움직임 = 산 콜 중간가 09:59:55→10:00:00 변화 ÷ 델타 0.5 (319일), 10초는 √2배
    dF5 = (((M.ask_0 + M.bid_0) - (M["ask_-5"] + M["bid_-5"])) / 2).abs().dropna().values / 0.5
    # MNQ 움직임 = SPX 움직임 × (MNQ 1분 변화 ÷ SPX(F) 1분 변화, 09:59→10:00 중앙값)
    mc = mnq1["close"]
    t = [pd.Timestamp(f"{d:%Y-%m-%d} 09:59", tz=NY) for d in X.index]
    dm = pd.Series([abs(mc.get(x, np.nan) - mc.get(x - pd.Timedelta(minutes=1), np.nan)) for x in t], X.index)
    df1 = (FP["10:00"] - FP["09:59"]).abs().reindex(X.index)
    ratio = (dm / df1).replace([np.inf, -np.inf], np.nan).median()
    rows = []
    for nm, dF in (("5초 전 (09:59:55)", dF5), ("10초 전 (09:59:50)", dF5 * np.sqrt(2))):
        n1 = n2 = nsig = nbuy_wrong = 0.0
        for d in X.index:
            a, b = m1.get(d), m2.get(d)
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            q1 = (dF >= abs(a)).mean() * 0.5; q2 = (dF * ratio >= abs(b)).mean() * 0.5    # 선매수 시점엔 반대편이었을 확률
            n1 += q1; n2 += q2
            c1, c2 = a > 0, b > 0
            for f1, pf1 in ((c1, 1 - q1), (not c1, q1)):
                for f2, pf2 in ((c2, 1 - q2), (not c2, q2)):
                    if (f1 and f2) != (c1 and c2):
                        nsig += pf1 * pf2
                        nbuy_wrong += pf1 * pf2 * (f1 and f2)                                  # 선매수 땐 신호, 10:00엔 아님
        rows.append({"선매수 시점": nm, "SPX 움직임 중앙값 pt": np.median(dF), "MNQ 움직임 중앙값 pt": np.median(dF) * ratio,
                     "조건1 뒤집힐 날": n1, "조건2 뒤집힐 날": n2, "신호 다를 날": nsig, "그중 잘못 산 날": nbuy_wrong,
                     "1,093일 중 %": nsig / len(X) * 100})
    N3 = pd.DataFrame(rows); N3.to_csv(R / "N3_flip.csv", index=False, encoding="utf-8-sig")
    print(f"\nC. 선매수 신호 뒤집힘 기대 (1,093일, MNQ/SPX 움직임 비율 {ratio:.2f})\n", N3.round(2).to_string(index=False))

    # ---------- D ----------
    D = pd.DataFrame({f"{k}초": M[f"asksz_{k}"].describe(percentiles=[.1, .25, .5, .75]) for k in (0, 1, 10)}).T
    D["1계약 이상 %"] = [(M[f"asksz_{k}"] >= 1).mean() * 100 for k in (0, 1, 10)]
    D["3계약 이상 %"] = [(M[f"asksz_{k}"] >= 3).mean() * 100 for k in (0, 1, 10)]
    D["10계약 이상 %"] = [(M[f"asksz_{k}"] >= 10).mean() * 100 for k in (0, 1, 10)]
    D.to_csv(R / "N4_depth.csv", encoding="utf-8-sig")
    print("\nD. 매도호가 수량 (계약)\n", D.round(1).to_string())
    print("  발표일 1초 수량 중앙값:", M.loc[big, "asksz_1"].median(), " 나머지:", M.loc[~big, "asksz_1"].median())
    M.to_csv(R / "N_days.csv", encoding="utf-8-sig")

    # ---------- 차트 ----------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ks = [k for k in OFFS]
    for g in ORDER + ("전체",):
        S = A[A["발표"] == g]
        if not len(S):
            continue
        ax[0].plot(S["초"], S["매도호가 변화% 평균"], marker="o", lw=2.5 if g in ("전체",) else 1.5,
                   ls="--" if g == "전체" else "-", label=f"{g} ({int(S['날 수'].iloc[0])}일)")
    ax[0].axvline(0, color="gray", lw=0.8); ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_xscale("symlog", linthresh=5); ax[0].set_xticks(ks); ax[0].set_xticklabels(ks, fontsize=8)
    ax[0].set_title("산 콜 매도호가: 10:00:00 대비 평균 변화 (%)"); ax[0].set_xlabel("10:00:00 기준 초"); ax[0].legend(fontsize=8)
    Bp = B.set_index("방식")["기준 대비$"]
    ax[1].barh(range(len(Bp)), Bp.values, color=["#888"] + ["#4C72B0" if v >= 0 else "#C44E52" for v in Bp.values[1:]])
    ax[1].set_yticks(range(len(Bp))); ax[1].set_yticklabels(Bp.index, fontsize=8); ax[1].invert_yaxis()
    ax[1].axvline(0, color="gray", lw=0.8); ax[1].set_title("매수 방식별 총손익 차이 ($, ① 1초 즉시 대비, 319일)")
    for i, v in enumerate(Bp.values):
        ax[1].text(v, i, f" {v:+,.0f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(R / "N1_release.png", dpi=130)
    print("저장:", R / "N1_release.png")
