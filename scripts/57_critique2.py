"""
두 번째 외부 검토(Claude) 대응 (2026-09-27).
  ① 요일별 분해: 현재 규칙(무손절 · −90%) vs 매일 매수 기준선 — 건수·건당·t값·연도별, 4/9 포함·제외. 표본외(2020-01~2022-05)도 같은 표
  ② 국면 표(8절 ⑤)를 기준선으로도 — 같은 3분위 경계(전체 날 기준)에서 조건부 − 무조건
  ⑤ 검정력: 가정 엣지 $100·$160·$270별 t=2까지 필요한 매매 수·기간
  ⑥ XSP 1·3계약 재표본 최대낙폭 (계좌 $100k 대비 %)
결과: output/critique2/*.csv, critique2.json, C1~C3 png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, os, warnings
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV, DATA
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_rule, oos_call_path

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
D = OUT / "critique2"; D.mkdir(exist_ok=True)
ODD = pd.Timestamp("2025-04-09"); DOW = "월화수목금"; ACCOUNT = 100_000; RNG = np.random.default_rng(11); NSIM = 10_000
XSP_FEE = 2.5


def day_pnls(rows, stop):
    """날짜별 SPX 1계약 손익$ + XSP 1계약 손익$ (XSP = SPX 가격의 1/10, 수수료 계약당 편도 $2.5, 스프레드 차이는 무시)."""
    out = {}
    for r in rows:
        t = trade_rule(r, stop=stop)
        exitp = t["stop"][1] if t["stop"] else r.pay
        sides = 1 + (1 if (t["stop"] or r.pay > 0) else 0)
        out[r.date] = {"spx$": t["pnl"] * 100, "xsp$": (exitp - t["cost"]) * 10 - XSP_FEE * sides, "cost$": t["cost"] * 100}
    return pd.DataFrame(out).T.sort_index()


def dow_table(P, sig, label):
    """P: 날짜별 손익 (전체 날), sig: 신호 (bool Series)."""
    rows = []
    for who, m in (("현재 규칙", sig), ("매일 매수", pd.Series(True, index=P.index))):
        s = P["spx$"][m.reindex(P.index).fillna(False).astype(bool)]
        for ex in (False, True):
            ss = s.drop(ODD, errors="ignore") if ex else s
            for k in range(5):
                v = ss[ss.index.dayofweek == k]
                if not len(v):
                    continue
                rows.append({"구간": label, "대상": who, "4/9": "제외" if ex else "포함", "요일": DOW[k], "건수": len(v), "건당$": v.mean(),
                             "t값": v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan, "총$": v.sum(),
                             **{str(y): v[v.index.year == y].sum() for y in sorted(set(P.index.year))}})
            v = ss
            rows.append({"구간": label, "대상": who, "4/9": "제외" if ex else "포함", "요일": "전체", "건수": len(v), "건당$": v.mean(),
                         "t값": v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), "총$": v.sum(),
                         **{str(y): v[v.index.year == y].sum() for y in sorted(set(P.index.year))}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    FP = pd.read_pickle(OUT / "f_paths.pkl"); px = load_spx_ohlc(SPX_CSV)
    X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
    L = pd.read_pickle(OUT / "legs10_full.pkl")
    L = L[(L.leg == "C") & (L.offset == 0) & (L.entry == "10:00")].set_index("date").sort_index()
    rows = [SimpleNamespace(date=d, mid=r.mid, ask=r.ask, pay=r.pay, mid_path=r.mid_path, bid_path=r.bid_path, ask_path=r.ask_path) for d, r in L.iterrows()]
    sig = X.signal.reindex(L.index).fillna(False)
    P0, P9 = day_pnls(rows, None), day_pnls(rows, 0.9)

    # 표본외 전체 날 경로 (캐시)
    cache = D / "oos_all_paths.pkl"
    O = pd.read_csv(OUT / "oos" / "days.csv", index_col=0, parse_dates=True); O = O[O["행사가"].notna() & O["SPX 종가"].notna()]
    if cache.exists():
        OR = [SimpleNamespace(**x) for x in pd.read_pickle(cache)]
    else:
        jobs = [(DATA / "0dte_oos" / f"{d:%Y-%m-%d}.dbn.zst", float(r["행사가"]), float(r["SPX 종가"])) for d, r in O.iterrows()]
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
            res = sorted(ex.map(oos_call_path, jobs, chunksize=4), key=lambda x: x["date"])
        pd.to_pickle(res, cache); OR = [SimpleNamespace(**x) for x in res]
    osig = (O["신호"] == True)
    Q0, Q9 = day_pnls(OR, None), day_pnls(OR, 0.9)
    chk = (Q0["spx$"] - O["손익$"].reindex(Q0.index)).abs().max()
    print(f"표본외 재계산 확인: days.csv 손익과 최대 차이 ${chk:.2f}")

    # ① 요일
    W = pd.concat([dow_table(P0, sig, "2022-05~2026-09 무손절"), dow_table(P9, sig, "2022-05~2026-09 −90%"),
                   dow_table(Q0, osig, "표본외 2020-01~2022-05 무손절"), dow_table(Q9, osig, "표본외 2020-01~2022-05 −90%")], ignore_index=True)
    W.to_csv(D / "C1_weekday.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    show = ["구간", "대상", "4/9", "요일", "건수", "건당$", "t값", "총$", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    print(W[[c for c in show if c in W]].round(1).to_string(index=False))

    # ② 국면: 같은 경계 (전체 날 3분위) — 무손절
    daily = pd.read_csv(OUT / "daily_features.csv", parse_dates=["date"]).set_index("date")
    Z = pd.DataFrame({"10시 IV": daily["iv_10:00"], "IV 전일 대비": daily["iv_10:00"] - daily["iv_10:00"].shift(1), "밤사이 갭%": daily.gap,
                      "개장 30분 상승폭%": X.r30 * 100, "전일 실현변동성": daily.rv_day.shift(1)}).reindex(P0.index)
    reg = []
    for col in Z.columns:
        z = Z[col].dropna(); edges = z.quantile([1 / 3, 2 / 3]).values
        b = pd.Series(np.digitize(z, edges), index=z.index).map({0: "낮음", 1: "중간", 2: "높음"})
        for k in ("낮음", "중간", "높음"):
            dd = b.index[b == k]
            for ex in (False, True):
                d2 = dd.drop(ODD, errors="ignore") if ex else dd
                allv = P0["spx$"].reindex(d2); sv = allv[sig.reindex(d2).values]
                reg.append({"피처": col, "구간": k, "4/9": "제외" if ex else "포함", "범위": f"{z[b == k].min():.2f} ~ {z[b == k].max():.2f}",
                            "매일 매수 건수": len(allv), "매일 매수 건당$": allv.mean(), "규칙 건수": len(sv), "규칙 건당$": sv.mean(),
                            "조건부−무조건$": sv.mean() - allv.mean()})
    G = pd.DataFrame(reg); G.to_csv(D / "C2_regime.csv", index=False, encoding="utf-8-sig")
    print("\n② 국면 (무손절, 경계 = 전체 날 3분위)\n", G[G["4/9"] == "포함"].drop(columns="4/9").round(0).to_string(index=False))

    # ⑤ 검정력
    pw = []
    n_year = sig.sum() / (len(P0) / 252)
    tt = sig & sig.index.dayofweek.isin([1, 3])
    n_year_tt = tt.sum() / (len(P0) / 252)
    for lab, P in (("무손절", P0), ("−90%", P9)):
        v = P["spx$"][sig]; sd = v.std(ddof=1); sdx = v.drop(ODD, errors="ignore").std(ddof=1)
        for e in (100, 160, 270):
            n = (2 * sd / e) ** 2; nx = (2 * sdx / e) ** 2
            pw.append({"손절": lab, "가정 엣지 $/건": e, "건당 표준편차$": sd, "t=2 필요 건수": n, "기간(년, 연 {:.0f}건)".format(n_year): n / n_year,
                       "기간(년, 화·목만 연 {:.0f}건)".format(n_year_tt): n / n_year_tt,
                       "4/9 빼고 표준편차$": sdx, "4/9 빼고 필요 건수": nx})
    PW = pd.DataFrame(pw); PW.to_csv(D / "C3_power.csv", index=False, encoding="utf-8-sig")
    print("\n⑤ 검정력\n", PW.round(1).to_string(index=False))

    # ⑥ XSP 재표본 최대낙폭
    xs = []
    for lab, P in (("무손절", P0), ("−90%", P9)):
        for ncon in (1, 3):
            v = P["xsp$"][sig].values * ncon
            idx = RNG.integers(0, len(v), (NSIM, len(v))); eq = v[idx].cumsum(1); mdd = (eq - np.maximum.accumulate(eq, 1)).min(1)
            e0 = np.cumsum(v); real = (e0 - np.maximum.accumulate(e0)).min()
            xs.append({"손절": lab, "XSP 계약": ncon, "총손익$": v.sum(), "건당$": v.mean(), "실제 최대낙폭$": real, "실제 %": real / ACCOUNT * 100,
                       "재표본 중앙값$": np.median(mdd), "중앙값 %": np.median(mdd) / ACCOUNT * 100,
                       "최악 5%$": np.percentile(mdd, 5), "최악 5% %": np.percentile(mdd, 5) / ACCOUNT * 100,
                       "최악 1%$": np.percentile(mdd, 1), "최악 1% %": np.percentile(mdd, 1) / ACCOUNT * 100})
    XS = pd.DataFrame(xs); XS.to_csv(D / "C4_xsp_drawdown.csv", index=False, encoding="utf-8-sig")
    print("\n⑥ XSP 재표본 최대낙폭 (계좌 $100k)\n", XS.round(1).to_string(index=False))

    json.dump({"weekday": W.to_dict("records"), "regime": G.to_dict("records"), "power": PW.to_dict("records"), "xsp": XS.to_dict("records"),
               "n_year": n_year, "n_year_tt": n_year_tt}, open(D / "critique2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    # 차트 C1: 요일별 건당 (95% 구간)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), sharey=False)
    for ax, lab in ((axs[0], "2022-05~2026-09 무손절"), (axs[1], "표본외 2020-01~2022-05 무손절")):
        w = W[(W["구간"] == lab) & (W["4/9"] == "포함") & (W["요일"] != "전체")]
        for off, who, col in ((-0.2, "현재 규칙", "#B23A48"), (0.2, "매일 매수", "#9AA3AD")):
            g = w[w["대상"] == who].set_index("요일").reindex(list(DOW))
            se = g["건당$"] / g["t값"]
            x = np.arange(5) + off
            ax.bar(x, g["건당$"], 0.4, color=col, label=who, yerr=1.96 * se, capsize=3)
            for xi, (n, t) in zip(x, zip(g["건수"], g["t값"])):
                if np.isfinite(n):
                    ax.text(xi, 0, f"{int(n)}건\nt={t:.1f}", ha="center", va="top", fontsize=7)
        ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(5), list(DOW)); ax.set_title(f"{lab}: 요일별 건당 손익 $ (95% 구간)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(D / "C1_weekday.png", dpi=110)
