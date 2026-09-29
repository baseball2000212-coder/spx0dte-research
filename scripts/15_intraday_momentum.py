"""
장중 모멘텀 (Gao·Han·Li·Zhou 2018, JFE "Market intraday momentum") 검증 — SPX 1분(F) + 인베스팅 종가.
  r1  = 전일 종가 → 10:00   (첫 30분, 밤사이 포함)
  r12 = 15:00 → 15:30        (12번째 30분)
  rL  = 15:30 → 종가(16:00)  (마지막 30분, 예측 대상)
① 지수: 회귀 rL ~ r1 + r12 (강건 표준오차), 방향 적중률, 분위·변동성 국면별, 학습/검증 나눠서
② 옵션: 15:30에 신호 방향 ATM 콜/풋 1개 매수 → 만기. 신호 강도·변동성 국면별.
결과: output/M1_momentum_index.png, M2_momentum_option.png, M_*.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.exits import FEE, EXERCISE

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
SPLIT = pd.Timestamp("2025-01-01")

FP = pd.read_pickle(OUT / "f_paths.pkl")
px = load_spx_ohlc(SPX_CSV); px["prev"] = px.close.shift(1)
half = FP.loc[:, "13:05":"15:59"].std(axis=1) < 1e-6                 # 조기폐장일 (13:00 마감) 제외
D = pd.DataFrame({"r1": FP["10:00"] / px.prev.reindex(FP.index) - 1,
                  "r12": FP["15:30"] / FP["15:00"] - 1,
                  "rL": px.close.reindex(FP.index) / FP["15:30"] - 1,
                  "F1530": FP["15:30"]})[~half].dropna()
daily = pd.read_csv(OUT / "daily_features.csv", parse_dates=["date"]).set_index("date")
D["iv10"] = daily["iv_10:00"].reindex(D.index)                    # 10:00 ATM IV (연율 %)
D["part"] = np.where(D.index < SPLIT, "학습", "검증")
for c in ("r1", "r12", "rL"):
    D[c + "_bp"] = D[c] * 1e4

# ═══ ① 지수 회귀 ═══
reg = []
for part, h in (("전체", D), ("학습", D[D.part == "학습"]), ("검증", D[D.part == "검증"])):
    for name, X in (("r1", ["r1_bp"]), ("r12", ["r12_bp"]), ("r1 + r12", ["r1_bp", "r12_bp"])):
        m = sm.OLS(h.rL_bp, sm.add_constant(h[X])).fit(cov_type="HC1")
        row = {"구간": part, "모형": name, "일수": len(h), "R²%": m.rsquared * 100}
        for x in X:
            row[f"{x[:-3]} 계수"] = m.params[x]; row[f"{x[:-3]} t값"] = m.tvalues[x]
        reg.append(row)
reg = pd.DataFrame(reg).round(3)
# 학습 계수로 검증 예측 → 표본외 R² (기준 = 학습 평균)
tr, va = D[D.part == "학습"], D[D.part == "검증"]
m = sm.OLS(tr.rL_bp, sm.add_constant(tr[["r1_bp", "r12_bp"]])).fit()
pred = m.predict(sm.add_constant(va[["r1_bp", "r12_bp"]]))
oos_r2 = 1 - ((va.rL_bp - pred) ** 2).sum() / ((va.rL_bp - tr.rL_bp.mean()) ** 2).sum()

hit = []
for part, h in (("전체", D), ("학습", D[D.part == "학습"]), ("검증", D[D.part == "검증"])):
    for sig in ("r1", "r12"):
        k = h[(h[sig] != 0) & (h.rL != 0)]
        n, w = len(k), int((np.sign(k[sig]) == np.sign(k.rL)).sum())
        hit.append({"구간": part, "신호": sig, "일수": n, "방향 적중%": w / n * 100,
                    "p값(동전던지기 대비)": binomtest(w, n, 0.5, alternative="greater").pvalue})
hit = pd.DataFrame(hit).round(4)

# 분위(|r1|, 학습 경계) · 변동성 국면(10시 IV 학습 중앙값)
_, ed = pd.qcut(D[D.part == "학습"].r1.abs(), 5, retbins=True); ed[0], ed[-1] = -np.inf, np.inf
D["q"] = pd.cut(D.r1.abs(), ed, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
D["vol"] = np.where(D.iv10 > D[D.part == "학습"].iv10.median(), "고변동", "저변동")
D["follow_bp"] = np.sign(D.r1) * D.rL_bp                             # r1 방향으로 마지막 30분 탔을 때 (bp)
D["follow_pt"] = np.sign(D.r1) * D.rL * D.F1530                      # 같은 걸 SPX 포인트로
D.to_csv(OUT / "M_days.csv", encoding="utf-8-sig")

# ═══ ② 옵션: 15:30 신호 방향 ATM 1개 → 만기 ═══
L = pd.read_pickle(OUT / "leg_paths.pkl")
L = L[L.entry == "15:30"].set_index("date")
O = D.join(L[["call_ask", "put_ask", "call_pay", "put_pay"]], how="inner")
up = O.r1 > 0
O["cost"] = np.where(up, O.call_ask, O.put_ask)
pay = np.where(up, O.call_pay, O.put_pay)
O["pnl"] = pay - O.cost - FEE - np.where(pay > 0, EXERCISE, 0)
O["both"] = np.sign(O.r1) == np.sign(O.r12)                          # r1·r12 같은 방향
ret = lambda g: g.pnl.sum() / g.cost.sum() * 100 if len(g) else np.nan
opt = []
for part in ("학습", "검증"):
    h = O[O.part == part]
    opt.append({"구간": part, "조건": "전체", "건수": len(h), "수익률%": ret(h)})
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        opt.append({"구간": part, "조건": f"|r1| {q}", "건수": int((h.q == q).sum()), "수익률%": ret(h[h.q == q])})
    for v in ("저변동", "고변동"):
        opt.append({"구간": part, "조건": v, "건수": int((h.vol == v).sum()), "수익률%": ret(h[h.vol == v])})
    opt.append({"구간": part, "조건": "r1·r12 같은 방향", "건수": int(h.both.sum()), "수익률%": ret(h[h.both])})
    opt.append({"구간": part, "조건": "고변동 + Q5", "건수": int(((h.vol == "고변동") & (h.q == "Q5")).sum()),
                "수익률%": ret(h[(h.vol == "고변동") & (h.q == "Q5")])})
opt = pd.DataFrame(opt).round(1)

# ═══ 차트 ═══
fig, ax = plt.subplots(2, 2, figsize=(16, 10))
x = ax[0, 0]
for part, col in (("학습", "C0"), ("검증", "C1")):
    h = D[D.part == part]
    x.scatter(h.r1_bp, h.rL_bp, s=6, alpha=.35, color=col, label=part)
b = reg[(reg.구간 == "전체") & (reg.모형 == "r1")].iloc[0]
xs = np.linspace(D.r1_bp.quantile(.005), D.r1_bp.quantile(.995), 50)
x.plot(xs, b["r1 계수"] * xs + D.rL_bp.mean(), color="k", lw=1.5, label=f"기울기 {b['r1 계수']:.3f} (t={b['r1 t값']:.1f})")
x.axhline(0, color="grey", lw=.8); x.axvline(0, color="grey", lw=.8)
x.set_xlim(D.r1_bp.quantile(.005), D.r1_bp.quantile(.995)); x.set_ylim(D.rL_bp.quantile(.005), D.rL_bp.quantile(.995))
x.set_xlabel("첫 30분 (전일종가→10:00, bp)"); x.set_ylabel("마지막 30분 (15:30→종가, bp)"); x.legend(fontsize=8)
x.set_title("첫 30분 방향 vs 마지막 30분 움직임")
x = ax[0, 1]
g = D.groupby(["q", "part"], observed=False).follow_pt.mean().unstack()[["학습", "검증"]]
g.plot.bar(ax=x, rot=0); x.axhline(0, color="k", lw=1)
x.set_title("첫 30분 방향으로 마지막 30분 탔을 때 평균 (SPX 포인트)\nQ1 첫 30분 작게 움직임 → Q5 크게 움직임"); x.set_xlabel("")
x = ax[1, 0]
for sig, col in (("r1", "C2"), ("r12", "C4")):
    s = (np.sign(D[sig]) * D.rL * D.F1530).cumsum()
    x.plot(D.index, s, color=col, label=f"{sig} 방향으로 마지막 30분 (SPX 1배)")
x.axhline(0, color="k", lw=1); x.axvline(SPLIT, color="grey", ls="--"); x.legend(); x.grid(alpha=.3)
x.set_title("누적 포인트 (분석용 — 지수 자체, 옵션 아님)")
x = ax[1, 1]
yr = D.groupby(D.index.year).apply(lambda h: pd.Series({
    "적중%": (np.sign(h.r1) == np.sign(h.rL)).mean() * 100,
    "평균 포인트": h.follow_pt.mean()}), include_groups=False)
x.bar(yr.index.astype(str), yr["평균 포인트"], color=["C2" if v > 0 else "C3" for v in yr["평균 포인트"]])
for i, (a, b2) in enumerate(zip(yr["평균 포인트"], yr["적중%"])):
    x.text(i, a, f"적중 {b2:.0f}%", ha="center", va="bottom" if a > 0 else "top", fontsize=9)
x.axhline(0, color="k", lw=1); x.set_title("연도별: r1 방향 마지막 30분 평균 포인트")
fig.tight_layout(); fig.savefig(OUT / "M1_momentum_index.png", dpi=115); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for j, part in enumerate(("학습", "검증")):
    h = opt[opt.구간 == part]
    ax[j].barh(h.조건, h["수익률%"], color=["C2" if v > 0 else "C3" for v in h["수익률%"]])
    for k, (v, n) in enumerate(zip(h["수익률%"], h.건수)):
        ax[j].text(v, k, f" {v:.0f}% ({n}건)", va="center", fontsize=8)
    ax[j].axvline(0, color="k", lw=1); ax[j].invert_yaxis()
    ax[j].set_title(f"15:30 신호 방향 ATM 1개 매수 → 만기 | {part} (수익률 %, 수수료 포함)")
fig.tight_layout(); fig.savefig(OUT / "M2_momentum_option.png", dpi=115); plt.close(fig)

reg.to_csv(OUT / "M_regression.csv", index=False, encoding="utf-8-sig")
opt.to_csv(OUT / "M_option.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print("① 회귀 (bp 단위, 강건 t값):\n", reg.to_string(index=False))
print(f"\n표본외 R² (학습 계수 → 검증): {oos_r2 * 100:.2f}%")
print("\n방향 적중:\n", hit.to_string(index=False))
print("\n|r1| 분위별 평균 (SPX 포인트):\n", D.groupby(["q", "part"], observed=False).follow_pt.mean().unstack().round(2).to_string())
print("\n변동성 국면별 평균 (SPX 포인트):\n", D.groupby(["vol", "part"]).follow_pt.mean().unstack().round(2).to_string())
print(f"\n마지막 30분 평균 |움직임| {(D.rL.abs() * D.F1530).mean():.2f}pt  vs  15:30 ATM 콜·풋 평균 매수가 {O.cost.mean():.2f}pt")
print("\n② 옵션:\n", opt.to_string(index=False))
