"""
VIX 기간구조 탐색 (output/holdout/탐색_VIX_계획.md, 50칸). 판정은 2023·2024만, 2022(5~12월)는 참고.
결과: output/holdout/V1_screen.csv, V1_screen.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.exits import FEE, EXERCISE
from spx0dte.holdout import DIR, discovery
from spx0dte import vix

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
TRADE_UNIT = np.sqrt(390 * 252 / 525600)          # 캘린더시간 IV → 거래시간 IV

# 인베스팅 CSV와 CBOE 값 일치 확인 (2024년까지만)
inv = pd.read_csv("data/vix9d_investing.csv", encoding="utf-8-sig")
inv.index = pd.to_datetime(inv.iloc[:, 0].str.replace(" ", "")); inv = discovery(inv.iloc[:, 1].astype(float), disc_start=pd.Timestamp("2020-01-01"))
V = vix.load()
diff = (V.VIX9D.reindex(inv.index) - inv).abs()
print(f"VIX9D CBOE vs 인베스팅 (2020~2024, {diff.notna().sum()}일): 최대 차이 {diff.max():.3f}")

prev = V.shift(1)                                   # 전날 종가 (그날 아침에 아는 값)
ch = pd.read_pickle(DIR / "disc_chain.pkl")["chain"]
t = ch[ch.entry == "10:00"].set_index("date")
t = t.join(prev.rename(columns={"VIX": "vix_prev", "VIX9D": "v9_prev"}))
t["ts_ratio"] = t.v9_prev / t.vix_prev
t["vix_chg"] = V.VIX.pct_change().shift(1).reindex(t.index)
t["iv_rel"] = t.iv_atm * TRADE_UNIT * 100 / t.v9_prev
assert t.index.max() < pd.Timestamp("2025-01-01")


def struct(kind):
    if kind == "스트래들":
        pay = t.c_pay + t.p_pay; cost = t.c_ask + t.p_ask; mid = t.c_mid + t.p_mid; x = 2 * FEE + EXERCISE * (pay > 0)
    else:
        p = "c" if kind == "콜" else "p"; pay, cost, mid = t[f"{p}_pay"], t[f"{p}_ask"], t[f"{p}_mid"]; x = FEE + EXERCISE * (pay > 0)
    return pd.DataFrame({"cost": cost, "pnl": pay - cost - x, "pnl_mid": pay - mid - x})


def stats(r):
    ret = lambda x: x.pnl.sum() / x.cost.sum() * 100 if len(x) else np.nan
    y = r.index.year; m = r[y >= 2023]; top3 = m.drop(m.pnl.nlargest(3).index)
    return {"건수": len(m), "수익률%": ret(m), "2023%": ret(m[m.index.year == 2023]), "2024%": ret(m[m.index.year == 2024]),
            "최고3일 제외%": ret(top3), "중간가%": m.pnl_mid.sum() / m.cost.sum() * 100, "$총": m.pnl.sum() * 100, "2022참고%": ret(r[y == 2022])}


rows = []
for idea, feat, kinds in (("V1 VIX9D÷VIX", "ts_ratio", ("콜", "풋", "스트래들")), ("V2 VIX 수준", "vix_prev", ("콜", "풋", "스트래들")),
                          ("V3 0DTE IV÷VIX9D", "iv_rel", ("스트래들", "콜")), ("V4 전날 VIX 변화", "vix_chg", ("콜", "풋"))):
    x = t[feat]; edges = x[x.index.year >= 2023].quantile([.2, .4, .6, .8]).values
    q = pd.Series(np.digitize(x, edges) + 1, index=t.index).where(x.notna())
    for kind in kinds:
        r = struct(kind)
        for k in range(1, 6):
            s = stats(r[q == k]); s.update({"아이디어": idea, "구조": kind, "분위": k,
                                           "범위": f"{x[(q == k) & (x.index.year >= 2023)].min():.3f}~{x[(q == k) & (x.index.year >= 2023)].max():.3f}"})
            rows.append(s)
H = pd.DataFrame(rows)
mono = H.groupby(["아이디어", "구조"]).apply(lambda g: g["분위"].corr(g["수익률%"], method="spearman"), include_groups=False)
H["단조(순위상관)"] = [mono[(a, b)] for a, b in zip(H["아이디어"], H["구조"])]
H["후보"] = (H["2023%"] > 0) & (H["2024%"] > 0) & (H["최고3일 제외%"] > 0) & (H["단조(순위상관)"].abs() >= 0.9)
H = H[["아이디어", "구조", "분위", "범위", "건수", "수익률%", "2023%", "2024%", "최고3일 제외%", "중간가%", "$총", "2022참고%", "단조(순위상관)", "후보"]]
H.to_csv(DIR / "V1_screen.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(H.round(2).to_string(index=False))
print(f"\n시도 {len(H)}칸, 두 해 모두 플러스 {((H['2023%'] > 0) & (H['2024%'] > 0)).sum()}, 후보 {H['후보'].sum()}")

grp = H.assign(묶음=H["아이디어"] + " · " + H["구조"]); names = grp["묶음"].unique()
fig, axs = plt.subplots(2, 5, figsize=(22, 8.5)); axs = axs.ravel()
for ax, nm in zip(axs, names):
    g = grp[grp["묶음"] == nm]; x = np.arange(1, 6)
    ax.bar(x - 0.2, g["2023%"], 0.4, color="#3D5A80", label="2023"); ax.bar(x + 0.2, g["2024%"], 0.4, color="#C9822B", label="2024")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_title(f"{nm} (순위상관 {g['단조(순위상관)'].iloc[0]:+.1f})", fontsize=10)
    ax.set_xlabel("분위 (1 = 가장 낮음)"); ax.set_ylabel("수익률 % (매도호가)")
axs[0].legend()
fig.suptitle("V1 VIX 기간구조 탐색 (2023·2024만, 10:00 ATM 만기 보유, 매도호가·수수료 포함, 전날 종가 기준)", fontsize=13)
fig.tight_layout(); fig.savefig(DIR / "V1_screen.png", dpi=110)
