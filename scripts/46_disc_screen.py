"""
탐색 계획(output/holdout/탐색_2023-24_계획.md) 69칸 채점. 판정은 2023·2024만, 2022(5~12월)는 참고.
결과: output/holdout/H1_screen.csv, H1_screen.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spx0dte.exits import FEE, EXERCISE
from spx0dte.holdout import DIR

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
P = pd.read_pickle(DIR / "disc_chain.pkl"); ch = P["chain"]
D = pd.read_csv(DIR / "disc_days.csv", index_col=0, parse_dates=True)


def leg(t, pre):
    pay, a, m = t[f"{pre}_pay"], t[f"{pre}_ask"], t[f"{pre}_mid"]
    return pd.DataFrame({"cost": a, "pnl": pay - a - FEE - EXERCISE * (pay > 0), "pnl_mid": pay - m - FEE - EXERCISE * (pay > 0)}, index=t.index)


def struct(t, kind):
    if kind == "콜":
        return leg(t, "c")
    if kind == "풋":
        return leg(t, "p")
    a, b = ("c", "p") if kind == "스트래들" else ("c5", "p5")
    pay = t[f"{a}_pay"] + t[f"{b}_pay"]
    cost = t[f"{a}_ask"] + t[f"{b}_ask"]; mid = t[f"{a}_mid"] + t[f"{b}_mid"]
    x = 2 * FEE + EXERCISE * (pay > 0)
    return pd.DataFrame({"cost": cost, "pnl": pay - cost - x, "pnl_mid": pay - mid - x}, index=t.index)


def at(e):
    t = ch[ch.entry == e].set_index("date")
    return t


def stats(r):
    """r: cost·pnl (날짜 인덱스). 판정 구간 2023-24."""
    ret = lambda x: x.pnl.sum() / x.cost.sum() * 100 if len(x) else np.nan
    y = r.index.year; m = r[(y >= 2023)]
    top3 = m.drop(m.pnl.nlargest(3).index)
    return {"건수": len(m), "수익률%": ret(m), "2023%": ret(m[m.index.year == 2023]), "2024%": ret(m[m.index.year == 2024]),
            "최고3일 제외%": ret(top3), "중간가%": m.pnl_mid.sum() / m.cost.sum() * 100 if len(m) else np.nan,
            "$총": m.pnl.sum() * 100, "2022참고%": ret(r[y == 2022])}


rows = []
def add(idea, cond, kind, entry, r, grp=None, q=None):
    s = stats(r.dropna()); s.update({"아이디어": idea, "조건": cond, "구조": kind, "진입": entry, "묶음": grp, "분위": q}); rows.append(s)


def quint(idea, e, feat, kinds, label):
    t = at(e).join(D)
    x = t[feat]
    edges = x[(x.index.year >= 2023)].quantile([.2, .4, .6, .8]).values          # 경계는 2023-24로
    qq = np.digitize(x, edges) + 1
    for kind in kinds:
        r = struct(t, kind)
        for q in range(1, 6):
            add(idea, f"{label} {q}분위", kind, e, r[(qq == q) & x.notna()], grp=f"{idea}-{kind}", q=q)


# A 스큐 (10:00): 풋 −0.5% IV − 콜 +0.5% IV
ch["skew"] = ch["p5_iv"] - ch["c5_iv"]
D = D.join(ch[ch.entry == "10:00"].set_index("date")[["skew"]].rename(columns={"skew": "skew10"}))
quint("A 스큐", "10:00", "skew10", ("콜", "풋", "스트래들"), "스큐")
quint("B 밤사이 MNQ", "09:45", "mnq_on", ("콜", "풋"), "밤사이")
quint("C 전날 SPX", "09:45", "prev_ret", ("콜", "풋"), "전날")
quint("D 밤사이 금", "09:45", "mgc_on", ("콜", "풋"), "금")

t10 = at("10:00").join(D)
c10, s10 = struct(t10, "콜"), struct(t10, "스트래들")
add("E 월말·월초", "월말·월초", "콜", "10:00", c10[t10.tom])
add("E 월말·월초", "나머지", "콜", "10:00", c10[~t10.tom])
for kind in ("콜", "풋", "스트래들"):
    r = struct(t10, kind)
    for k, nm in enumerate("월화수목금"):
        add("F 요일", nm, kind, "10:00", r[t10.dow == k], grp=f"F 요일-{kind}")
add("G 만기·휴일", "월물 만기 금요일", "스트래들", "10:00", s10[t10.opex])
add("G 만기·휴일", "월물 만기 금요일", "콜", "10:00", c10[t10.opex])
add("G 만기·휴일", "휴일 전날", "스트래들", "10:00", s10[t10.pre_hol])
add("G 만기·휴일", "휴일 전날", "콜", "10:00", c10[t10.pre_hol])
for e in ("09:45", "10:00", "12:00"):
    add("H 스트랭글", "무조건 ±0.5%", "스트랭글", e, struct(at(e), "스트랭글"))

H = pd.DataFrame(rows)
# 분위 묶음 단조성 (1→5분위 순위상관)
mono = H.dropna(subset=["분위"]).groupby("묶음").apply(lambda g: g["분위"].corr(g["수익률%"], method="spearman"), include_groups=False)
H["단조(순위상관)"] = H["묶음"].map(mono)
H["후보"] = (H["2023%"] > 0) & (H["2024%"] > 0) & (H["최고3일 제외%"] > 0) & (H["분위"].isna() | (H["단조(순위상관)"].abs() >= 0.9))
cols = ["아이디어", "조건", "구조", "진입", "건수", "수익률%", "2023%", "2024%", "최고3일 제외%", "중간가%", "$총", "2022참고%", "단조(순위상관)", "후보"]
H = H[cols]; H.to_csv(DIR / "H1_screen.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(H.round(1).to_string(index=False))
print(f"\n시도 {len(H)}칸, 두 해 모두 플러스 {((H['2023%'] > 0) & (H['2024%'] > 0)).sum()}, 후보 {H['후보'].sum()}")

# 차트: 분위 묶음별 2023·2024 막대
grp = H.dropna(subset=["단조(순위상관)"]).copy()
grp["묶음"] = grp["아이디어"] + " · " + grp["구조"]
names = grp["묶음"].unique()
fig, axs = plt.subplots(3, 4, figsize=(18, 11)); axs = axs.ravel()
for ax, nm in zip(axs, names):
    g = grp[grp["묶음"] == nm]; x = np.arange(1, 6)
    ax.bar(x - 0.2, g["2023%"], 0.4, color="#3D5A80", label="2023"); ax.bar(x + 0.2, g["2024%"], 0.4, color="#C9822B", label="2024")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_title(f"{nm} (순위상관 {g['단조(순위상관)'].iloc[0]:+.1f})", fontsize=10)
    ax.set_xlabel("분위 (1 = 가장 낮음)"); ax.set_ylabel("수익률 % (매도호가)")
for ax in axs[len(names):]:
    ax.axis("off")
axs[0].legend()
fig.suptitle("H1 새 아이디어 탐색 (2023·2024만, 만기 보유, 매도호가 체결·수수료 포함)", fontsize=13)
fig.tight_layout(); fig.savefig(DIR / "H1_screen.png", dpi=110)
