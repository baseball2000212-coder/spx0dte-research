"""
포트폴리오용 PDF (NH선물 Rest API Pioneer 지원 참고자료): 전략 요약·백테스트·2026년 9월 매매 기록·연구 과정·API 구현 계획.
결과: output/portfolio/SPX0DTE_전략_포트폴리오.pdf (+ 차트 png)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, shutil, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_detail, trade_rule
from spx0dte.events import label_events

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
P = OUT / "portfolio"; P.mkdir(exist_ok=True)
INK, ACC, LOSS, MUTED = "#17201C", "#0F6E5A", "#B3261E", "#6B7570"

FP = pd.read_pickle(OUT / "f_paths.pkl")
px = load_spx_ohlc(SPX_CSV)
X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), px)
L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date")
L = L[(L.leg == "C") & (L.offset == 0)].join(X, how="inner")
S = L[L.signal].sort_index()
T = pd.DataFrame([trade_rule(r, stop=None) for r in S.itertuples()], index=S.index)      # 사전 등록 규칙 (무손절, 만기) = 헤드라인
T9 = pd.DataFrame([trade_rule(r) for r in S.itertuples()], index=S.index)                # 운용 변형: −90% 손절 (표본 내에서 고름)
T9["pnl$"] = T9.pnl * 100
T["pnl$"] = T.pnl * 100
eq = T["pnl$"].cumsum(); dd = eq - eq.cummax()
RV = json.load(open(OUT / "rule_v2" / "summary.json", encoding="utf-8"))
V2 = RV["backtest"]; MAIN = V2["지연 0초 · 중간가 · 손절 없음"]; VAR9 = V2["지연 0초 · 중간가 · 손절 −90%"]
C2 = json.load(open(OUT / "critique2" / "critique2.json", encoding="utf-8"))
shutil.copy(OUT / "critique2" / "C1_weekday.png", P / "c2_weekday.png")
C3 = json.load(open(OUT / "critique3" / "critique3.json", encoding="utf-8"))
N = {"yearly": MAIN["연도별$"], "yearly_n": MAIN["연도별 건수"], "avg_cost": round(MAIN["평균 투입$"]), "worst": round(MAIN["최악의 날$"]),
     "best": round(MAIN["최고의 날$"]), "total_ex": round(MAIN["4/9 빼고$"]),
     "train": round(T[T.index < "2025-01-01"].pnl.sum() / T[T.index < "2025-01-01"].cost.sum() * 100, 1),
     "val": round(T[T.index >= "2025-01-01"].pnl.sum() / T[T.index >= "2025-01-01"].cost.sum() * 100, 1)}
for f_, g_ in ((OUT / "stops" / "S1_stop_sweep.png", "stop_sweep.png"),):
    shutil.copy(f_, P / g_)
SW = pd.read_csv(OUT / "stops" / "stop_sweep.csv"); SW = SW[SW["매수가"] == "중간가"]
e26 = T["pnl"][T.index.year == 2026].mul(100).cumsum(); d26 = e26 - e26.cummax(); t26 = d26.idxmin(); p26 = e26[:t26].idxmax()
r26 = e26[t26:][e26[t26:] >= e26[p26]]; rec26 = f"{r26.index[0].month}월 {r26.index[0].day}일에 회복했다" if len(r26) else "아직 회복 전이다"

# ── 차트 ──
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=.25)

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(eq.index, eq.values / 1000, color=ACC, lw=1.6)
ax.fill_between(eq.index, eq.values / 1000, (eq.cummax()).values / 1000, color=LOSS, alpha=.12, label="고점 대비 하락")
ax.axhline(0, color=INK, lw=.8)
ax.axvline(pd.Timestamp("2025-04-09"), color=LOSS, ls=":", lw=.8); ax.text(pd.Timestamp("2025-04-09"), eq.max() / 1000 * .92, " 2025-04-09 관세 유예", color=LOSS, fontsize=8)
ax.set_ylabel("누적 손익 (천 달러)"); ax.legend(frameon=False, fontsize=8, loc="upper left"); style(ax)
fig.tight_layout(); fig.savefig(P / "equity.png", dpi=170); plt.close(fig)

m26 = T["pnl$"][T.index.year == 2026]
mm = m26.groupby(m26.index.month).sum()
fig, ax = plt.subplots(figsize=(9, 2.9))
ax.bar([f"{i}월" for i in mm.index], mm.values / 1000, color=[ACC if v > 0 else LOSS for v in mm.values])
ax2 = ax.twinx(); ax2.plot([f"{i}월" for i in mm.index], mm.cumsum().values / 1000, color=INK, marker="o", lw=1.2)
ax2.set_ylabel("누적 (천 달러)"); ax.set_ylabel("월 손익 (천 달러)"); ax.axhline(0, color=INK, lw=.8); style(ax)
ax2.spines["top"].set_visible(False)
fig.tight_layout(); fig.savefig(P / "monthly2026.png", dpi=170); plt.close(fig)

# ── 전체 기간 월별 (2022-05 ~) ──
allm = T["pnl$"].groupby([T.index.year, T.index.month]).sum()
alln = T["pnl$"].groupby([T.index.year, T.index.month]).size()
labels = [f"{y % 100:02d}.{m:02d}" for y, m in allm.index]
fig, ax = plt.subplots(figsize=(9, 3.0))
ax.bar(range(len(allm)), allm.values / 1000, color=[ACC if v > 0 else LOSS for v in allm.values], width=.8)
ax.set_xticks(range(0, len(allm), 3), labels[::3], fontsize=7, rotation=45)
ax.axhline(0, color=INK, lw=.8); ax.set_ylabel("월 손익 (천 달러)"); style(ax)
ax.annotate("2025-04 (관세)", (labels.index("25.04"), allm.loc[(2025, 4)] / 1000), fontsize=7, color=LOSS, xytext=(4, -4), textcoords="offset points")
fig.tight_layout(); fig.savefig(P / "monthly_all.png", dpi=170); plt.close(fig)
years = sorted({y for y, _ in allm.index})
mtab = "<tr><th>연도</th>" + "".join(f"<th>{m}</th>" for m in range(1, 13)) + "<th>합계</th></tr>"
for y in years:
    cells = []
    for m in range(1, 13):
        if (y, m) in allm.index:
            v = allm.loc[(y, m)]
            cells.append(f"<td class='{'pos' if v > 0 else 'neg'}'>{v / 1000:+.1f}</td>")
        else:
            cells.append("<td></td>")
    tot = allm.loc[y].sum()
    mtab += f"<tr><th>{y}</th>{''.join(cells)}<td class='{'pos' if tot > 0 else 'neg'}'><b>{tot / 1000:+.1f}</b></td></tr>"
n_pos, n_all = int((allm > 0).sum()), len(allm)

# ── 국면 (MNQ 일봉 200일선, 전날 기준) ──
mq = load_mnq()[0]
dc = mq.close[(mq.index.hour == 15) & (mq.index.minute == 59)]; dc.index = dc.index.tz_localize(None).normalize()
dc = dc[~dc.index.duplicated()]
reg = (dc > dc.rolling(200).mean()).shift(1).map({True: "강세", False: "약세"}).reindex(T.index)
nqdd = ((dc / dc.cummax() - 1).shift(1) * 100).reindex(T.index)
daily = pd.read_csv(OUT / "daily_features.csv", parse_dates=["date"]).set_index("date")
G = pd.DataFrame({"pnl": T.pnl, "cost": T.cost, "pay": T.pay, "reg": reg, "iv": daily["iv_10:00"].reindex(T.index),
                  "prem": T.cost / S.f10 * 100, "move": (px.close.reindex(T.index) / S.f10 - 1) * 100}).dropna(subset=["reg"])
def rg(g):
    return {"n": len(g), "ret": g.pnl.sum() / g.cost.sum() * 100, "usd": g.pnl.sum() * 100, "per": g.pnl.mean() * 100,
            "prem": g.prem.mean(), "iv": g.iv.mean(), "abs": g.move.abs().mean(), "up": (g.move > 0).mean() * 100,
            "drift": g.move.mean(), "x2": (g.pay / g.cost >= 2).mean() * 100, "zero": (g.pay <= 0).mean() * 100}
RG = {k: rg(g) for k, g in G.groupby("reg")}
RGX = {k: rg(g) for k, g in G.drop(pd.Timestamp("2025-04-09"), errors="ignore").groupby("reg")}
ddb = pd.cut(nqdd, [-100, -20, -10, -5, 0.01], labels=["−20% 이하", "−10~−20%", "−5~−10%", "−5% 이내"])
DDT = G.assign(b=ddb).groupby("b", observed=False).apply(lambda g: (len(g), g.pnl.sum() / g.cost.sum() * 100 if len(g) else np.nan), include_groups=False)
m22 = T["pnl$"][T.index.year == 2022].groupby(T.index[T.index.year == 2022].month).sum()
nqm = (dc[dc.index.year == 2022].resample("ME").last().pct_change() * 100); nqm.index = nqm.index.month

# ── 2026년 9월 일별 기록 (신호 없는 날 포함) ──
sep = FP.index[(FP.index.year == 2026) & (FP.index.month == 9)]
ev = dict(zip(sep, label_events(sep)))
rows = []
for d in sep:
    x = X.loc[d]
    r = T.loc[d] if d in T.index else None
    r9 = T9.loc[d] if d in T9.index else None
    k = S.loc[d, "K"] if d in S.index else np.nan
    rows.append({"날짜": f"{d:%m-%d} ({'월화수목금'[d.weekday()]})", "이벤트": "" if ev[d] == "이벤트 없음" else ev[d],
                 "개장→10:00": f"{x.f_open:,.1f} → {x.f10:,.1f} ({x.r30 * 100:+.2f}%)",
                 "MNQ 5분 구름": "위" if x.c5 == 1 else ("아래" if x.c5 == -1 else "안"),
                 "신호": "매수" if bool(x.signal) else "쉼",
                 "매수": f"{k:.0f} 콜 @ {r['buys'][0][1]:.2f}" if r is not None else "",
                 "정산": f"{r['pay']:.2f}" if r is not None else "",
                 "손익": f"${r['pnl$']:+,.0f}" if r is not None else "",
                 "변형": ((f"손절 {(pd.Timestamp('2000-01-01 10:01') + pd.Timedelta(minutes=r9['stop'][0])):%H:%M} @{r9['stop'][1]:.2f} " if r9['stop'] else "") + f"${r9['pnl$']:+,.0f}") if r9 is not None else ""})
SEP = pd.DataFrame(rows)
sep_t = T[(T.index.year == 2026) & (T.index.month == 9)]
shutil.copy(OUT / "trade_book_ATM_단일_손절90_2026-09.pdf", P / "2026-09_매매_차트.pdf")

# 9월 매매 한눈에: 매매마다 SPX 1분 (매수 ▲, 행사가 점선)
cols = list(FP.columns); s0 = cols.index("10:00")
n = len(sep_t); nc = 4; nr = int(np.ceil(n / nc))
fig, axs = plt.subplots(nr, nc, figsize=(9, 2.3 * nr))
for ax, (d, r) in zip(np.ravel(axs), sep_t.iterrows()):
    f = FP.loc[d].values; k = S.loc[d, "K"]
    ax.plot(range(len(cols)), f, color=INK, lw=.8); ax.axhline(k, color=MUTED, ls=":", lw=.8)
    ax.scatter([s0], [f[s0]], marker="^", color=ACC, s=30, zorder=5)
    st9 = T9.loc[d, "stop"]
    if st9:
        ax.scatter([s0 + 1 + st9[0]], [f[s0 + 1 + st9[0]]], marker="v", color=LOSS, s=30, zorder=5)
    ax.set_title(f"{d:%m-%d}  {k:.0f} 콜 @{r['buys'][0][1]:.2f}  ${r['pnl$']:+,.0f}", fontsize=8,
                 color=ACC if r['pnl$'] > 0 else LOSS)
    ax.set_xticks([cols.index(c) for c in ("10:00", "12:00", "14:00", "16:00") if c in cols] or [], [c for c in ("10:00", "12:00", "14:00") if c in cols], fontsize=6)
    ax.tick_params(axis="y", labelsize=6); style(ax)
for ax in np.ravel(axs)[n:]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(P / "sep_trades.png", dpi=170); plt.close(fig)

# ── 통계 검증 (39_stats.py 결과) ──
ST = json.load(open(OUT / "stats" / "stats.json", encoding="utf-8"))
def money(v): return f"{'+' if v >= 0 else '−'}${abs(v) / 1000:,.1f}k"
for fimg in ("baseline.png", "random.png", "bootstrap.png", "grid.png"):
    shutil.copy(OUT / "stats" / fimg, P / f"st_{fimg}")
yrs = ("2022", "2023", "2024", "2025", "2026")
base_rows = "".join(
    f"<tr><td>{'<b>' + b['기준'] + '</b>' if '현재' in b['기준'] else b['기준']}</td><td>{b['매수 건수']}</td><td>{money(b['총손익$'])}</td>"
    f"<td>${b['건당 평균$']:,.0f}</td><td>{b['수익률%']:+.1f}%</td>" + "".join(f"<td class='{'pos' if b.get(y, 0) > 0 else 'neg'}'>{b.get(y, 0) / 1000:+.1f}</td>" for y in yrs) + "</tr>"
    for b in ST["baseline"])
fee_rows = "".join(f"<tr><td>${r['편도 수수료$']:.1f}</td><td>${r['만기 정산 수수료$']:.0f}</td><td>{money(r['총손익$'])}</td><td>${r['건당 평균$']:,.0f}</td>"
                   f"<td>{r['수익률%']:+.1f}%</td><td>−${abs(r['최대낙폭$']) / 1000:,.1f}k</td><td>{money(r['4/9 빼고$'])}</td></tr>" for r in ST["fees"])
RA, RB = ST["random"]["4/9 포함"], ST["random"]["4/9 제외"]
BA, BB = ST["bootstrap"]["4/9 포함"], ST["bootstrap"]["4/9 제외"]
GR = ST["grid"]
# ── 외부 비판 대응 (42_critique.py 결과) ──
CR = json.load(open(OUT / "critique" / "critique.json", encoding="utf-8"))
shutil.copy(OUT / "critique" / "momentum.png", P / "cr_momentum.png")
tim_rows = "".join(f"<tr><td>{k}</td><td>{money(v['총손익$'])}</td><td>{money(v['4/9 빼고$'])}</td><td>{v['수익률%']:+.1f}%</td>"
                   f"<td>−${abs(v['최대낙폭$']) / 1000:,.1f}k</td><td>{v['플러스 연도']}/5</td></tr>" for k, v in CR["timing"].items())
mom_rows = "".join(f"<tr><td>{'<b>' + k + '</b>' if '현재' in k else k}</td><td>{v['매수']}</td><td>{money(v['총손익$'])}</td><td>${v['건당$']:,.0f}</td>"
                   f"<td>{v['수익률%']:+.1f}%</td><td>{CR['momentum_overlap_with_cloud%'][k]:.0f}%</td></tr>" for k, v in CR["momentum"].items())
BT = CR["bootstrap"]
bt_rows = "".join(f"<tr><td>{lab}</td><td>{b['P(평균≤0)%']:.1f}%</td><td>${b['연환산 손익$'] / 1000:,.1f}k (${b['연환산 95%'][0] / 1000:,.1f}k ~ ${b['연환산 95%'][1] / 1000:,.1f}k)</td>"
                  f"<td>{b['샤프(연환산)']:.2f} ({b['샤프 95%'][0]:.2f} ~ {b['샤프 95%'][1]:.2f})</td><td>${b['중앙값$']:,.0f}</td>"
                  f"<td>−${abs(b['최대낙폭 중앙값$']) / 1000:,.1f}k / −${abs(b['최대낙폭 5% 최악$']) / 1000:,.1f}k</td></tr>" for lab, b in BT.items())
tail_rows = "<tr>" + "".join(f"<th>{k}</th>" for k in CR["tail"]) + "</tr><tr>" + "".join(
    f"<td class='{'pos' if v['총손익$'] > 0 else 'neg'}'>{money(v['총손익$'])}</td>" for v in CR["tail"].values()) + "</tr>"
reg_rows = "".join(f"<tr><td>{col}</td>" + "".join(f"<td>${d[k]['건당$']:,.0f} <span class='note'>({d[k]['구간']})</span></td>" for k in ("낮음", "중간", "높음")) + "</tr>"
                   for col, d in CR["regime"].items())
# ── 표본외 검증 (43·44) ──
OO = json.load(open(OUT / "oos" / "result.json", encoding="utf-8"))
shutil.copy(OUT / "oos" / "oos_equity.png", P / "oos_equity.png"); shutil.copy(OUT / "oos" / "oos_random.png", P / "oos_random.png")
oo_rows = "".join(f"<tr><td>{k}</td><td>{v['매수']}</td><td>{money(v['총손익$'])}</td><td>${v['건당$']:,.0f}</td><td>{v['승률%']:.0f}%</td>"
                  + "".join(f"<td class='{'pos' if v['연도별$'].get(y, 0) > 0 else 'neg'}'>{v['연도별$'].get(y, 0) / 1000:+.1f}</td>" for y in ("2020", "2021", "2022")) + "</tr>"
                  for k, v in {**OO["기준선"], "현재 규칙 (매도호가)": OO["현재 규칙 (매도호가)"], "1분 지연": OO["1분 지연 (09:59 판단 → 10:01 매수)"]}.items())
# ── HTML → PDF ──
yr = N["yearly"]
def money(v): return f"{'+' if v >= 0 else '−'}${abs(v) / 1000:,.1f}k"
sep_rows = "".join(
    f"<tr><td>{r['날짜']}</td><td>{r['이벤트']}</td><td>{r['개장→10:00']}</td><td>{r['MNQ 5분 구름']}</td>"
    f"<td class='{'buy' if r['신호'] == '매수' else 'rest'}'>{r['신호']}</td><td>{r['매수']}</td><td>{r['정산']}</td>"
    f"<td class='{'pos' if r['손익'].startswith('$+') else ('neg' if r['손익'] else '')}'>{r['손익']}</td><td class='note'>{r['변형']}</td></tr>" for r in rows)
def _w(r):
    return f"<tr><td>{r['구간'].replace(' 무손절', '')}</td><td>{r['대상']}</td><td>{r['요일']}</td><td>{r['건수']}</td><td>${r['건당$']:,.0f}</td><td>{r['t값']:.1f}</td><td class='{'pos' if r['총$'] > 0 else 'neg'}'>{money(r['총$'])}</td></tr>"
wd_rows = "".join(_w(r) for r in C2["weekday"] if r["구간"].endswith("무손절") and r["4/9"] == "포함" and r["대상"] == "현재 규칙") \
    + "".join(_w({**r, "대상": "규칙 (4/9 제외)"}) for r in C2["weekday"] if r["구간"].startswith("2022") and r["구간"].endswith("무손절") and r["4/9"] == "제외" and r["대상"] == "현재 규칙" and r["요일"] in ("수", "전체")) \
    + "".join(_w(r) for r in C2["weekday"] if r["구간"].endswith("무손절") and r["4/9"] == "포함" and r["대상"] == "매일 매수" and r["요일"] == "전체")
_rx = {(r["피처"], r["구간"]): r for r in C2["regime"] if r["4/9"] == "제외"}
rg_rows = "".join(f"<tr><td>{r['피처']}</td><td>{r['구간']}</td><td>{r['범위']}</td><td>${r['매일 매수 건당$']:,.0f}</td>"
                  + (f"<td>${r['규칙 건당$']:,.0f}</td><td class='{'pos' if r['조건부−무조건$'] > 0 else 'neg'}'>${r['조건부−무조건$']:+,.0f}</td><td>${_rx[(r['피처'], r['구간'])]['조건부−무조건$']:+,.0f}</td>" if r['규칙 건수'] else "<td>매수 없음</td><td></td><td></td>") + "</tr>"
                  for r in C2["regime"] if r["4/9"] == "포함")
pw_rows = "".join(f"<tr><td>${r['가정 엣지 $/건']}</td><td>{r['t=2 필요 건수']:,.0f}건</td><td>{list(v for k, v in r.items() if k.startswith('기간(년, 연'))[0]:.1f}년</td>"
                  f"<td>{list(v for k, v in r.items() if k.startswith('기간(년, 화·목'))[0]:.1f}년</td><td>{r['4/9 빼고 필요 건수']:,.0f}건</td></tr>" for r in C2["power"] if r["손절"] == "무손절")
xs_rows = "".join(f"<tr><td>{r['손절']}</td><td>{r['XSP 계약']}</td><td>{money(r['총손익$'])}</td><td>{money(r['실제 최대낙폭$'])} ({r['실제 %']:.0f}%)</td>"
                  f"<td>{money(r['재표본 중앙값$'])} ({r['중앙값 %']:.0f}%)</td><td>{money(r['최악 5%$'])} ({r['최악 5% %']:.0f}%)</td><td>{money(r['최악 1%$'])} ({r['최악 1% %']:.0f}%)</td></tr>" for r in C2["xsp"])
pt_rows = "".join(f"<tr><td>{r['가정 화·목 엣지']}</td><td>{r['통과 확률 t≥2 & 화·목>월수금']:.0f}%</td><td>{r['통과 확률 t≥1 & 화·목>월수금']:.0f}%</td></tr>" for r in C3["power_tt"]["rows"])
_we = pd.DataFrame(C3["weekday_events"]); _we = _we[_we["대상"] == "규칙"]
we_rows = "".join(f"<tr><td>{g}</td><td>{d}</td>" + "".join(
    (lambda q: f"<td>${q['건당$'].iloc[0]:,.0f} ({int(q['건수'].iloc[0])})</td>" if len(q) else "<td></td>")(_we[(_we['구간'] == g) & (_we['요일'] == d) & (_we['제외'] == e)])
    for e in ("전체", "이벤트일 제외", "이벤트일 제외 · 4/9 제외")) + "</tr>" for g in ("2022-05~2026-09", "표본외 2020-01~2022-05") for d in "월화수목금")
et_rows = "".join(f"<tr><td>{r['구간']}</td><td>{r['이벤트']}</td><td>{r['건수']}</td><td class='{'neg' if r['건당$'] < 0 else 'pos'}'>${r['건당$']:,.0f}</td></tr>" for r in C3["event_types"])
ro_rows = "".join(f"<tr><td>{r['피처']}</td><td>{r['구간']}</td><td>{r['범위']}</td><td>${r['매일 매수 건당$']:,.0f}</td>"
                  + (f"<td>${r['규칙 건당$']:,.0f} ({r['규칙 건수']})</td><td class='{'pos' if r['조건부−무조건$'] > 0 else 'neg'}'>${r['조건부−무조건$']:+,.0f}</td>" if r['규칙 건수'] else "<td>매수 없음</td><td></td>") + "</tr>"
                  for r in C3["regime_oos"])
xf_rows = "".join(f"<tr><td>{r['손절']}</td><td>{r['체결']}</td><td>${r['수수료 편도$']:.1f}</td><td>${r['정산$']:.0f}</td><td>{money(r['XSP 1계약 총손익$'])}</td>"
                  f"<td>{r['1계약 중앙값%']:.0f}% / {r['1계약 최악5%%']:.0f}%</td><td>{r['3계약 중앙값%']:.0f}% / {r['3계약 최악5%%']:.0f}%</td></tr>" for r in C3["xsp_fees"])
month_cells = "".join(f"<td class='{'pos' if v > 0 else 'neg'}'>{money(v)}</td>" for v in mm.values)
month_head = "".join(f"<th>{i}월</th>" for i in mm.index)

html = f"""
<h1>SPX 0DTE 옵션 매수 전략 — 10시 구름 콜</h1>
<p class="sub">1분 호가 1,093거래일 백테스트, 2026년 9월 매매 기록, 2020~2022 표본외 검증 · 작성 2026-09-27 (헤드라인 = 사전 등록 규칙, −90% 손절은 변형안으로 병기)</p><p><b>요약 판정: 2022-05 ~ 2026-09에서는 강한 성과였지만, 사전 등록한 2020~2022 표본외 검증을 통과하지 못했다 (9절). 실전 투입 보류, 앞으로 기록으로 계속 확인 중.</b></p>

<h2>1. 요약</h2>
<p>S&amp;P500 지수(SPX)의 당일 만기(0DTE) 옵션을 <b>매수만</b> 하는 전략이다. 개장 후 30분 동안 SPX가 올랐고, 나스닥100 마이크로 선물(MNQ) 5분봉이 일목균형표 구름 위에 있으면 뉴욕 시간 10:00에 등가격(ATM) 콜 1계약을 사서 만기까지 보유한다(<b>사전 등록 규칙</b>). 옵션 1분 호가로 매 거래를 재현한 결과, 2022-05 ~ 2026-09 동안 {MAIN['매수']}번 매매해 <b>총 {money(MAIN['총손익$'])}</b>(1계약 기준), 5개 연도 모두 플러스였다.</p>
<p><b>운용 변형안: −90% 손절</b> — 콜 값이 매수가의 10% 아래로 떨어지면 남은 값에 판다. 총 {money(VAR9['총손익$'])}. 단, 이 변형은 같은 기간 결과를 보고 고른 것이고, 2020~2022 표본외에서는 효과의 부호가 반대(손해)였다 (3·9절).</p>
<table class="kpi"><tr>
<td><b>{MAIN['매수']}건</b><br/>매매 (3거래일에 1번)</td><td><b>{money(MAIN['총손익$'])}</b><br/>총손익, 사전 등록 규칙 (60초 지연·매도호가 {money(V2['지연 60초 · 매도호가 · 손절 없음']['총손익$'])})</td>
<td><b>{MAIN['승률%']:.0f}%</b><br/>승률</td><td><b>{money(MAIN['최대낙폭$'])}</b><br/>최대낙폭</td><td><b>{money(MAIN['연도별$']['2026'])}</b><br/>2026년 (9/23까지)</td></tr></table>

<h2>2. 매매 규칙</h2>
<table>
<tr><th>판단·진입</th><td>뉴욕 10:00:00 (한국 23:00, 겨울 24:00). 판단에 쓰는 정보는 모두 10:00:00 이전</td></tr>
<tr><th>조건 1</th><td>SPX 10:00:00 가격 &gt; 개장가 (09:31)</td></tr>
<tr><th>조건 2</th><td>MNQ 5분봉(09:55~09:59 봉, 10:00:00 마감) 종가 &gt; 일목균형표 구름 윗선 (9·26·52)</td></tr>
<tr><th>매수</th><td>SPX 0DTE ATM 콜 1계약, 중간가 지정가</td></tr>
<tr><th>청산</th><td>만기 보유 (16:00 SPX 종가 현금정산). 손절·익절·물타기 없음 — <b>사전 등록 규칙</b></td></tr>
<tr><th>변형안</th><td>콜 중간가가 매수가의 10% 이하(−90%)가 되면 그 분 매수호가로 매도. 표본 내에서 고른 변형, 표본외에선 효과 부호 반대</td></tr>
<tr><th>안 하는 날</th><td>조건 중 하나라도 아니면 매매하지 않음 (풋도 사지 않음)</td></tr>
</table>

<h2>3. 백테스트 결과 (2022-05-16 ~ 2026-09-23)</h2>
<img src="equity.png" width="500"/>
<table>
<tr><th>연도</th><th>2022 (5월~)</th><th>2023</th><th>2024</th><th>2025</th><th>2026 (9/23까지)</th></tr>
<tr><td>손익 (1계약)</td>{''.join(f"<td class='pos'>{money(yr[k])}</td>" for k in ('2022', '2023', '2024', '2025', '2026'))}</tr>
<tr><td>매매 수</td>{''.join(f"<td>{N['yearly_n'][k]}</td>" for k in ('2022', '2023', '2024', '2025', '2026'))}</tr>
</table>
<table>
<tr><th>평균 투입 (1계약)</th><td>${N['avg_cost']:,}</td><th>프리미엄 대비 수익률</th><td>+{T.pnl.sum() / T.cost.sum() * 100:.1f}%</td></tr>
<tr><th>최악의 날</th><td>−${abs(N['worst']):,}</td><th>최고의 날</th><td>+${N['best']:,} (2025-04-09)</td></tr>
<tr><th>2025-04-09 제외 총손익</th><td>+${N['total_ex']:,}</td><th>매도호가로 샀다면</th><td>{money(V2['지연 0초 · 매도호가 · 손절 없음']['총손익$'])} (낙폭 {money(V2['지연 0초 · 매도호가 · 손절 없음']['최대낙폭$'])})</td></tr>
<tr><th>−90% 변형 총손익</th><td>{money(VAR9['총손익$'])} (낙폭 {money(VAR9['최대낙폭$'])})</td><th>변형 손절 비율</th><td>{VAR9['손절 비율%']:.0f}%</td></tr>
<tr><th>가장 긴 손실 구간</th><td>거래 {MAIN['가장 긴 손실 구간(거래)']}번 ({MAIN['손실 구간']})</td><th>2024 이전 / 이후 수익률</th><td>+{N['train']}% / +{N['val']}%</td></tr>
</table>

<h3>변형안: 손절선을 어디에 두나 (−10% ~ −95% 비교, 표본 내)</h3>
<img src="stop_sweep.png" width="500"/>
<table class="small"><tr><th>손절</th>{''.join(f"<th>{x}</th>" for x in SW['손절'])}</tr>
<tr><td>총손익</td>{''.join(f"<td class='{'pos' if v > 0 else 'neg'}'>{money(v)}</td>" for v in SW['총손익$'])}</tr>
<tr><td>4/9 제외</td>{''.join(f"<td>{money(v)}</td>" for v in SW['4/9 빼고$'])}</tr>
<tr><td>손절 비율</td>{''.join(f"<td>{v * 100:.0f}%</td>" if v > 0 else "<td>-</td>" for v in SW['손절 비율'])}</tr></table>
<p><b>−90%가 가장 좋지만 이유는 예측이 아니다.</b> 거래의 약 절반이 −90%까지 가고, 그중 만기까지 뒀으면 이익이었던 건 2%뿐이다. 어차피 0이 될 콜을 남은 10% 값에 파는 <b>잔존가치 회수</b> 효과다. −50%보다 촘촘하게 잡으면 큰 날을 잘라 성과가 단조롭게 무너진다 — 이 전략은 손절을 촘촘히 하면 안 되는 구조다. 단, 2020~2022 표본외(9절)에서는 −90% 손절이 오히려 약간 손해였다. 효과는 작고 기간에 따라 부호가 바뀐다.</p>
<h3>체결 지연 — API 주문이 늦어지면</h3>
<table class="small"><tr><th></th><th>무손절 · 중간가</th><th>무손절 · 매도호가</th><th>−90% · 중간가</th><th>−90% · 매도호가</th></tr>
<tr><td>지연 0초 (10:00:00 호가)</td>{''.join(f"<td>{money(V2[f'지연 0초 · {a} · {b}']['총손익$'])}</td>" for b in ('손절 없음', '손절 −90%') for a in ('중간가', '매도호가'))}</tr>
<tr><td>지연 60초 (10:01:00 호가)</td>{''.join(f"<td>{money(V2[f'지연 60초 · {a} · {b}']['총손익$'])}</td>" for b in ('손절 없음', '손절 −90%') for a in ('중간가', '매도호가'))}</tr></table>
<p>옵션 1분 호가의 "10:00" 값은 정확히 10:00:00 순간의 호가이고, 판단 재료(SPX 가격, MNQ 09:55 봉)도 10:00:00에 확정된다. 실제로는 판단 → 주문 → 체결까지 수 초가 걸리므로 성과는 0초와 60초 사이로 본다. 가장 보수적인 60초 지연·매도호가 체결에서도 사전 등록 규칙 {money(V2['지연 60초 · 매도호가 · 손절 없음']['총손익$'])}이다.</p>

<h2>4. 2026년 월별과 9월 매매 기록</h2>
<img src="monthly2026.png" width="500"/>
<table><tr>{month_head}</tr><tr>{month_cells}</tr></table>
<p>2026년 최대낙폭은 {p26.month}월 {p26.day}일 고점({money(e26[p26])})에서 {t26.month}월 {t26.day}일({money(e26[t26])})까지 {money(d26.min())}였고, {rec26}. 연초 원금 대비 최저는 {money(e26.min())}였다. 5월은 SPX가 5% 올랐지만 8번 모두 졌다 — 이 전략은 오르는 달이 아니라 하루 안에 크게 움직이는 날에 번다.</p>
<h3>2026년 9월 일별 기록 ({len(SEP)}거래일, 매수 {len(sep_t)}번, {int((sep_t.pnl > 0).sum())}승 {int((sep_t.pnl <= 0).sum())}패, 합계 {money(sep_t['pnl$'].sum())})</h3>
<img src="sep_trades.png" width="500"/>
<p class="note">매매별 SPX 1분 흐름 (▲ 10:00 매수, ▼ −90% 변형이었다면 손절한 곳, 점선 = 행사가). 매매별 상세(가격·정산·손익 상자)는 첨부 「2026-09_매매_차트.pdf」.</p>
<table class="small">
<tr><th>날짜</th><th>이벤트</th><th>SPX 개장 → 10:00</th><th>MNQ 5분 구름</th><th>판단</th><th>매수</th><th>정산</th><th>손익</th><th>−90% 변형</th></tr>
{sep_rows}
</table>
<p class="note">매수 가격은 10:00 중간가, 정산은 16:00 SPX 종가 기준 콜 가치 (사전 등록 규칙). 오른쪽 칸은 −90% 변형이었다면의 손절 시각·매도가·손익 (9월 합계 {money(T9['pnl$'][(T9.index.year == 2026) & (T9.index.month == 9)].sum())}). 매매별 장중 차트는 첨부 「2026-09_매매_차트.pdf」.</p>

<h2>5. 전체 기간 월별 (2022-05 ~ 2026-09)</h2>
<img src="monthly_all.png" width="500"/>
<table class="small">{mtab}</table>
<p class="note">단위: 천 달러 (SPX ATM 콜 1계약). 플러스 달 {n_pos} / {n_all}. 2025-04는 4/9 관세 유예 하루 +$37.1k 포함.</p>

<h2>6. 약세장이 오면? — 국면별 성과와 이유</h2>
<p>이 전략은 나스닥 강세에 기대는 것처럼 보이지만, 나스닥이 200일 이동평균 아래(약세)인 국면에서 오히려 더 벌었다. 옵션 데이터가 시작한 2022-05 ~ 10월이 나스닥 약세장 한가운데였고, 그 8개월 중 5개월이 플러스였다.</p>
<table class="small">
<tr><th>국면 (전날 나스닥 기준)</th><th>매매</th><th>수익률</th><th>총손익</th><th>건당 평균</th><th>4/9 제외 수익률</th></tr>
<tr><td>약세 (200일선 아래)</td><td>{RG['약세']['n']}</td><td class="pos">+{RG['약세']['ret']:.1f}%</td><td>{money(RG['약세']['usd'])}</td><td>${RG['약세']['per']:,.0f}</td><td class="pos">+{RGX['약세']['ret']:.1f}%</td></tr>
<tr><td>강세 (200일선 위)</td><td>{RG['강세']['n']}</td><td class="pos">+{RG['강세']['ret']:.1f}%</td><td>{money(RG['강세']['usd'])}</td><td>${RG['강세']['per']:,.0f}</td><td class="pos">+{RGX['강세']['ret']:.1f}%</td></tr>
</table>
<table class="small">
<tr><th>나스닥 고점 대비</th>{''.join(f"<th>{k}</th>" for k in DDT.index)}</tr>
<tr><td>매매 / 수익률</td>{''.join(f"<td>{n}건 / {r:+.0f}%</td>" for n, r in DDT.values)}</tr>
</table>
<table class="small">
<tr><th>2022년</th>{''.join(f"<th>{m}월</th>" for m in m22.index)}</tr>
<tr><td>전략 손익</td>{''.join(f"<td class='{'pos' if v > 0 else 'neg'}'>{money(v)}</td>" for v in m22.values)}</tr>
<tr><td>나스닥 월 수익률</td>{''.join(f"<td>{nqm.get(m, np.nan):+.1f}%</td>" for m in m22.index)}</tr>
</table>
<h3>왜 약세장에서 콜 매수가 더 버나 (신호 날만, 4/9 제외)</h3>
<table class="small">
<tr><th>항목</th><th>강세 국면</th><th>약세 국면</th><th>의미</th></tr>
<tr><td>콜 가격 (SPX 대비)</td><td>{RGX['강세']['prem']:.2f}%</td><td>{RGX['약세']['prem']:.2f}%</td><td rowspan="3">약세장에선 옵션값도 2배, 실제 움직임도 2배 → 옵션이 특별히 싼 건 아니다</td></tr>
<tr><td>10시 내재변동성</td><td>{RGX['강세']['iv']:.1f}%</td><td>{RGX['약세']['iv']:.1f}%</td></tr>
<tr><td>10시→종가 움직임 크기</td><td>{RGX['강세']['abs']:.2f}%</td><td>{RGX['약세']['abs']:.2f}%</td></tr>
<tr><td><b>오후에 오른 비율</b></td><td>{RGX['강세']['up']:.0f}%</td><td><b>{RGX['약세']['up']:.0f}%</b></td><td rowspan="2"><b>차이는 방향이다.</b> 약세장에서 아침 반등 + 나스닥 강세 신호가 나면 오후까지 이어진다</td></tr>
<tr><td><b>오후 평균 변화</b></td><td>{RGX['강세']['drift']:+.2f}%</td><td><b>{RGX['약세']['drift']:+.2f}%</b></td></tr>
<tr><td>2배 이상 / 전액 손실</td><td>{RGX['강세']['x2']:.0f}% / {RGX['강세']['zero']:.0f}%</td><td>{RGX['약세']['x2']:.0f}% / {RGX['약세']['zero']:.0f}%</td><td>이기는 날이 더 자주, 전액 잃는 날은 덜</td></tr>
</table>
<p><b>해석.</b> 약세장에서는 많은 참가자가 풋·공매도로 하락에 대비해 있다. 아침에 지수가 오르고 기술주가 앞장서면 그 헤지와 숏이 되감기면서(숏커버링) 반등이 오후까지 이어지기 쉽다. 반대로 잔잔한 강세장에서는 아침 강세가 그대로 유지될 힘이 약해 오후 방향이 거의 반반이고(평균 {RGX['강세']['drift']:+.2f}%), 콜은 시간가치만 녹는 날이 많다. 실제로 가장 긴 손실 구간(2024-02 ~ 12)과 2026년 5월 8연패는 모두 변동성이 낮은 강세장이었다.</p>
<p><b>남은 위험.</b> 반등 없이 계속 빠지는 장(예: 2000~2002 닷컴 붕괴)은 데이터에 없다. 그런 장에서는 "개장 후 상승 + 나스닥 5분봉 구름 위" 신호 자체가 드물어 매매가 줄어든다. 가장 불리한 환경은 약세장이 아니라 <b>변동성 없이 천천히 오르는 강세장이 길게 이어지는 경우</b>다.</p>

<h2>7. 통계 검증 — 운이 아닌가?</h2>
<p class="note">7~9절의 검증은 모두 사전 등록 규칙(무손절, 만기 보유) 기준이다. −90% 변형의 표본외 결과는 9절 끝에 따로 적었다.</p>
<h3>① 기준선: 조건을 하나씩 빼면 (10:00 ATM 콜 1계약, 만기)</h3>
<table class="small"><tr><th>기준</th><th>매수</th><th>총손익</th><th>건당 평균</th><th>수익률</th><th>2022</th><th>2023</th><th>2024</th><th>2025</th><th>2026</th></tr>{base_rows}</table>
<img src="st_baseline.png" width="500"/>
<p>매일 사면 거의 본전(수익률 +1%)이다. 두 조건 모두 각각 성과를 끌어올린다. MNQ 5분 구름 조건만 써도 총손익은 현재 규칙과 비슷하지만(525건), 두 조건을 같이 쓰면 매매 수가 줄고 건당 평균이 가장 높으며 5개 연도가 모두 플러스다.</p>
<h3>② 무작위 대조: 같은 수의 날을 무작위로 골랐다면 (10,000회)</h3>
<img src="st_random.png" width="500"/>
<table class="small"><tr><th></th><th>현재 규칙</th><th>무작위 중앙값</th><th>무작위 상위 5% 선</th><th>무작위가 현재 규칙 이상인 비율</th></tr>
<tr><td>4/9 포함 ({RA['뽑은 날 수']}일)</td><td>{money(RA['실제'])}</td><td>{money(RA['무작위 중앙값'])}</td><td>{money(RA['무작위 95% 상단'])}</td><td><b>{RA['무작위가 실제 이상인 비율%']:.1f}%</b></td></tr>
<tr><td>4/9 제외 ({RB['뽑은 날 수']}일)</td><td>{money(RB['실제'])}</td><td>{money(RB['무작위 중앙값'])}</td><td>{money(RB['무작위 95% 상단'])}</td><td><b>{RB['무작위가 실제 이상인 비율%']:.1f}%</b></td></tr></table>
<p>무작위로 날을 골라서는 현재 규칙만큼 벌 확률이 1% 안팎이다. 날짜를 고르는 조건에 정보가 있다는 뜻이다.</p>
<h3>③ 부트스트랩: 건당 평균 손익의 95% 신뢰구간 (10,000회 재표본)</h3>
<img src="st_bootstrap.png" width="500"/>
<table class="small"><tr><th></th><th>건당 평균</th><th>95% 구간</th><th>평균이 0보다 클 확률</th></tr>
<tr><td>4/9 포함</td><td>${BA['건당 평균$']:,.0f}</td><td>${BA['95% 하한$']:,.0f} ~ ${BA['95% 상한$']:,.0f}</td><td>{BA['평균>0 확률%']:.1f}%</td></tr>
<tr><td>4/9 제외</td><td>${BB['건당 평균$']:,.0f}</td><td>${BB['95% 하한$']:,.0f} ~ ${BB['95% 상한$']:,.0f}</td><td>{BB['평균>0 확률%']:.1f}%</td></tr></table>
<p>4/9를 빼면 95% 구간이 0을 살짝 포함한다. 건당 손익의 흔들림이 커서(대부분 소액 손실, 가끔 큰 이익) 394건으로는 "확실한 플러스"라고 단정하기 어렵다. 앞으로의 실거래 기록으로 표본을 늘려 확인해야 한다.</p>
<h3>④ 225개 조합 중 현재 규칙의 위치</h3>
<img src="st_grid.png" width="500"/>
<p>진입 시각 5 × 행사가 5 × 물타기 3 × 신호 3 = 225개 조합 중 <b>{GR['플러스 조합%']:.0f}%가 플러스</b>(4/9 빼면 {GR['플러스 조합 (4/9 빼고)%']:.0f}%)다. 현재 규칙은 수익률로는 {GR['수익률 순위']}위(중앙값 근처)지만 위험 대비 효율(손익÷낙폭)로는 {GR['점수 순위']}위다. 가장 수익률이 높은 칸을 고른 것이 아니라, 좋은 구역 안에서 단순하고 낙폭이 작은 조합을 고정했다.</p>
<h3>⑤ 수수료 민감도</h3>
<table class="small"><tr><th>편도 수수료</th><th>만기 정산 수수료</th><th>총손익</th><th>건당 평균</th><th>수익률</th><th>최대낙폭</th><th>4/9 제외</th></tr>{fee_rows}</table>
<p>수수료가 가장 비싼 경우(편도 $8, 정산 $12)에도 총손익 감소는 약 5%다. 1계약 평균 $1,348짜리 옵션이라 수수료 비중이 작다.</p>

<h2>8. 외부 비판에 대한 추가 검증</h2>
<p>다른 AI(ChatGPT)에 비판적 검토를 맡겨 받은 지적 중 새로 확인할 수 있는 것을 모두 돌렸다. 지적 가운데 기준선 비교·매도호가 체결·수수료 민감도·225개 조합 분포·걸어가며 검증은 7절과 앞 절에 이미 있다.</p>
<h3>① 진입 타이밍 — 판단 시점과 체결 지연</h3>
<p>지적: "10:00 데이터로 판단하고 10:00에 사는 건 시점이 맞지 않는다." 확인 결과 옵션 1분 호가(cbbo-1m)의 "10:00" 값은 <b>정확히 10:00:00.000 순간의 호가</b>(09:59 분의 마지막 호가)이고, MNQ 09:55 5분봉도 10:00:00에 마감한다. 따라서 판단 재료에 미래 정보는 없다. 남는 문제는 판단부터 체결까지의 지연뿐이라, 체결을 10:00:00(지연 0초)과 10:01:00(지연 60초) 호가로 각각 계산해 3절에 실었다 (−90% 손절 기준 {money(V2['지연 0초 · 중간가 · 손절 −90%']['총손익$'])} → {money(V2['지연 60초 · 중간가 · 손절 −90%']['총손익$'])}). 예전 판에서 "09:59 판단"으로 계산한 수치는 SPX 가격을 1분 묵은 09:59:00 호가로 판단한 것이라 필요 이상으로 보수적이었다.</p>
<h3>② 일목 구름 vs 단순 모멘텀 (모두 SPX 30분 상승 조건 포함)</h3>
<img src="cr_momentum.png" width="500"/>
<table class="small"><tr><th>두 번째 조건</th><th>매수</th><th>총손익</th><th>건당</th><th>수익률</th><th>구름 신호와 겹침</th></tr>{mom_rows}</table>
<p>"구름 위"는 단순히 "올랐다"와 같지 않다. 30분 상승, 전일 종가 위, 이동평균 위, 마지막 5분 양봉 등 단순 모멘텀 조건은 모두 구름보다 약했고, 일부는 두 번째 조건이 없는 경우보다도 못했다.</p>
<h3>③ 부트스트랩 확장 (매매 단위 10,000회 재표본)</h3>
<table class="small"><tr><th></th><th>평균≤0 확률</th><th>연환산 손익 (95%)</th><th>샤프 (95%)</th><th>건당 중앙값</th><th>최대낙폭 중앙값 / 최악 5%</th></tr>{bt_rows}</table>
<p>보통의 거래(중앙값)는 약 −$600으로 진다. 같은 거래를 다른 순서로 겪으면 최대낙폭이 중앙값 약 −$100k, 나쁜 경우 −$41k 이상으로, 실제 기록된 −$13.5k보다 크다. 계좌 규모 대비 1계약의 위험을 과소평가하면 안 된다.</p>
<h3>④ 상위 거래 제거</h3>
<table class="small">{tail_rows}</table>
<p>394건 중 상위 약 20건(5%)이 수익 전부를 만든다. 옵션 매수 전략의 구조상 자연스러운 모양이지만, 앞으로 이런 날을 몇 번 만나느냐에 성과가 크게 좌우된다.</p>
<h3>⑤ 국면 구간별 건당 손익 (3분위)</h3>
<table class="small"><tr><th>구분</th><th>낮음</th><th>중간</th><th>높음</th></tr>{reg_rows}</table>
<p>10시 내재변동성이 높고, 전날 대비 오르고, 개장 30분 상승폭이 큰 날에 수익이 몰린다. 전날 실현변동성이 낮은 날은 본전이다. "10시까지의 강한 위험선호 → 오후 방향성 → 0DTE 볼록성"이라는 메커니즘과 일치한다.</p>
<h3>⑥ 남은 한계와 진짜 표본외 검증 계획</h3>
<ul>
<li>규칙(MNQ·구름·10시)을 2022-05 ~ 2026-09 전체를 보면서 골랐기 때문에, 이 기간 안의 어떤 검증도 연구자의 선택 과정까지 지우지는 못한다. 프로젝트 전체에서 약 1,500개 이상의 조합을 봤고, 위의 통계 수치는 이 탐색을 보정하지 않은 값이다.</li>
<li>진짜 표본외 1: <b>2026-09-24부터 매일 자동으로 쌓는 앞으로 기록</b> (규칙 고정, 실시간 엔진과 같은 코드).</li>
<li>진짜 표본외 2: <b>2020-01 ~ 2022-05 월·수·금 0DTE</b> — 9절에서 시행, <b>불통과</b>.</li>
</ul>

<h2>8-2. 두 번째 외부 검토(Claude)에 대한 추가 분석</h2>
<h3>① 요일별 분해 — 사전 등록 규칙 vs 매일 매수 (무손절)</h3>
<img src="c2_weekday.png" width="500"/>
<table class="small"><tr><th>구간</th><th>대상</th><th>요일</th><th>건수</th><th>건당</th><th>t값</th><th>총손익</th></tr>{wd_rows}</table>
<p>2022-05 ~ 2026-09에서 규칙의 이익은 <b>화·목</b>에 몰려 있다 (화·목 139건 건당 $394, t = 2.29, 5개 연도 모두 플러스 / 월·수·금은 4/9를 빼면 건당 $31, t = 0.28). 2020 ~ 2022 표본외는 월·수·금 만기만 있던 시기라, 불통과는 "화·목이 없었기 때문"일 수 있다. 반대로 요일 5개 중 좋은 둘을 사후에 고른 것일 수도 있다 — 표본외에선 월요일이 가장 좋고 수요일이 가장 나빠, 요일 순서 자체는 기간마다 흔들린다. 그래서 규칙은 바꾸지 않고 <b>"화·목 가설"을 사전 등록</b>해 앞으로 기록으로 판정한다 (④).</p>
<h3>② 국면 표 — 조건부(규칙) − 무조건(매일 매수), 같은 3분위 경계</h3>
<table class="small"><tr><th>피처</th><th>구간</th><th>범위</th><th>매일 매수 건당</th><th>규칙 건당</th><th>차이</th><th>차이 (4/9 제외)</th></tr>{rg_rows}</table>
<p>모든 국면 칸에서 규칙이 매일 매수보다 건당 손익이 높다 (개장 30분 하락 칸은 규칙상 매수 없음). 즉 규칙의 성과는 "변동성 큰 장세였기 때문"만은 아니고, 같은 장세 안에서도 조건이 날을 더 잘 고른다. 차이는 IV가 높거나 전날보다 오른 날, 밤사이 하락 갭 날에 가장 크다.</p>
<h3>④ 사전 등록: 화·목 가설 (<code>output/사전등록_화목가설.md</code>)</h3>
<p>2026-09-24 이후 앞으로 기록만으로 판정. 화·목 신호일 <b>106건</b>(연 약 {C2['n_year_tt']:.0f}건 → 약 3.3년) 시점에 ① 화·목 건당 &gt; 0, t ≥ 2 ② 화·목 건당 &gt; 월·수·금 건당이면 통과. 중간 점검(32·64건)은 기록만, 조기 판정 없음. 앞으로 기록에 요일 칸을 추가했다.</p>
<h3>⑤ 검정력 — 엣지가 이 정도면 확인에 몇 건이 필요한가 (무손절, t = 2 기준)</h3>
<table class="small"><tr><th>가정 엣지 (건당)</th><th>필요 매매 수</th><th>기간 (전체 신호, 연 {C2['n_year']:.0f}건)</th><th>기간 (화·목만, 연 {C2['n_year_tt']:.0f}건)</th><th>필요 수 (4/9 제외 분산)</th></tr>{pw_rows}</table>
<p>건당 표준편차가 약 $2,600(4/9 제외 $1,900)이라, 건당 $100 엣지를 확인하려면 수십 년이 필요하다. 백테스트 평균($253)만큼이면 약 4년. 앞으로 기록은 오래 쌓아야 의미가 있다.</p>
<h3>⑥ XSP(1/10 크기) 1·3계약 기준 최대낙폭 — 계좌 $100k 대비 (10,000회 재표본)</h3>
<table class="small"><tr><th>규칙</th><th>XSP 계약</th><th>총손익</th><th>실제 최대낙폭</th><th>재표본 중앙값</th><th>최악 5%</th><th>최악 1%</th></tr>{xs_rows}</table>
<p class="note">XSP 가격 = SPX의 1/10, 수수료 계약당 편도 $2.5(정산 포함)로 계산. XSP는 SPX보다 호가 간격이 상대적으로 넓어 실제 비용은 더 클 수 있다 (미반영).</p>

<h2>8-3. 세 번째 외부 검토에 대한 추가 분석</h2>
<h3>① 화·목 가설 판정 기준의 검정력 (106건, 부트스트랩 2만 회)</h3>
<table class="small"><tr><th>가정한 진짜 화·목 엣지</th><th>통과 확률 (t ≥ 2 &amp; 화·목 &gt; 월·수·금)</th><th>기준을 t ≥ 1로 낮추면</th></tr>{pt_rows}</table>
<p>표본 내 추정치($394)가 진짜여도 106건 판정의 통과 확률은 약 50%다. 판정 기준은 그대로 두되, 불통과를 "엣지 없음"으로 읽지 않는다. t ≥ 1 기준은 엣지가 없을 때도 15%가 통과하므로 보조 지표로만 기록한다. 80% 검정력에는 약 208건(약 6.5년)이 필요하다.</p>
<h3>② 요일 × 이벤트일(FOMC·CPI·고용) 제외 — 규칙 건당 손익 (괄호 = 건수, 무손절)</h3>
<table class="small"><tr><th>구간</th><th>요일</th><th>전체</th><th>이벤트일 제외</th><th>이벤트일·4/9 제외</th></tr>{we_rows}</table>
<table class="small"><tr><th>구간</th><th>이벤트</th><th>규칙 매수</th><th>건당</th></tr>{et_rows}</table>
<p><b>이벤트일 매수는 두 기간 모두 손실이다</b> (2022-05~ 49건 건당 −$455, 표본외 20건도 전부 음수). 이벤트일을 빼면 표본 내 모든 요일의 건당이 오른다. 수요일의 약세 일부는 FOMC·CPI가 수요일에 몰린 탓이지만, 4/9를 빼면 이벤트를 제외해도 수요일은 여전히 약하다(표본 내 −$132, 표본외 −$287). 이 역시 사후 발견이라 "가설 2"로 사전 등록했다 (이벤트일 신호 50건, 약 4.5년 뒤 판정).</p>
<h3>③ 국면 표를 표본외(2020-01 ~ 2022-05)로 — 조건부 − 무조건</h3>
<table class="small"><tr><th>피처</th><th>구간</th><th>범위</th><th>매일 매수 건당</th><th>규칙 건당 (건수)</th><th>차이</th></tr>{ro_rows}</table>
<p class="note">표본외는 월·수·금 위주라 "IV 전일 대비"는 직전 만기일 대비. 전일 실현변동성은 ES 1분봉으로 계산.</p>
<p><b>표본외에서는 방향이 뒤집힌다.</b> 표본 내에선 IV가 높고 변동성이 큰 칸에서 규칙의 우위가 가장 컸지만, 2020~2022에선 바로 그 칸(10시 IV 높음 −$329, IV 상승 −$434, 전일 변동성 높음 −$510)에서 규칙이 매일 매수보다 못했다. 규칙의 우위는 잔잔한 칸에서만 남았다. 코로나 급락·급반등 같은 고변동 장에서는 "개장 30분 상승 + 나스닥 구름 위" 조건이 오히려 나쁜 날을 골랐다 — 표본외 불통과의 한 원인으로 보인다.</p>
<h3>④ XSP 비용 민감도 — 수수료·정산비·체결가 (규칙 신호일, 계좌 $100k 대비 재표본 최대낙폭: 중앙값 / 최악 5%)</h3>
<table class="small"><tr><th>손절</th><th>체결</th><th>편도 수수료</th><th>정산 비용</th><th>XSP 1계약 총손익 (4.3년)</th><th>1계약 낙폭</th><th>3계약 낙폭</th></tr>{xf_rows}</table>
<p>XSP 1계약의 4.3년 총손익은 가장 싼 경우 약 $9k, 가장 비싼 경우(편도 $8·정산 $8·매도호가 체결) 약 $4k — 1년에 $1k 안팎이다. 3계약이면 비용이 비싼 경우 최악 5% 낙폭이 계좌의 80~90%에 이른다. NH 실제 수수료 확인 전에는 XSP 소액으로도 기대값이 작다.</p>
<h3>종합 판정</h3>
<p>좋은 백테스트 → 사전 등록 표본외 불통과 → 원인 분해(요일·이벤트·국면) → 새 가설 사전 등록까지 진행했다. 표본 내에서 조건의 가치는 여러 방식으로 확인되지만 표본외에서는 재현되지 않았고, 남은 가설(화·목, 이벤트일)은 모두 사후 발견이라 앞으로 기록으로 판정하는 데 수년이 걸린다. <b>실전은 보류한다. 한다면 수수료를 확인한 뒤, 체결 품질 검증 목적의 최소 크기(XSP 1계약)로만 한다.</b></p>

<h2>9. 진짜 표본외 검증 (2020-01 ~ 2022-05) — 불통과</h2>
<p>규칙을 한 글자도 바꾸지 않고, 이 프로젝트에서 한 번도 보지 않은 2020-01 ~ 2022-05의 SPXW 0DTE(당일 만기가 월·수·금만 있던 시기, 378일)에 한 번 적용했다. 판정 기준은 결과를 보기 전에 파일로 등록했다 (<code>output/oos/사전등록_판정기준.md</code>).</p>
<table class="small"><tr><th>판정 기준</th><th>결과</th></tr>
<tr><td>① 총손익 &gt; 0</td><td class="pos">통과 ({money(OO['현재 규칙 (중간가)']['총손익$'])})</td></tr>
<tr><td>② 건당 평균 &gt; 0</td><td class="pos">통과 (${OO['현재 규칙 (중간가)']['건당$']:,.0f})</td></tr>
<tr><td>③ 무작위 대비 상위 10%</td><td class="neg"><b>불통과</b> — 무작위로 고른 날의 {OO['무작위 대조']['무작위가 실제 이상인 비율%']:.0f}%가 현재 규칙 이상</td></tr>
<tr><td><b>판정</b></td><td class="neg"><b>불통과</b></td></tr></table>
<img src="oos_equity.png" width="500"/>
<table class="small"><tr><th>기준</th><th>매수</th><th>총손익</th><th>건당</th><th>승률</th><th>2020</th><th>2021</th><th>2022(~5월)</th></tr>{oo_rows}</table>
<img src="oos_random.png" width="500"/>
<p><b>해석.</b> 이 기간에는 조건 없이 매일 사는 것(+$20.6k)이 현재 규칙(+$5.6k)보다 나았고, MNQ 5분 구름 조건만 쓰면 손실이었다. 특히 2020년 3월 코로나 급락 뒤 급반등 날들은 MNQ가 구름 아래여서 신호가 나지 않았다. 건당 평균의 95% 구간은 ${OO['부트스트랩']['95% 구간'][0]:,.0f} ~ ${OO['부트스트랩']['95% 구간'][1]:,.0f}로 0을 크게 포함한다. 2022-05 ~ 2026-09에서 보인 성과는 상당 부분 그 기간에 맞춰 고른 결과일 가능성이 높다.</p>
<p><b>−90% 손절을 넣으면 (같은 2020-01 ~ 2022-05, 지연 0초·중간가):</b> 전체 {RV['oos']['지연 0초 · 손절 −90% · 전체']['매수']}건 {money(RV['oos']['지연 0초 · 손절 −90% · 전체']['총손익$'])} (손절 없음 {money(RV['oos']['지연 0초 · 손절 없음 · 전체']['총손익$'])}), 월·수·금 만기만 {RV['oos']['지연 0초 · 손절 −90% · 월·수·금만']['매수']}건 {money(RV['oos']['지연 0초 · 손절 −90% · 월·수·금만']['총손익$'])} (손절 없음 {money(RV['oos']['지연 0초 · 손절 없음 · 월·수·금만']['총손익$'])}). 이 기간엔 손절이 오히려 손해였다 — 2022년 급락 후 오후에 반등한 날들을 잘랐다. 표본외 매수일 중 화·목 11일은 월요일 휴일로 밀린 만기·월말 만기 등 그날 만기가 있던 날이다.</p>
<p><b>결정.</b> 사전 약속대로 규칙은 고치지 않는다. 실전 투입은 보류하고, 2026-09-24부터 쌓는 앞으로 기록만 계속한다. 이 결과는 전략의 한계이자, 검증 절차가 제대로 작동했다는 증거다.</p>

<h2>10. 연구 과정 — 검증하고 버린 것</h2>
<table class="small">
<tr><th>시도</th><th>결과</th></tr>
<tr><td>매일 무조건 ATM 스트래들 매수 (7개 시각)</td><td>모든 시각 손실 (10:00 연 −$25k). 옵션 가격이 실제 움직임보다 비쌈</td></tr>
<tr><td>손절·익절 규칙 (112개 조합)</td><td>손실을 줄일 뿐 플러스로 못 바꿈. 익절은 오히려 악화</td></tr>
<tr><td>장중 모멘텀 (Gao 외 2018, 첫 30분 → 마지막 30분)</td><td>2022~2026엔 효과 없음 (적중 50.7%, p=0.34)</td></tr>
<tr><td>이벤트일 (FOMC·CPI·고용)</td><td>CPI 날은 옵션이 10~20% 비쌈</td></tr>
<tr><td>일목 구름 터치 매매 (SPX·MNQ, 144개 조합)</td><td>추세 방향 되돌림 반등 &gt; 돌파 기대. 터치만으론 수익 부족</td></tr>
<tr><td>15시 이후 역매매</td><td>−15%, 5개 연도 모두 손실 → 폐기</td></tr>
<tr><td>물타기·손절 조합</td><td>위험 대비 효율 개선 없음. 손절 붙이면 큰 날을 잘라 최악</td></tr>
<tr><td>단일 콜 손절선 −10% ~ −95% (11개)</td><td>−90%만 소폭 개선(잔존가치 회수), −50%보다 촘촘하면 단조 악화 → −90% 채택</td></tr>
<tr><td>MNQ 대신 ES 신호</td><td>ES도 플러스지만 약함. MNQ 구름 = "기술주 주도 상승"을 걸러줌</td></tr>
<tr><td>진입 5 × 행사가 5 × 물타기 3 × 신호 3 = 225개 + 걸어가며 검증</td><td>매년 최적 조합을 바꾸는 것보다 규칙 고정이 더 벌었음 → 규칙 고정</td></tr>
</table>

<h2>11. 한계와 위험</h2>
<ul>
<li>수익의 상당 부분이 1년에 몇 번 오는 큰 날에서 나온다 (2026년 상위 3건이 수익의 85%).</li>
<li>여러 조합을 시험해 고른 규칙이라 실제 성과는 백테스트보다 낮을 수 있다.</li>
<li>중간가 체결을 가정했다. 실제 체결 품질은 모의·실거래로 확인해야 한다.</li>
<li>SPX 1계약 최대낙폭은 실제 {money(MAIN['최대낙폭$'])}였지만, 거래 순서를 바꿔 재표본하면 중앙값 약 −$100k, 나쁜 경우 −$41k다 (계좌 $100k 대비 100% 이상). XSP 1계약이면 최악 5%가 계좌의 약 20%, 3계약이면 약 60% (8-2절 ⑥). 8연패 구간도 실제로 있었다.</li>
<li>이익이 화·목에 몰려 있는데, 이는 사후 분할이라 앞으로 기록으로 확인해야 한다 (8-2절 ①④).</li>
</ul>

<h2>12. NH선물 REST API로 구현할 계획</h2>
<table class="small">
<tr><th>단계</th><th>내용</th></tr>
<tr><td>시세</td><td>웹소켓 실시간 시세로 MNQ 1분봉 수집 → 5분봉·일목균형표 계산, SPX 0DTE 옵션 호가 수집</td></tr>
<tr><td>신호</td><td>매일 뉴욕 10:00에 두 조건 자동 판정, 판정 근거(개장가·10시 가격·구름 윗선) 로그 저장</td></tr>
<tr><td>주문</td><td>ATM 행사가 자동 선택 → 중간가 지정가 주문 → 일정 시간 미체결 시 호가 단계적 상향</td></tr>
<tr><td>손절</td><td>체결 후 매분 콜 중간가 감시 → 매수가의 10% 이하면 매도 주문</td></tr>
<tr><td>안전장치</td><td>모의/실거래 스위치 (기본 모의), 하루 1회 주문 제한, 주문·체결 기록</td></tr>
<tr><td>검증</td><td>모의투자로 선행 기록 → 신호 일치율·체결가와 백테스트 비교 → 사용자 승인 후 실거래</td></tr>
</table>

<h2>13. 데이터와 방법</h2>
<ul>
<li>옵션: SPXW 당일 만기 1분 최우선 호가 (Databento OPRA, 2022-05-16 ~ 2026-09-23, 1,093거래일)</li>
<li>선물: MNQ·ES 1분봉 연결선물 (Databento CME, 월물 교체 가격 보정), SPX 일봉 종가 (Investing.com)</li>
<li>체결: 10:00:00 옵션 중간가 매수, −90% 손절은 그 분 매수호가 매도, 아니면 16:00 SPX 종가 현금정산. 수수료 계약당 편도 $2.5 (정산 포함)</li>
<li>신호는 10:00:00까지 확정된 정보만 사용 (옵션 cbbo-1m "10:00" = 10:00:00.000 스냅샷, MNQ 09:55 봉 10:00:00 마감) — 미래 정보 없음</li>
</ul>
<p class="note">본 자료는 과거 데이터 백테스트이며 미래 수익을 보장하지 않는다.</p>
"""
css = """
@font-face { font-family: kr; src: url(malgun.ttf); }
@font-face { font-family: kr; font-weight: bold; src: url(malgunbd.ttf); }
* { font-family: kr; }
body { font-size: 9.5pt; color: #17201C; line-height: 1.45; }
h1 { font-size: 19pt; margin-bottom: 2pt; }
.sub { color: #6B7570; margin-top: 0; }
h2 { font-size: 13pt; color: #0F6E5A; margin-top: 14pt; margin-bottom: 4pt; border-bottom: 1px solid #D8DED9; }
h3 { font-size: 10.5pt; margin-top: 8pt; margin-bottom: 3pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0; }
th, td { border: 1px solid #D8DED9; padding: 3pt 5pt; text-align: left; }
th { font-weight: bold; color: #0F6E5A; }
table.kpi td { text-align: center; font-size: 9pt; }
table.small th, table.small td { font-size: 8pt; padding: 2pt 4pt; }
.pos { color: #0F6E5A; } .neg { color: #B3261E; } .buy { color: #0F6E5A; font-weight: bold; } .rest { color: #6B7570; }
.note { color: #6B7570; font-size: 8pt; }
img { margin: 4pt 0; }
"""
for f in ("malgun.ttf", "malgunbd.ttf"):
    shutil.copy(pathlib.Path(r"C:\Windows\Fonts") / f, P / f)
arch = fitz.Archive(str(P))
story = fitz.Story(html=html, user_css=css, archive=arch)
out = P / "SPX0DTE_전략_포트폴리오.pdf"
writer = fitz.DocumentWriter(str(out))
mediabox = fitz.paper_rect("a4"); where = mediabox + (42, 42, -42, -42)
more = 1
while more:
    dev = writer.begin_page(mediabox)
    more, _ = story.place(where)
    story.draw(dev)
    writer.end_page()
writer.close()
_d = fitz.open(out); _d.subset_fonts(); _tmp = out.with_suffix(".tmp.pdf")          # 글꼴 통째 포함 → 쓴 글자만 (65MB → 1~2MB)
_d.save(_tmp, garbage=4, deflate=True, clean=True); _d.close()
try:
    _tmp.replace(out)
except PermissionError:                                                           # PDF 뷰어에서 열려 있으면 다른 이름으로
    out = out.with_name(out.stem + f"_{pd.Timestamp.now():%m%d_%H%M}.pdf"); _tmp.replace(out)
for f in ("malgun.ttf", "malgunbd.ttf"):
    (P / f).unlink()
print("저장:", out, "쪽수:", len(fitz.open(out)))
print(SEP.to_string(index=False))
