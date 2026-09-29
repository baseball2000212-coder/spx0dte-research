"""
포트폴리오 PDF v2 (2026-09-29 사용자 지시): 매도호가 매수 + 체결 지연 5초·10초, −90% 손절, 수수료 편도 $5 + 정산 $5 가정,
CME 시세료 월 $228.80 — 모든 숫자에 기본 반영 (표마다 "빼기 전" 같은 단서 달지 말 것, 사용자 요청).
구성: 요약 / 매매 규칙 / 백테스트 분석 / 2025-04-09 / 월별 / 위험 / 요일별 분석(유의성·이유) / 한계 / 데이터.
결과: output/portfolio/SPX0DTE_네이키드_매수전략.pdf
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, shutil, subprocess, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import fitz
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from scipy import stats as sps
from spx0dte.events import label_events
from spx0dte import realistic as RL

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
P = OUT / "portfolio"; P.mkdir(exist_ok=True)
INK, INK2, BLUE, ORANGE, GRID = "#0b0b0b", "#5c5b55", "#2a78d6", "#eb6834", "#e6e5e0"
D49 = pd.Timestamp("2025-04-09")
DOW = "월화수목금"


def money(v):
    return f"{'+' if v >= 0 else '−'}${abs(v) / 1000:,.1f}k"


def usd(v):
    return f"{'+' if v >= 0 else '−'}${abs(v):,.0f}"


def cls(v):
    return "pos" if v > 0 else "neg"


# ── 데이터 ──
TK = RL.load_ticks(); R = RL.ratios(TK)
LA = RL.load_legs(signal_only=False); LS = LA[LA.signal]
TR = {k: RL.trades(LS, k, TK, R) for k in RL.DELAYS}
FW = {k: RL.forward(k, R) for k in RL.DELAYS}
FW = {k: v[v.index > LS.index.max()] for k, v in FW.items()}
ALL = {k: pd.concat([TR[k], FW[k]]).sort_index() for k in RL.DELAYS}
first, last = LS.index.min(), max(v.index.max() for v in ALL.values())
months = (last.year - first.year) * 12 + last.month - first.month + 1
yrs = (last - first).days / 365.25

# 체결 가정 비교 (−90% 손절 / 손절 없음, CME 반영)
comp = []
for nm, sec in (("10:00:00 매도호가 (지연 없음)", 0), ("5초 뒤 매도호가", 5), ("10초 뒤 매도호가", 10)):
    a = RL.trades(LS, sec, TK, R)["손익$"].sum() - months * RL.CME; b = RL.trades(LS, sec, TK, R, stop=None)["손익$"].sum() - months * RL.CME
    comp.append((nm, a, b))
MONTHS = pd.period_range(first, last, freq="M")

# 핵심 지표
K = {}
for k in RL.DELAYS:
    t = ALL[k]; p = t["손익$"]; c = RL.net_cum(p, first); mdd, pk, tr = RL.max_dd(c)
    under = (c < c.cummax()).values; run = best = e = 0
    for i, u in enumerate(under):
        run = run + 1 if u else 0
        if run > best:
            best, e = run, i
    s0, s1 = c.index[e - best + 1], c.index[e]
    top3 = p.drop(p.nlargest(3).index).sum()
    K[k] = dict(n=len(p), gross=p.sum(), cme=months * RL.CME, net=c.iloc[-1], per_yr=c.iloc[-1] / yrs, ex49=p.drop(D49).sum() - months * RL.CME,
                top3=top3 - months * RL.CME, avg=p.mean(), win=(p > 0).mean() * 100, stop=t["손절"].mean() * 100, cost=t["진입가"].mean() * 100,
                mdd=mdd, pk=pk, tr=tr, uw=(s1 - s0).days, uw0=s0, uw1=s1, worst=p.min(), best=p.max(), share49=p[D49] / p.sum() * 100,
                boot=RL.boot_dd(p, months), boot49=RL.boot_dd(p.drop(D49), months), tick=int(t["틱실측"].sum()),
                yearly=p.groupby(p.index.year).sum() - pd.Series(1, MONTHS).groupby(MONTHS.year).sum() * RL.CME, yearly_n=p.groupby(p.index.year).size(),
                monthly=p.groupby(p.index.to_period("M")).sum().reindex(MONTHS, fill_value=0) - RL.CME)

# 매일 매수 기준선 (모든 날, 10:00:00 매도호가 × 10초 비율 — 틱이 신호일에만 있어서 같은 방식으로 맞춤)
ev = pd.Series({d: RL.rule_at(r, float(r.ask) * R[10])[0] for d, r in zip(LA.index, LA.itertuples()) if np.isfinite(r.ask) and r.ask > 0})
rl = pd.Series({d: RL.rule_at(r, float(r.ask) * R[10])[0] for d, r in zip(LS.index, LS.itertuples())})
BASE = {k: v - months * RL.CME for k, v in dict(ev=ev.sum(), ev49=ev.drop(D49).sum(), rl=rl.sum(), rl49=rl.drop(D49).sum()).items()}; BASE["n_ev"] = len(ev)

# 4/9 사실
r49 = LS.loc[D49]; FP = pd.read_pickle(OUT / "f_paths.pkl"); f49 = FP.loc[D49].astype(float)
mp49 = np.asarray(r49.mid_path, float); imin = int(np.nanargmin(mp49[:240]))
e49 = TR[10].loc[D49]
F49 = dict(open=r49.f_open, f10=r49.f10, r30=r49.r30 * 100, K=r49.K, entry=e49["진입가"], low=mp49[imin], low_t=f"{10 + (imin + 1) // 60}:{(imin + 1) % 60:02d}",
           low_pct=(mp49[imin] / e49["진입가"] - 1) * 100, fmin=f49.min(), fmin_t=f49.idxmin(), close=f49["15:59"], pay=r49.pay, pnl=e49["손익$"],
           second=TR[10]["손익$"].drop(D49).max(), second_d=TR[10]["손익$"].drop(D49).idxmax())
tr_all = TR[10]["손익$"]
sym = {n: (tr_all.groupby(tr_all.index.dayofweek, group_keys=False).apply(lambda x: x.drop(x.nlargest(n).index))).sum() - months * RL.CME for n in (1, 3)}

# ── 요일 분석 (2022-05 ~ 만, 10초 매도호가 · −90%, 거래당 손익) — 2020~2022 기간은 쓰지 말 것 (사용자 지시) ──
WIN = TR[10]["손익$"]
def wtab(x):
    g = x.groupby(x.index.dayofweek)
    return {DOW[k]: (len(v), v.mean(), (v > 0).mean() * 100) for k, v in g}
WT_IN = wtab(WIN)
trim1 = lambda x: x.groupby(x.index.dayofweek, group_keys=False).apply(lambda y: y.drop(y.nlargest(1).index))
WT1 = trim1(WIN)
_w, _o = WT1[WT1.index.dayofweek == 2], WT1[WT1.index.dayofweek != 2]
W = dict(wed=(_w.mean(), _o.mean(), sps.ttest_ind(_w, _o, equal_var=False).pvalue),
         rank=sps.mannwhitneyu(WIN[WIN.index.dayofweek == 2], WIN[WIN.index.dayofweek != 2], alternative="less").pvalue)
_a, _b = WIN[WIN.index.dayofweek.isin([1, 3])], WIN[~WIN.index.dayofweek.isin([1, 3])]
W["tt"] = (_a.mean(), _b.mean(), sps.ttest_ind(_a, _b, equal_var=False).pvalue, len(_a))
# 이유: 신호일 요일별 10시→종가 SPX, 오후까지 오른 비율, 콜값(SPX 대비 %), 이벤트
FPp = pd.read_pickle(OUT / "f_paths.pkl")
M = LA.drop(D49).copy(); M["ret"] = (FPp["15:59"].reindex(M.index) / M.f10 - 1) * 100; M["cost"] = M.ask / M.f10 * 100
M["ev"] = label_events(M.index); M["d"] = M.index.dayofweek
MS = M[M.signal]
WHY = {DOW[k]: (g.ret.mean(), (g.ret > 0).mean() * 100, g[g.ev == "이벤트 없음"].cost.mean(), (g.ev != "이벤트 없음").sum(), len(g)) for k, g in MS.groupby("d")}
_ne = MS[MS.ev == "이벤트 없음"]
WHY_NE = {DOW[k]: (g.ret > 0).mean() * 100 for k, g in _ne.groupby("d")}
COST_EV = {e: M[M.ev == e].cost.mean() for e in ("FOMC", "CPI", "이벤트 없음")}
_all = M; _w, _o = _all[_all.d == 2].ret, _all[_all.d != 2].ret
MKT = (_w.mean(), _o.mean(), sps.ttest_ind(_w, _o, equal_var=False).pvalue)
_ae = _all[_all.ev == "이벤트 없음"]; MKT_NE = (_ae[_ae.d == 2].ret.mean(), _ae[_ae.d != 2].ret.mean())

# ── 차트 ──
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c1")
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=8)

subprocess.run([sys.executable, str(pathlib.Path(__file__).parent / "62_equity_curve.py")], check=True, capture_output=True)
shutil.copy(OUT / "equity_2022_now.png", P / "v2_equity.png")

# 4/9 장중
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
t = pd.to_datetime([f"2025-04-09 {h}" for h in f49.index])
a1.plot(t, f49.values, color=INK, lw=1.2); style(a1)
for hh, txt, c in (("10:00", "10:00 매수", BLUE), ("13:18", "13:18 관세 90일 유예 발표", ORANGE)):
    x = pd.Timestamp(f"2025-04-09 {hh}"); a1.axvline(x, color=c, lw=1, ls="--"); a1.text(x, f49.max(), " " + txt, color=c, fontsize=8, va="top")
a1.axhline(r49.K, color=INK2, lw=.8, ls=":"); a1.text(t[0], r49.K + 8, f"행사가 {r49.K:,.0f}", fontsize=8, color=INK2)
a1.set_title("SPX (옵션 패리티, 1분)", fontsize=10, loc="left"); a1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
tt = pd.date_range("2025-04-09 10:01", periods=len(mp49), freq="1min")
a2.plot(tt, mp49, color=BLUE, lw=1.3); style(a2)
a2.axhline(F49["entry"], color=INK2, lw=.8, ls=":"); a2.text(tt[0], F49["entry"] + 12, f"매수가 {F49['entry']:.1f}", fontsize=8, color=INK2)
a2.axhline(F49["entry"] * 0.1, color=ORANGE, lw=.8, ls="--"); a2.text(tt[0], F49["entry"] * 0.1 + 12, "-90% 손절선", fontsize=8, color=ORANGE)
a2.annotate(f"최저 {F49['low']:.1f} ({F49['low_pct']:.0f}%)", xy=(tt[imin], F49["low"]), xytext=(tt[imin] + pd.Timedelta(minutes=20), F49["low"] + 90),
            fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=.8))
a2.text(tt[-1], mp49[-1], f"정산 {F49['pay']:.1f} ", fontsize=8, color=INK, ha="right", va="bottom")
a2.set_title(f"산 콜 {r49.K:,.0f} 중간가", fontsize=10, loc="left"); a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.tight_layout(); fig.savefig(P / "v2_0409.png", dpi=160, facecolor="white"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.0)); style(ax)
xs = np.arange(5); wdt = 0.38
v = [WT_IN[d][1] for d in DOW]; n = [WT_IN[d][0] for d in DOW]
ax.bar(xs, v, 0.55, color=[BLUE if x_ >= 0 else ORANGE for x_ in v])
for x_, v_, n_ in zip(xs, v, n):
    ax.text(x_, v_ + (20 if v_ >= 0 else -20), f"{v_:+,.0f}\n({n_}건)", ha="center", va="bottom" if v_ >= 0 else "top", fontsize=8, color=INK2)
ax.axhline(0, color="#c9c8c1", lw=1); ax.set_xticks(xs); ax.set_xticklabels([f"{d}요일" for d in DOW])
ax.set_ylabel("거래당 손익 ($)", color=INK2, fontsize=9)
ax.set_ylim(-150, 560)
fig.tight_layout(); fig.savefig(P / "v2_weekday.png", dpi=160, facecolor="white"); plt.close(fig)

# ── 표 ──
k5, k10 = K[5], K[10]
kpi = f"""<table class="kpi"><tr>
<td><b>{k10['n']}건</b><br/>매매 (3거래일에 1번)</td>
<td><b>{money(k5['net'])} ~ {money(k10['net'])}</b><br/>총손익 (5초·10초 지연)</td>
<td><b>{money(k5['per_yr'])} ~ {money(k10['per_yr'])}</b><br/>1년 평균</td>
<td><b>{money(k5['mdd'])} ~ {money(k10['mdd'])}</b><br/>최대낙폭</td>
<td><b>{k10['win']:.0f}%</b><br/>승률</td></tr></table>"""
main_rows = "".join(f"<tr><th>{lab}</th><td>{f(K[5])}</td><td>{f(K[10])}</td></tr>" for lab, f in (
    ("매매 수", lambda k: f"{k['n']}건"),
    ("총손익", lambda k: f"<b>{money(k['net'])}</b> (1년 {money(k['per_yr'])})"),
    ("2025-04-09 제외 (4절)", lambda k: money(k["ex49"])),
    ("최고 3건 제외", lambda k: money(k["top3"])),
    ("거래당 평균 / 평균 매수 금액", lambda k: f"{usd(k['avg'])} / ${k['cost']:,.0f}"),
    ("승률 / 손절 비율", lambda k: f"{k['win']:.0f}% / {k['stop']:.0f}%"),
    ("최악의 날 / 최고의 날", lambda k: f"{usd(k['worst'])} / {usd(k['best'])}"),
    ("최대낙폭", lambda k: f"{money(k['mdd'])} ({k['pk']:%Y-%m} → {k['tr']:%Y-%m})"),
    ("가장 길게 물린 기간", lambda k: f"{k['uw']}일 ({k['uw0']:%Y-%m} ~ {k['uw1']:%Y-%m})"),
    ("1년 수익 ÷ 최대낙폭", lambda k: f"{k['per_yr'] / -k['mdd']:.2f}")))
comp_rows = "".join(f"<tr><td>{nm}</td><td class='{cls(a)}'>{money(a)}</td><td class='{cls(b)}'>{money(b)}</td></tr>" for nm, a, b in comp)
years = sorted(set(K[10]["yearly"].index))
yr_rows = "".join(f"<tr><td>{y}</td><td>{int(K[10]['yearly_n'].get(y, 0))}</td>" + "".join(f"<td class='{cls(K[s]['yearly'].get(y, 0))}'>{money(K[s]['yearly'].get(y, 0))}</td>" for s in (5, 10)) + "</tr>" for y in years)
mm = K[10]["monthly"]
mget = lambda y, m: mm.get(pd.Period(f"{y}-{m:02d}", "M"))
mon_rows = "".join(f"<tr><td>{y}</td>" + "".join((lambda v: f"<td class='{cls(v)}'>{v / 1000:+.1f}k</td>" if v is not None else "<td></td>")(mget(y, m)) for m in range(1, 13))
                   + f"<td class='{cls(K[10]['yearly'][y])}'><b>{K[10]['yearly'][y] / 1000:+.1f}k</b></td></tr>" for y in years)
sep = ALL[10][(ALL[10].index.year == 2026) & (ALL[10].index.month == 9)]
sep_rows = "".join(f"<tr><td>{d:%m-%d} ({DOW[d.dayofweek]})</td><td>{r.K:,.0f} 콜</td><td>{r['진입가']:.2f}</td><td>{r['정산']:.2f}</td><td>{'손절' if r['손절'] else '만기 정산'}</td>"
                   f"<td class='{cls(r['손익$'])}'>{usd(r['손익$'])}</td></tr>" for d, r in sep.iterrows())
wd_rows = "".join(f"<tr><td>{d}</td><td>{WT_IN[d][0]}</td><td class='{cls(WT_IN[d][1])}'>{usd(WT_IN[d][1])}</td><td>{WT_IN[d][2]:.0f}%</td></tr>" for d in DOW)
c10 = {nm: (a, b) for nm, a, b in comp}["10초 뒤 매도호가"]
why_rows = "".join(f"<tr><td>{d}</td><td>{WHY[d][4]}</td><td class='{cls(WHY[d][0])}'>{WHY[d][0]:+.3f}%</td><td>{WHY[d][1]:.0f}%</td><td>{WHY_NE[d]:.0f}%</td><td>{WHY[d][2]:.3f}%</td><td>{WHY[d][3]}일</td></tr>" for d in DOW)

html = f"""
<h1>SPX 0DTE 네이키드 매수 전략</h1>
<p class="sub">작성 {pd.Timestamp.now():%Y-%m-%d} · 백테스트 2022-05-16 ~ {last:%Y-%m-%d} · SPX 옵션 1계약 기준</p>

<h2>1. 요약</h2>
<p>S&amp;P500 지수(SPX)의 당일 만기(0DTE) 옵션을 <b>매수만</b> 하는 전략이다. 개장 후 30분 동안 SPX가 올랐고 나스닥100 마이크로 선물(MNQ) 5분봉이 일목균형표 구름 위에 있으면, 뉴욕 시간 10:00에 등가격(ATM) 콜 1계약을 매도호가로 산다. 콜 값이 매수가의 10% 아래로 떨어지면 판다(−90% 손절). 아니면 만기에 현금정산을 받는다.</p>
<p>주문 지연을 고려해 <b>5초·10초 지연 시점의 매도호가</b>로 샀다고 계산했다. 초 단위 호가 데이터는 2023-03-28부터 있다. 그 전(2022-05 ~ 2023-03, {int((ALL[10].index < '2023-03-28').sum())}건)은 지연 없이 10:00:00 매도호가로 계산했다는 점을 감안해야 한다. 모든 숫자에는 NH선물 수수료(계약당 $5)와 MNQ 시세 이용료(CME, 월 $228.80)가 반영돼 있다. SPX 옵션 시세 이용료(CBOE)는 무료다.</p>
{kpi}
<img src="v2_equity.png" width="510"/>
<p class="note">파랑 = 5초 지연, 주황 = 10초 지연, 점선 = 2025-04-09 제외. 아래 = 고점 대비 하락.</p>

<h2>2. 매매 규칙</h2>
<table>
<tr><th>판단 시각</th><td>뉴욕 10:00:00 (한국 23:00, 겨울 24:00)</td></tr>
<tr><th>조건 1</th><td>SPX 10:00:00 가격 &gt; 09:31:00 가격</td></tr>
<tr><th>조건 2</th><td>MNQ 5분봉(09:55~10:00) 종가 &gt; 일목 구름 윗선 (9-26-52)</td></tr>
<tr><th>매수</th><td>ATM 콜 1계약, 매도호가 지정가. 미체결 시 매초 매도호가로 재주문</td></tr>
<tr><th>손절</th><td>호가 중간값이 매수 금액 대비 −90%에 도달하면 바로 손절</td></tr>
<tr><th>그 외</th><td>장중 추가 매수 없음, 하루 1번만 매수. 손절 안 되면 16:00 SPX 종가로 현금정산</td></tr>
</table>

<h2>3. 백테스트 분석</h2>
<table><tr><th></th><th>5초 지연</th><th>10초 지연</th></tr>{main_rows}</table>
<h3>매수 가격에 따라 얼마나 달라지나</h3>
<table><tr><th>매수 가격</th><th>−90% 손절</th><th>손절 없음</th></tr>{comp_rows}</table>
<p>10:00:00 직후 1~5초 동안 매도호가가 평균 1~2% 튀었다가 10초쯤이면 원래대로 돌아온다. 10시에 나오는 경제지표(특히 매달 3번째 영업일 ISM 서비스업) 발표 때문이다. 그래서 10초 지연이 5초보다 조금 낫다. 또 모든 지연에서 <b>−90% 손절이 손절 없이 들고 가는 것보다 낫다</b>(10초 기준 {money(c10[0])} vs {money(c10[1])}). 끝까지 가도 0이 될 콜의 남은 값을 회수하기 때문이다.</p>
<h3>연도별</h3>
<table><tr><th>연도</th><th>매매</th><th>5초</th><th>10초</th></tr>{yr_rows}</table>

<h2 style="page-break-before: always">4. 2025-04-09 (관세 유예 발표일)</h2>
<p>10:00에 SPX +{F49['r30']:.2f}%로 신호가 나와 {F49['K']:,.0f} 콜을 {F49['entry']:.1f}에 샀다. 11:00에 콜이 {F49['low_pct']:.0f}%까지 빠졌지만 손절선엔 닿지 않았고, 13:18 관세 90일 유예 발표로 SPX가 +9.5% 마감해 {F49['pay']:.1f}로 정산됐다. <b>{usd(F49['pnl'])} (매수 금액 대비 {(F49['pay'] - F49['entry']) / F49['entry'] * 100:+.0f}%)</b>, 전체 이익의 약 {k10['share49']:.0f}%다.</p>
<img src="v2_0409.png" width="510"/>
<ul>
<li><b>따로 보는 이유:</b> 이익이 10:00 판단이 아니라 장중 정책 발표에서 나왔고, 한 건이 전체 이익의 {k10['share49']:.0f}%를 차지한다.</li>
<li><b>그래도 규칙의 우위와는 무관:</b> 매일 사도 이날 이익은 똑같이 들어오므로, 규칙이 매일 매수보다 나은 폭({money(BASE['rl'] - BASE['ev'])})은 4/9 포함 여부와 상관없다.</li>
<li><b>큰 날 의존도:</b> 요일마다 최고의 날을 1건씩(5건) 빼면 {money(sym[1])}, 3건씩(15건, 전체의 4%) 빼면 {money(sym[3])}.</li>
</ul>

<h2>5. 전체 기간 월별</h2>
<table class="small"><tr><th>연도</th>{''.join(f'<th>{m}월</th>' for m in range(1, 13))}<th>합계</th></tr>{mon_rows}</table>
<p class="note">플러스 달 {int((mm > 0).sum())} / {len(mm)}.</p>
<h3>2026년 9월 매매</h3>
<table class="small"><tr><th>날짜</th><th>행사가</th><th>매수가</th><th>정산값</th><th>청산</th><th>손익</th></tr>{sep_rows}</table>

<h2>6. 위험</h2>
<table><tr><th></th><th>5초</th><th>10초</th><th>10초, 4/9 제외</th></tr>
<tr><td>실제 최대낙폭</td><td>{money(k5['mdd'])}</td><td>{money(k10['mdd'])}</td><td>{money(k10['mdd'])}</td></tr>
<tr><td>거래 순서 재표본 최대낙폭 — 중앙값</td><td>{money(k5['boot'][0])}</td><td>{money(k10['boot'][0])}</td><td>{money(k10['boot49'][0])}</td></tr>
<tr><td>재표본 — 최악 5%</td><td>{money(k5['boot'][1])}</td><td>{money(k10['boot'][1])}</td><td>{money(k10['boot49'][1])}</td></tr>
<tr><td>재표본 — 최악 1%</td><td>{money(k5['boot'][2])}</td><td>{money(k10['boot'][2])}</td><td>{money(k10['boot49'][2])}</td></tr></table>
<p>같은 거래를 무작위 순서로 5,000번 다시 늘어놓으면 최대낙폭 중앙값이 실제보다 크다. 과거 최대낙폭은 손실이 몰려 오지 않은 순서였고, 앞으로는 더 크게 빠질 수 있다.</p>

<h2>7. 요일별 분석</h2>
<img src="v2_weekday.png" width="470"/>
<table class="small"><tr><th>요일</th><th>건수</th><th>거래당</th><th>승률</th></tr>{wd_rows}</table>
<h3>통계적으로 의미가 있나</h3>
<table>
<tr><th>수요일</th><td>수요일 평균은 4/9(수요일) 하루에 끌려 높게 나온다. 요일마다 최고의 날을 1건씩 똑같이 빼면 수요일 거래당 {usd(W['wed'][0])} vs 다른 요일 {usd(W['wed'][1])} (p = {W['wed'][2]:.2f}), 순위 검정 p = {W['rank']:.3f}. 방향은 뚜렷하지만 아직 우연과 확실히 구분되지는 않는다.</td></tr>
<tr><th>화·목</th><td>화·목 거래당 {usd(W['tt'][0])} ({W['tt'][3]}건) vs 월·수·금 {usd(W['tt'][1])} — 차이는 크지만 p = {W['tt'][2]:.2f}로 우연과 구분되지 않는다.</td></tr>
</table>
<h3>왜 수요일이 약한가 (신호가 뜬 날, 2025-04-09 제외)</h3>
<table class="small"><tr><th>요일</th><th>신호일</th><th>10시→종가 SPX</th><th>오후까지 오른 비율</th><th>같은 비율 (이벤트일 제외)</th><th>콜 값 (SPX 대비, 이벤트일 제외)</th><th>FOMC·CPI·고용일</th></tr>{why_rows}</table>
<ul>
<li><b>이벤트가 수요일에 몰려 있다.</b> FOMC(14:00 발표)는 항상 수요일이고 CPI도 수요일이 많다. 이런 날 10:00 콜 값은 SPX의 {COST_EV['FOMC']:.2f}%(FOMC)·{COST_EV['CPI']:.2f}%(CPI)로 보통 날({COST_EV['이벤트 없음']:.2f}%)보다 비싸고, 오후 발표로 방향이 뒤집히기 쉽다.</li>
<li><b>이벤트가 없는 수요일에도 아침 상승이 덜 이어진다.</b> 오후까지 오른 비율 {WHY_NE['수']:.0f}% vs 다른 요일 {min(WHY_NE[d] for d in '월화목금'):.0f}~{max(WHY_NE[d] for d in '월화목금'):.0f}%. 콜 값은 비슷하므로 가격이 아니라 방향 문제다.</li>
<li><b>시장 전체의 수요일 약세는 아니다.</b> 모든 날 기준 수요일 10시→종가 SPX는 {MKT[0]:+.3f}% vs 다른 요일 {MKT[1]:+.3f}% (p = {MKT[2]:.2f}), 이벤트일을 빼면 차이가 거의 없다. "아침에 오른 수요일"에서만 나타나는 현상이며 원인은 아직 확인되지 않았다.</li>
</ul>
<p>규칙은 바꾸지 않고 "수요일 쉬기"를 가설로 등록해 앞으로 쌓이는 매매로 확인한다.</p>

<h2>8. 한계와 다음 단계</h2>
<ul>
<li>백테스트 기간 4.3년, 매매 약 400건. 이익이 드문 큰 날에 몰려 있어 이 정도 표본으론 우위를 확정하기 어렵다(거래당 표준편차 약 $2,600 → 확정에 수년 치 필요).</li>
<li>규칙은 많은 조합(약 1,600개 시도) 중 고른 것이다. 실제 기대치는 백테스트보다 낮다고 봐야 한다.</li>
<li>요일 가설(수요일 약세)은 앞으로 쌓이는 매매로 판정.</li>
<li>2023-03 이전 {int((ALL[10].index < '2023-03-28').sum())}건은 지연 없는 10:00:00 매도호가 기준. 실전 주문 지연은 아직 측정 전.</li>
<li>매일 자동으로 새 날짜를 받아 판단·손익을 기록 중(2026-09-24~).</li>
</ul>

<h2>9. 데이터와 방법</h2>
<ul>
<li>옵션: SPXW 당일 만기 1분 최우선 호가 (Databento OPRA, 2022-05-16 ~ 2026-09-23, 1,093거래일) + 매수일 초 단위 호가 (2023-03-28 ~, 319일)</li>
<li>선물: MNQ 1분봉 연결선물 (Databento CME, 월물 교체 가격 보정), SPX 일봉 종가 (Investing.com)</li>
<li>SPX 값: 옵션 풋콜패리티로 계산 (실제 SPX와 분 단위 차이 중앙값 0.35pt)</li>
<li>체결: 5·10초 뒤 매도호가 매수 (2023-03 전은 10:00:00 매도호가), 손절은 그 분 매수호가 매도, 아니면 16:00 SPX 종가 현금정산. 수수료 편도 $5 + 정산 $5 가정, CME 시세 이용료 월 $228.80</li>
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
h2 { font-size: 13pt; color: #1c5cab; margin-top: 14pt; margin-bottom: 4pt; border-bottom: 1px solid #D8DED9; }
h3 { font-size: 10.5pt; margin-top: 8pt; margin-bottom: 3pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0; }
th, td { border: 1px solid #D8DED9; padding: 3pt 5pt; text-align: left; }
th { font-weight: bold; color: #1c5cab; }
table.kpi td { text-align: center; font-size: 9pt; }
table.small th, table.small td { font-size: 8pt; padding: 2pt 4pt; }
.pos { color: #0F6E5A; } .neg { color: #B3261E; }
.note { color: #6B7570; font-size: 8pt; }
img { margin: 4pt 0; }
"""
for f in ("malgun.ttf", "malgunbd.ttf"):
    shutil.copy(pathlib.Path(r"C:\Windows\Fonts") / f, P / f)
story = fitz.Story(html=html, user_css=css, archive=fitz.Archive(str(P)))
out = P / "SPX0DTE_네이키드_매수전략.pdf"
writer = fitz.DocumentWriter(str(out))
mediabox = fitz.paper_rect("a4"); where = mediabox + (42, 42, -42, -42)
more = 1
while more:
    dev = writer.begin_page(mediabox); more, _ = story.place(where); story.draw(dev); writer.end_page()
writer.close(); del writer, story; import gc; gc.collect()
_d = fitz.open(out); _d.subset_fonts(); _tmp = out.with_suffix(".tmp.pdf")
_d.save(_tmp, garbage=4, deflate=True, clean=True); _d.close()
try:
    _tmp.replace(out)
except PermissionError:
    out = out.with_name(out.stem + f"_{pd.Timestamp.now():%m%d_%H%M}.pdf"); _tmp.replace(out)
for f in ("malgun.ttf", "malgunbd.ttf"):
    (P / f).unlink()
print("저장:", out, "쪽수:", len(fitz.open(out)))
for k in RL.DELAYS:
    print(k, {x: (round(v) if isinstance(v, (int, float, np.floating)) else v) for x, v in K[k].items() if x in ("n", "gross", "net", "per_yr", "ex49", "top3", "mdd", "share49", "win", "stop")})
print("기준선", {k: round(v) for k, v in BASE.items()}, "대칭 제거", {k: round(v) for k, v in sym.items()}, "요일", W, WHY_NE)
print(F49)
