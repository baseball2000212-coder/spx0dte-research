"""
표본외 검증 판정·보고 (43_oos.py --eval 결과 output/oos/days.csv 사용). 판정 기준은 사전등록_판정기준.md.
결과: output/oos/result.json, oos_equity.png, oos_random.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
R = OUT / "oos"; RNG = np.random.default_rng(20260927); NSIM = 10_000
ACC, LOSS, MUTED, INK = "#0F6E5A", "#B3261E", "#9AA39E", "#17201C"
D = pd.read_csv(R / "days.csv", parse_dates=["date"], encoding="utf-8-sig").set_index("date")
ok = D[D["손익$"].notna()]
for c in ("조건1", "조건2", "신호", "조건1(09:59)"):
    ok[c] = ok[c].astype(str).str.lower().eq("true")
sig = ok[ok.신호]


def stats(p, cost=None):
    eq = p.cumsum()
    return {"매수": int(len(p)), "총손익$": float(p.sum()), "건당$": float(p.mean()) if len(p) else np.nan,
            "승률%": float((p > 0).mean() * 100) if len(p) else np.nan, "최대낙폭$": float((eq - eq.cummax()).min()) if len(p) else 0,
            "연도별$": {int(k): float(v) for k, v in p.groupby(p.index.year).sum().items()}}


out = {"평가한 날": int(len(ok)), "제외(오류)": int(len(D) - len(ok))}
out["현재 규칙 (중간가)"] = stats(sig["손익$"])
out["현재 규칙 (매도호가)"] = stats(sig["손익$ 매도호가"])
d_sig = ok[ok["조건1(09:59)"] & ok.조건2]
out["1분 지연 (09:59 판단 → 10:01 매수)"] = stats(d_sig["손익$ 10:01"].dropna())
out["기준선"] = {"매일 매수": stats(ok["손익$"]), "조건1만": stats(ok[ok.조건1]["손익$"]),
               "조건2만": stats(ok[ok.조건2]["손익$"]), "둘 다 (현재)": stats(sig["손익$"])}
# 무작위 대조
v = ok["손익$"].values; n = len(sig)
sims = np.array([v[RNG.choice(len(v), n, replace=False)].sum() for _ in range(NSIM)])
real = sig["손익$"].sum(); pct = float((sims < real).mean() * 100)
out["무작위 대조"] = {"실제": float(real), "무작위 중앙값": float(np.median(sims)), "무작위 상위 10% 선": float(np.percentile(sims, 90)),
                   "백분위": pct, "무작위가 실제 이상인 비율%": 100 - pct}
# 부트스트랩
p = sig["손익$"].values
ms = np.array([p[RNG.integers(0, n, n)].mean() for _ in range(NSIM)])
out["부트스트랩"] = {"건당 평균$": float(p.mean()), "95% 구간": [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))],
                  "P(평균≤0)%": float((ms <= 0).mean() * 100)}
out["상위 제거"] = {f"상위 {k}건": float(sig["손익$"].drop(sig["손익$"].nlargest(k).index).sum()) for k in (1, 3, 5, 10)}
out["최고 5일"] = {f"{d:%Y-%m-%d}": float(x) for d, x in sig["손익$"].nlargest(5).items()}
# 판정
crit = {"① 총손익 > 0": real > 0, "② 건당 평균 > 0": p.mean() > 0, "③ 무작위 대비 상위 10%": pct >= 90}
out["판정 기준"] = {k: bool(x) for k, x in crit.items()}
out["판정"] = "통과" if all(crit.values()) else "불통과"
json.dump(out, open(R / "result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

# 차트
fig, ax = plt.subplots(figsize=(9, 3.4))
for lab, s, c, lw in (("매일 매수", ok["손익$"], MUTED, 1), ("조건1만", ok[ok.조건1]["손익$"], "#3A86A8", 1),
                      ("조건2만", ok[ok.조건2]["손익$"], "#C77B2B", 1), ("현재 규칙", sig["손익$"], ACC, 2)):
    ax.plot(s.index, s.cumsum() / 1000, color=c, lw=lw, label=f"{lab} ({len(s)}건)")
ax.axhline(0, color=INK, lw=.8); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25); ax.set_ylabel("누적 손익 (천 달러)")
ax.set_title("표본외 2020-01 ~ 2022-05 (월·수·금 0DTE), 10:00 ATM 콜 1계약", fontsize=10)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(R / "oos_equity.png", dpi=170); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.hist(sims / 1000, bins=60, color=MUTED); ax.axvline(real / 1000, color=ACC, lw=2); ax.axvline(0, color=INK, lw=.6)
ax.axvline(np.percentile(sims, 90) / 1000, color=LOSS, ls=":", lw=1)
ax.text(real / 1000, ax.get_ylim()[1] * .85, f" 현재 규칙\n 상위 {100 - pct:.1f}%", color=ACC, fontsize=8)
ax.text(np.percentile(sims, 90) / 1000, ax.get_ylim()[1] * .6, " 상위 10% 선", color=LOSS, fontsize=8)
ax.set_xlabel(f"무작위 {n}일 × {NSIM:,}회 총손익 (천 달러)")
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(R / "oos_random.png", dpi=170); plt.close(fig)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
