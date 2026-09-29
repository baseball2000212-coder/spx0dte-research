"""
기준 전략 보고서용 숫자·차트. 몇 분.
결과: output/report/numbers.json, output/report/equity.png, output/report/yearly.png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT, SPX_CSV
from spx0dte.core import load_spx_ohlc
from spx0dte.touch import load_mnq, signal_grids_ext
from spx0dte.strategy import features_10, trade_detail

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.parse_math": False})
SPLIT, ODD = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-09")
R = OUT / "report"; R.mkdir(exist_ok=True)

FP = pd.read_pickle(OUT / "f_paths.pkl")
X = features_10(FP, signal_grids_ext(load_mnq()[0], FP), load_spx_ohlc(SPX_CSV))
L = pd.read_pickle(OUT / "legs10_full.pkl").set_index("date").join(X, how="inner")
YEARS = len(FP) / 252
VARS = {"ATM 1계약": (0, False, "중간가"), "+10pt 1계약": (10, False, "중간가"), "+10pt 물타기": (10, True, "중간가"),
        "ATM 1계약 (매도호가 체결)": (0, False, "매도호가"), "+10pt 1계약 (매도호가 체결)": (10, False, "매도호가"),
        "ATM 1계약 (조건 없이 30분 상승만)": (0, False, "중간가")}
out, curves = {}, {}
for name, (off, avg, fill) in VARS.items():
    sub = L[(L.leg == "C") & (L.offset == off)]
    sub = sub[(sub.r30 > 0)] if "조건 없이" in name else sub[sub.signal]
    t = pd.DataFrame([trade_detail(r, averaging=avg, fill=fill) for r in sub.itertuples()], index=sub.index).sort_index()
    ret = lambda g: float(g.pnl.sum() / g.cost.sum() * 100)
    eq = (t.pnl * 100).cumsum()
    dd = eq - eq.cummax()
    # 가장 긴 손실 구간: 고점 갱신 못한 연속 거래 수 + 달력 기간
    under = (dd < 0).values; best, cur, s0, span = 0, 0, None, (None, None)
    for i, u in enumerate(under):
        if u:
            cur += 1; s0 = s0 if s0 is not None else i
            if cur > best:
                best, span = cur, (t.index[s0], t.index[i])
        else:
            cur, s0 = 0, None
    v = t[t.index >= SPLIT]
    yr = (t.pnl * 100).groupby(t.index.year).sum()
    out[name] = {"trades": int(len(t)), "per_year": round(len(t) / YEARS, 1), "avg_cost": round(float(t.cost.mean() * 100)),
                 "train": round(ret(t[t.index < SPLIT]), 1), "val": round(ret(v), 1), "val_ex": round(ret(v[v.index != ODD]), 1),
                 "top10_out": round(ret(t.drop(t.pnl.nlargest(10).index)), 1), "win": round(float((t.pnl > 0).mean() * 100), 1),
                 "total": round(float(eq.iloc[-1])), "total_ex": round(float(eq.iloc[-1] - t.pnl.get(ODD, 0) * 100)),
                 "odd": round(float(t.pnl.get(ODD, 0) * 100)), "maxdd": round(float(dd.min())), "worst": round(float(t.pnl.min() * 100)),
                 "best": round(float(t.pnl.max() * 100)), "yearly": {int(k): round(float(v_)) for k, v_ in yr.items()},
                 "yearly_n": {int(k): int(v_) for k, v_ in t.groupby(t.index.year).size().items()},
                 "longest_dd_trades": int(best), "longest_dd_span": [f"{span[0]:%Y-%m-%d}", f"{span[1]:%Y-%m-%d}"] if span[0] is not None else None,
                 "avg_adds": round(float((t.n - 1).mean()), 2), "add_rate": round(float((t.n > 1).mean() * 100), 1)}
    curves[name] = eq
json.dump(out, open(R / "numbers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# 차트 (밝은 바탕, 보고서용)
C = {"ATM 1계약": "#0F6E5A", "+10pt 1계약": "#3A86A8", "+10pt 물타기": "#C77B2B", "ATM 1계약 (조건 없이 30분 상승만)": "#9AA39E"}
fig, ax = plt.subplots(figsize=(10, 4.2))
for k, col in C.items():
    e = curves[k]; ax.plot(e.index, e.values / 1000, color=col, lw=1.6 if "조건 없이" not in k else 1.1, label=k)
ax.axhline(0, color="#1B2320", lw=.8); ax.axvline(SPLIT, color="#9AA39E", ls="--", lw=.8)
ax.axvline(ODD, color="#B3261E", ls=":", lw=.9); ax.text(ODD, ax.get_ylim()[1] * .93, " 4/9 관세 유예", color="#B3261E", fontsize=8)
ax.text(SPLIT, ax.get_ylim()[0] * .9 if ax.get_ylim()[0] < 0 else 2, " 검증 구간 →", color="#6B7570", fontsize=8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False, loc="upper left"); ax.set_ylabel("누적 손익 (천 달러)")
fig.tight_layout(); fig.savefig(R / "equity.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.4))
yrs = sorted(out["ATM 1계약"]["yearly"]); w = .27
for j, (k, col) in enumerate([("ATM 1계약", "#0F6E5A"), ("+10pt 1계약", "#3A86A8"), ("+10pt 물타기", "#C77B2B")]):
    vals = [out[k]["yearly"].get(y, 0) / 1000 for y in yrs]
    ax.bar(np.arange(len(yrs)) + (j - 1) * w, vals, w, color=col, label=k)
ax.set_xticks(range(len(yrs)), [f"{y}" + (" (9월까지)" if y == 2026 else "") + (" *4/9 포함" if y == 2025 else "") for y in yrs], fontsize=8)
ax.axhline(0, color="#1B2320", lw=.8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8, frameon=False); ax.set_ylabel("연도별 손익 (천 달러)")
fig.tight_layout(); fig.savefig(R / "yearly.png", dpi=150); plt.close(fig)
print(json.dumps(out, ensure_ascii=False, indent=1))
