"""
매수 시각 × 매도 시각 전체 조합 (롱 ATM 스트래들, 08 결과 사용).
  - 매수: 진입 시각 ask
  - 매도: 정한 시각의 bid (시간 청산) 또는 '만기' = 16:00 정산
  - 규칙 2개: 시간 청산만 / 시간 청산 + −30% 손절
  - 학습 < 2025-01-01 ≤ 검증
결과: output/G1_entry_exit.png, G1_entry_exit.csv
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spx0dte.config import OUT
from spx0dte.features import PATH_ENTRIES
from spx0dte.exits import run_rule, minutes

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
SPLIT = pd.Timestamp("2025-01-01")
EXITS = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30",
         "15:00", "15:30", "15:45", "15:50", "만기"]
RULES = {"시간 청산만": None, "시간 청산 + -30% 손절": 0.3}

P = pd.read_pickle(OUT / "straddle_paths.pkl")
rows = []
for rule, sl in RULES.items():
    for x in EXITS:
        t = run_rule(P, sl=sl, exit_at=None if x == "만기" else x)
        t = t[t.entry.map(minutes) < (16 * 60 if x == "만기" else minutes(x))]
        t["part"] = np.where(t.date < SPLIT, "학습", "검증")
        for (e, part), g in t.groupby(["entry", "part"]):
            rows.append({"rule": rule, "entry": e, "exit": x, "part": part, "n": len(g),
                         "수익률%": g.pnl.sum() / g.cost_ask.sum() * 100, "승률%": (g.pnl > 0).mean() * 100,
                         "건당 평균$": g.pnl.mean() * 100, "손절%": (g.why == "손절").mean() * 100})
G = pd.DataFrame(rows)
G.round(2).to_csv(OUT / "G1_entry_exit.csv", index=False, encoding="utf-8-sig")

# ── 히트맵: 행 = 매수 시각, 열 = 매도 시각 ──
fig, ax = plt.subplots(2, 2, figsize=(20, 16))
for r, rule in enumerate(RULES):
    for c, part in enumerate(("학습", "검증")):
        H = G[(G.rule == rule) & (G.part == part)].pivot(index="entry", columns="exit", values="수익률%") \
            .reindex(index=PATH_ENTRIES, columns=EXITS)
        x = ax[r, c]
        x.imshow(H.values, cmap="RdYlGn", vmin=-25, vmax=25, aspect="auto")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                v = H.values[i, j]
                if np.isfinite(v):
                    x.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
        x.set_xticks(range(len(EXITS)), EXITS, rotation=45); x.set_yticks(range(len(PATH_ENTRIES)), PATH_ENTRIES)
        x.set_xlabel("매도 시각"); x.set_ylabel("매수 시각")
        x.set_title(f"{rule} | {part} (수익률 %, 수수료 포함)")
fig.tight_layout(); fig.savefig(OUT / "G1_entry_exit.png", dpi=110); plt.close(fig)

# ── 두 구간 다 좋은 칸 + 이웃 칸 평균 (진짜 패턴이면 이웃도 비슷해야 함) ──
W = G.pivot_table(index=["rule", "entry", "exit"], columns="part", values="수익률%").dropna()
W["둘 중 나쁜 쪽"] = W[["학습", "검증"]].min(axis=1)
def nb(rule, e, x, part):
    i, j = PATH_ENTRIES.index(e), EXITS.index(x)
    cells = [(PATH_ENTRIES[a], EXITS[b]) for a in range(max(0, i - 1), min(len(PATH_ENTRIES), i + 2))
             for b in range(max(0, j - 1), min(len(EXITS), j + 2)) if (a, b) != (i, j)]
    v = [W.loc[(rule, a, b), part] for a, b in cells if (rule, a, b) in W.index]
    return np.mean(v) if v else np.nan
top = W.sort_values("둘 중 나쁜 쪽", ascending=False).head(12).copy()
top["이웃 평균(검증)"] = [nb(r, e, x, "검증") for r, e, x in top.index]
pd.set_option("display.width", 200)
print(top.round(1).to_string())
print(f"\n시도한 칸: {len(W)}개 (규칙 2 × 매수·매도 조합)")
