"""
시간대별 그릭 손익 분해.
  python scripts/04_greek_attribution.py --step 1 --bucket 30min
날짜별 결과는 OUT/cache_... 에 캐시. 결과 CSV와 ATM/OTM 차트 저장.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, warnings
import pandas as pd, databento as db
from spx0dte.config import OPT_DIR, OUT
from spx0dte.core import attribute_day, summ, finalize, plot_greeks, ALL

warnings.filterwarnings("ignore", category=RuntimeWarning)       # All-NaN slice (패리티 쌍 없는 분) 무시
ap = argparse.ArgumentParser()
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--bucket", default="30min")
ap.add_argument("--smooth", type=int, default=1)                 # 1 권장 (평활하면 진짜 IV 변화가 잔차로 샘)
a = ap.parse_args()

cache = OUT / f"cache_{a.bucket}_s{a.step}_m{a.smooth}"; cache.mkdir(parents=True, exist_ok=True)
files = sorted(OPT_DIR.glob("*.dbn.zst")); rows = []
for i, f in enumerate(files, 1):
    cf = cache / f"{f.name[:10]}.parquet"
    if cf.exists():
        rows.append(pd.read_parquet(cf)); continue
    try:
        x = attribute_day(db.DBNStore.from_file(str(f)).to_df(), pd.Timestamp(f.name[:10]), a.step, a.bucket, smooth=a.smooth)
    except Exception as ex:
        print(f"{f.name[:10]} 실패: {ex}"); continue
    if x.empty:
        continue
    s = x.groupby(["group", "bucket"]).apply(summ, include_groups=False).reset_index(); s["date"] = f.name[:10]
    s.to_parquet(cf); rows.append(s)
    if i % 50 == 0:
        print(f"{i}/{len(files)}")
allr = pd.concat(rows)
res = finalize(allr.drop(columns="date").groupby(["group", "bucket"]).sum())
res["days"] = allr.groupby(["group", "bucket"])["date"].nunique()
tag = f"{a.bucket}_s{a.step}_m{a.smooth}"
res.to_csv(OUT / f"greek_attribution_{tag}.csv", encoding="utf-8-sig")
print(res.loc["ATM", [f"{k}_share" for k in ALL]].round(3))
for g in ("ATM", "OTM"):
    plot_greeks(res, g, save=str(OUT / f"greeks_{g}_{tag}.png"))
