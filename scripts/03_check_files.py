"""0바이트·깨진 0DTE 파일 삭제 → 02를 다시 돌리면 그날만 재다운로드."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import databento as db
from spx0dte.config import OPT_DIR

bad = []
for f in sorted(OPT_DIR.glob("*.dbn.zst")):
    try:
        if f.stat().st_size == 0 or len(db.DBNStore.from_file(str(f)).to_df()) == 0:
            bad.append(f)
    except Exception:
        bad.append(f)
for f in bad:
    f.unlink()
print(f"전체 {len(list(OPT_DIR.glob('*.dbn.zst')))}개, 깨진 파일 {len(bad)}개 삭제:", [f.name for f in bad])
