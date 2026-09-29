"""
연구 규칙 (2026-09-27 사용자와 합의): 새 아이디어는 2023~2024로만 찾고, 2025-01-01 ~ 2026-09-23은 떼어둔다.
  - 찾는 단계: 모든 분석은 discovery()로 거른 데이터만 사용
  - 검증 단계: output/holdout/사전등록_<이름>.md 에 규칙·판정 기준을 먼저 쓰고 unlock()으로만 열람 (한 아이디어당 한 번)
  - 열람 기록은 output/holdout/access.log 에 남음. 결과 보고 규칙 수정 금지.
  - 주의: 현재 기준 전략(10:00 + SPX 30분 상승 + MNQ 5분 구름 계열)의 변형은 2025~26을 이미 봤으므로 이 방식으로 검증 불가.
  - 2022-05 ~ 2022-12는 2022 약세장 참고용으로 찾는 단계에 넣어도 됨 (disc_start로 조절).
"""
import datetime as dt
from pathlib import Path
import pandas as pd

DISC_START, DISC_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
HOLD_START, HOLD_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-23")
DIR = Path("output") / "holdout"


def _idx(x):
    return pd.DatetimeIndex(x.index if hasattr(x, "index") and not isinstance(x, pd.DatetimeIndex) else x)


def discovery(x, disc_start=DISC_START):
    """찾는 단계용: 2025-01-01 이후 날짜를 잘라낸 데이터 (DataFrame·Series·DatetimeIndex)."""
    idx = _idx(x).normalize()
    m = (idx >= disc_start) & (idx <= DISC_END)
    return x[m] if not isinstance(x, pd.DatetimeIndex) else x[m]


def unlock(name, x):
    """검증 단계: 사전등록 파일이 있어야 2025~26 데이터를 돌려줌. 같은 이름으로 두 번째 열람하면 경고."""
    DIR.mkdir(parents=True, exist_ok=True)
    reg = DIR / f"사전등록_{name}.md"
    if not reg.exists():
        raise PermissionError(f"{reg} 가 없음 — 규칙과 판정 기준을 먼저 적어야 떼어둔 데이터를 열 수 있음")
    log = DIR / "access.log"
    prev = log.read_text(encoding="utf-8").count(f"\t{name}\t") if log.exists() else 0
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M}\t{name}\t{'첫 열람' if prev == 0 else f'{prev + 1}번째 열람 (주의)'}\n")
    if prev:
        print(f"⚠ '{name}'은 떼어둔 데이터를 이미 {prev}번 봤음 — 이번 결과는 표본외로 볼 수 없음")
    idx = _idx(x).normalize()
    m = (idx >= HOLD_START) & (idx <= HOLD_END)
    return x[m] if not isinstance(x, pd.DatetimeIndex) else x[m]
