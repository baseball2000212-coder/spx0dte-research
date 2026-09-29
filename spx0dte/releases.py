"""
10:00 ET 경제지표 발표일 (1초 스파이크 분석용, 2026-09-29).
  - JOLTS: bls.gov/bls/news-release/jolts.htm 아카이브 링크 날짜 (내장 브라우저로 직접 추출, 2023-01 ~ 2026-09).
  - ISM 서비스업 = 매달 3번째 영업일, ISM 제조업 = 1번째 영업일 (ISM 발표 규칙, 미국 연방 공휴일 기준).
  - 미시간대 소비자심리 = 2번째(속보)·4번째(확정) 금요일, 컨퍼런스보드 소비자신뢰 = 마지막 화요일 (규칙 추정, 공식 날짜 대조 안 함).
겹치면 ISM 서비스업 > JOLTS > ISM 제조업 > 미시간대 > 소비자신뢰 순으로 하나만.
"""
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

JOLTS = pd.DatetimeIndex("""2023-01-04 2023-02-01 2023-03-08 2023-04-04 2023-05-02 2023-05-31 2023-07-06 2023-08-01 2023-08-29 2023-10-03 2023-11-01 2023-12-05
2024-01-03 2024-01-30 2024-03-06 2024-04-02 2024-05-01 2024-06-04 2024-07-02 2024-07-30 2024-09-04 2024-10-01 2024-10-29 2024-12-03
2025-01-07 2025-02-04 2025-03-11 2025-04-01 2025-04-29 2025-06-03 2025-07-01 2025-07-29 2025-09-03 2025-09-30 2025-12-09
2026-01-07 2026-02-05 2026-03-13 2026-03-31 2026-05-05 2026-06-02 2026-06-30 2026-08-04 2026-09-01""".split())

ORDER = ("ISM 서비스업", "JOLTS", "ISM 제조업", "미시간대", "소비자신뢰", "발표 없음")
_BD = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())


def nth_business_day(d):
    d = pd.Timestamp(d).normalize()
    return len(pd.date_range(d.replace(day=1), d, freq=_BD))


def label_release(d):
    d = pd.Timestamp(d).normalize(); bd = nth_business_day(d); nth = (d.day - 1) // 7 + 1
    if bd == 3:
        return "ISM 서비스업"
    if d in JOLTS:
        return "JOLTS"
    if bd == 1:
        return "ISM 제조업"
    if d.dayofweek == 4 and nth in (2, 4):
        return "미시간대"
    if d.dayofweek == 1 and (d + pd.Timedelta(days=7)).month != d.month:
        return "소비자신뢰"
    return "발표 없음"
