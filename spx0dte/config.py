"""경로·기간 설정. API 키는 파일에 쓰지 말고 환경변수 DATABENTO_API_KEY로."""
import os
from pathlib import Path

DATA = Path(os.environ.get("SPX0DTE_DATA", "data"))           # 기존 다운로드 폴더
SPX_CSV = os.environ.get("SPX0DTE_SPX_CSV", "data/spx_daily.csv")   # 인베스팅 SPX 일봉 2020-01~ (갱신 시 이 파일 덮어쓰기)

OPT_DIR = DATA / "0dte"                 # 날짜별 SPXW 0DTE cbbo-1m (YYYY-MM-DD.dbn.zst)
EXT_DIR = DATA / "0dte_ext"             # ±3% 밖 보충 행사가 (scripts/02b_extend_strikes.py), load_day가 자동 합침
FUT_RAW = DATA / "fut_1m.dbn.zst"       # MNQ·MGC 1분봉 연결선물 원본
FUT_PARQUET = DATA / "fut_1m.parquet"
OUT = Path(os.environ.get("SPX0DTE_OUT", "output"))

FUT_START, FUT_END = "2020-01-01", "2026-09-24"
OPT_START, OPT_END = "2022-05-16", "2026-09-24"
OPT_BAND = 0.03          # 전일 종가 ±3% 행사가
OPT_BUDGET = 100.0       # 한 번 실행에서 옵션 다운로드 예산 상한($)


def client():
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY", "")
    if not key and os.name == "nt":                     # 앱이 키 설정 전에 켜졌으면 Windows 사용자 환경변수에서 직접 읽기
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "DATABENTO_API_KEY")[0]
        except OSError:
            key = ""
    assert key.startswith("db-") and key.isascii(), "환경변수 DATABENTO_API_KEY를 설정해줘"
    return db.Historical(key)
