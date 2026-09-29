"""CBOE 공식 일간 지수 CSV (무료, 로그인 없음): VIX, VIX9D 등. 실전에선 매일 아침 전날 종가를 여기서 받음."""
import io
import urllib.request
import pandas as pd

URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"


def fetch(name="VIX", save=None):
    req = urllib.request.Request(URL.format(name), headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    df = df.set_index("DATE").sort_index(); df.columns = [c.lower() for c in df.columns]; df.index.name = "date"
    if save:
        df.to_csv(save)
    return df


def load(names=("VIX", "VIX9D"), folder=None):
    """전날 종가 표 (열 = 지수 이름). folder에 저장본이 있으면 그걸, 없으면 받아서 저장."""
    from .config import DATA
    folder = folder or DATA
    out = {}
    for n in names:
        p = folder / f"cboe_{n}.csv"
        out[n] = (pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else fetch(n, save=p))["close"]
    return pd.DataFrame(out)
