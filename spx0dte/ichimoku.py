"""
SPX 분봉 + 일목균형표. SPX 1분 = 옵션 패리티 선도가격 F (output/f_paths.pkl, 정규장 09:31~15:59).
날짜를 이어붙인 연속 시계열 (SPX 지수는 정규장만 움직임 → HTS SPX 차트와 같은 구성).
"""
import numpy as np
import pandas as pd

OPEN_MIN = 9 * 60 + 30


def bars(FP, n):
    """f_paths(날짜 × 'HH:MM') → n분봉 OHLC (09:30 기준으로 자름, 봉 이름 = 시작 시각). 조기폐장일 13:00 이후 제거."""
    fp = FP.copy()
    half = fp.loc[:, "13:05":"15:59"].std(axis=1) < 1e-6
    fp.loc[half, "13:01":"15:59"] = np.nan
    s = fp.stack().rename("px").reset_index()
    s.columns = ["date", "hhmm", "px"]
    mins = s.hhmm.str[:2].astype(int) * 60 + s.hhmm.str[3:].astype(int)
    s["bin"] = (mins - OPEN_MIN) // n
    g = s.groupby(["date", "bin"]).px
    o = pd.DataFrame({"open": g.first(), "high": g.max(), "low": g.min(), "close": g.last()}).reset_index()
    o.index = o.date + pd.to_timedelta(OPEN_MIN + o["bin"] * n, unit="min")
    return o[["open", "high", "low", "close"]].astype(float)


def ichimoku(b, t=9, k=26, s=52):
    """전환선·기준선·선행스팬1·2(현재 위치에 그려지는 값 = k봉 전에 계산된 값)·후행스팬(표시용, 미래값이라 신호 금지)."""
    mid = lambda w: (b.high.rolling(w).max() + b.low.rolling(w).min()) / 2
    out = pd.DataFrame(index=b.index)
    out["전환선"] = mid(t)
    out["기준선"] = mid(k)
    out["선행스팬1"] = ((out["전환선"] + out["기준선"]) / 2).shift(k)
    out["선행스팬2"] = mid(s).shift(k)
    out["후행스팬"] = b.close.shift(-k)
    out["구름위"] = out[["선행스팬1", "선행스팬2"]].max(axis=1)
    out["구름아래"] = out[["선행스팬1", "선행스팬2"]].min(axis=1)
    return out


def draw(ax, b, ic, title, candles=True, marks=()):
    """캔들(또는 선) + 전환선·기준선·구름. x축은 봉 순서 (밤사이 빈칸 없음)."""
    x = np.arange(len(b))
    up = (ic["선행스팬1"] >= ic["선행스팬2"]).values
    ax.fill_between(x, ic["선행스팬1"], ic["선행스팬2"], where=up, color="#2A9D8F", alpha=.25, interpolate=True, label="구름(상승)")
    ax.fill_between(x, ic["선행스팬1"], ic["선행스팬2"], where=~up, color="#E76F51", alpha=.25, interpolate=True, label="구름(하락)")
    if candles:
        col = np.where(b.close >= b.open, "#D62828", "#1D4ED8")      # 한국식: 양봉 빨강, 음봉 파랑
        ax.vlines(x, b.low, b.high, color=col, lw=.8)
        ax.bar(x, (b.close - b.open).abs().clip(lower=0.05), bottom=np.minimum(b.open, b.close), color=col, width=.7)
    else:
        ax.plot(x, b.close, color="black", lw=1, label="SPX")
    ax.plot(x, ic["전환선"], color="#E63946", lw=1, label="전환선(9)")
    ax.plot(x, ic["기준선"], color="#457B9D", lw=1.2, label="기준선(26)")
    for pos, txt, c in marks:
        ax.axvline(pos, color=c, lw=1.2, ls="--"); ax.text(pos, ax.get_ylim()[1], txt, color=c, fontsize=8, va="top")
    ax.set_title(title, fontsize=10); ax.grid(alpha=.25)
    return x
