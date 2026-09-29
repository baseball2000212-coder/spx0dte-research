"""SPXW 0DTE 분석 공용 함수: 심볼 생성, Black-76, 그릭 손익 분해, 차트."""
import math
import numpy as np
import pandas as pd
import databento as db
from scipy.special import ndtr

NY = "America/New_York"
YEAR_MIN = 365 * 24 * 60
TERMS = ["delta", "theta", "gamma", "vega"]
ALL = TERMS + ["resid"]                          # 잔차 = 호가 노이즈 + 테일러 고차항
NAMES = {"delta": "델타", "theta": "세타", "gamma": "감마", "vega": "베가", "resid": "잔차(노이즈·고차항)"}
COLORS = {"delta": "#3D5A80", "theta": "#B23A48", "gamma": "#2A7F62", "vega": "#C9822B", "resid": "#9AA3AD"}



# ---------- SPX 일봉 (인베스팅 CSV) ----------
def load_spx_ohlc(path):
    """인베스팅 SPX 일간 CSV → 날짜별 close·open·high·low (한글판·영문판 모두, 열 순서: 날짜 종가 시가 고가 저가)."""
    px = pd.read_csv(path, thousands=",", encoding="utf-8-sig")
    px.columns = [c.strip().strip('"') for c in px.columns]
    d = px.columns[0]
    px[d] = pd.to_datetime(px[d].astype(str).str.replace(" ", ""), errors="coerce")
    px = px.dropna(subset=[d]).set_index(d).sort_index()
    out = pd.DataFrame({k: pd.to_numeric(px[px.columns[i]].astype(str).str.replace(",", ""), errors="coerce")
                        for i, k in enumerate(("close", "open", "high", "low"))})
    out.index.name = "date"
    return out.dropna(subset=["close"])


def load_spx_close(path):
    return load_spx_ohlc(path)["close"]


def load_day(d):
    """그날 0DTE 호가 = 원본(±3%) + 보충 행사가 파일(있으면) 합친 것."""
    from .config import OPT_DIR, EXT_DIR
    dfs = [db.DBNStore.from_file(str(p)).to_df() for p in (OPT_DIR / f"{d:%Y-%m-%d}.dbn.zst", EXT_DIR / f"{d:%Y-%m-%d}.dbn.zst")
           if p.exists() and p.stat().st_size > 0]
    return pd.concat(dfs).sort_index() if len(dfs) > 1 else dfs[0]


def load_spx_ref(path):
    """그날 행사가 범위를 잡는 기준 = 전일 종가."""
    return load_spx_close(path).shift(1).dropna()


# ---------- 0DTE 심볼 ----------
def day_symbols(d, ref, band):
    lo = math.floor(ref * (1 - band) / 5) * 5
    hi = math.ceil(ref * (1 + band) / 5) * 5
    return [f"SPXW  {d:%y%m%d}{cp}{int(k * 1000):08d}" for k in range(lo, hi + 5, 5) for cp in "CP"]


def day_params(d, ref, band):
    s = pd.Timestamp(f"{d:%Y-%m-%d} 09:30", tz=NY).tz_convert("UTC")
    e = pd.Timestamp(f"{d:%Y-%m-%d} 16:01", tz=NY).tz_convert("UTC")
    return dict(dataset="OPRA.PILLAR", schema="cbbo-1m", stype_in="raw_symbol",
                symbols=day_symbols(d, ref, band), start=s, end=e)


# ---------- Black-76 ----------
def b76(F, K, T, s, call):
    sq = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    c = F * ndtr(d1) - K * ndtr(d1 - sq)
    return np.where(call, c, c - F + K)


def implied_vol(p, F, K, T, call, lo=1e-3, hi=8.0, it=60):
    intr = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    ok = np.isfinite(p) & np.isfinite(F) & (T > 0) & (p > intr + 1e-6)
    a, b = np.full(p.shape, lo), np.full(p.shape, hi)
    Fs, Ks, Ts, ps = (np.where(ok, x, y) for x, y in ((F, 1.0), (K, 1.0), (T, 1.0), (p, 0.0)))
    for _ in range(it):
        m = 0.5 * (a + b)
        up = b76(Fs, Ks, Ts, m, call) > ps
        b, a = np.where(up, m, b), np.where(up, a, m)
    iv = 0.5 * (a + b)
    iv[~ok | (iv >= hi * 0.999)] = np.nan
    return iv


def greeks(F, K, T, s, call):
    sq = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    n = np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    delta = np.where(call, ndtr(d1), ndtr(d1) - 1)
    gamma = n / (F * sq)
    vega = F * n * np.sqrt(T)
    theta = -F * n * s / (2 * np.sqrt(T))          # 연 단위
    return delta, gamma, vega, theta


# ---------- 하루치 손익 분해 ----------
def mid_matrix(df, d, max_spread=0.5):
    bid, ask = df["bid_px_00"].astype(float), df["ask_px_00"].astype(float)
    good = (bid > 0) & (ask >= bid) & ((ask - bid) <= np.maximum(max_spread * (ask + bid) / 2, 0.20))
    q = pd.DataFrame({"t": df.index.tz_convert(NY).floor("min"), "symbol": df["symbol"].values,
                      "mid": ((bid + ask) / 2).values})[good.values]
    grid = pd.date_range(f"{d:%Y-%m-%d} 09:31", f"{d:%Y-%m-%d} 15:59", freq="1min", tz=NY)
    return q.pivot_table(index="t", columns="symbol", values="mid", aggfunc="last").reindex(grid).ffill(limit=5)


def parse_osi(cols):
    s = pd.Index(cols).str.strip()
    return s.str[-9].values, (s.str[-8:].astype(float) / 1000).values


def forward_parity(M, cp, K, n_near=5):
    """풋콜패리티 F = K + C - P, |C-P| 가장 작은 행사가 n개 중앙값"""
    Ks = np.unique(K)
    C = np.full((len(M), len(Ks)), np.nan); P = C.copy()
    pos = np.searchsorted(Ks, K)
    for j in range(M.shape[1]):
        (C if cp[j] == "C" else P)[:, pos[j]] = M.values[:, j]
    syn = Ks[None, :] + C - P
    dist = np.abs(C - P); dist[~np.isfinite(dist)] = np.inf
    near = np.argsort(dist, axis=1)[:, :n_near]
    pick = np.take_along_axis(syn, near, axis=1)
    pick[~np.isfinite(np.take_along_axis(dist, near, axis=1))] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(pick, axis=1)


def attribute_day(df, d, step=1, bucket="30min", min_mid=0.20, smooth=1):
    """dV ≈ Δ·dF + ½Γ·dF² + Θ·dt + Vega·dσ + 잔차  (step분 간격)"""
    M = mid_matrix(df, d)
    if M.shape[1] < 10 or M.notna().sum().sum() < 1000:
        return pd.DataFrame()
    cp, K = parse_osi(M.columns)
    F = forward_parity(M, cp, K)
    M, F = M.iloc[::step], F[::step]
    t = M.index
    T = ((pd.Timestamp(f"{d:%Y-%m-%d} 16:00", tz=NY) - t).total_seconds().values / 60 / YEAR_MIN)[:, None]
    V = M.values
    Fm = F[:, None] * np.ones_like(V)
    Km = np.broadcast_to(K, V.shape)
    call = np.broadcast_to(cp == "C", V.shape)
    Tm = np.broadcast_to(T, V.shape)

    iv_raw = implied_vol(V, Fm, Km, Tm, call)
    # IV를 최근 smooth개 봉 중앙값으로 평활 → 호가 튐으로 생긴 가짜 dσ를 베가가 아니라 잔차로 보냄
    iv = pd.DataFrame(iv_raw).rolling(smooth, min_periods=1).median().values if smooth > 1 else iv_raw
    dl, g, vg, th = greeks(Fm, Km, Tm, iv, call)
    dV, dF, dsig = V[1:] - V[:-1], (F[1:] - F[:-1])[:, None], iv[1:] - iv[:-1]
    parts = {"delta": dl[:-1] * dF, "theta": th[:-1] * (step / YEAR_MIN),
             "gamma": 0.5 * g[:-1] * dF ** 2, "vega": vg[:-1] * dsig}

    valid = np.isfinite(dV) & (V[:-1] >= min_mid)
    for p in parts.values():
        valid &= np.isfinite(p)
    m = np.log(Km[:-1] / Fm[:-1])
    grp = np.full(dV.shape, "", dtype=object)
    grp[np.abs(m) < 0.0015] = "ATM"
    grp[(call[:-1] & (m >= 0.0015) & (m < 0.01)) | (~call[:-1] & (m <= -0.0015) & (m > -0.01))] = "OTM"
    valid &= grp != ""

    r, c = np.nonzero(valid)
    out = pd.DataFrame({"t": t[:-1][r], "group": grp[r, c], "prem": V[:-1][r, c], "dV": dV[r, c],
                        **{k: v[r, c] for k, v in parts.items()}})
    out["resid"] = out["dV"] - out[TERMS].sum(axis=1)
    out["bucket"] = out["t"].dt.floor(bucket).dt.strftime("%H:%M")
    return out


def summ(x):
    r = {"n": len(x), "prem": x["prem"].sum(), "dV": x["dV"].sum(),
         "dV2": (x["dV"] ** 2).sum(), "resid2": (x["resid"] ** 2).sum()}
    for k in TERMS:
        r[f"{k}_sum"], r[f"{k}_abs"] = x[k].sum(), x[k].abs().sum()
    r["resid_sum"], r["resid_abs"] = x["resid"].sum(), x["resid"].abs().sum()
    return pd.Series(r)


def finalize(agg):
    a = agg.copy()
    tot = a[[f"{k}_abs" for k in ALL]].sum(axis=1)
    for k in ALL:
        a[f"{k}_share"] = a[f"{k}_abs"] / tot
        a[f"{k}_bp"] = a[f"{k}_sum"] / a["prem"] * 1e4
    a["actual_bp"] = a["dV"] / a["prem"] * 1e4
    a["R2"] = 1 - a["resid2"] / a["dV2"]
    return a


# ---------- 차트 ----------
def plot_greeks(res, group="ATM", save=None):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP"):
        if f in have:
            plt.rcParams["font.family"] = f; break
    plt.rcParams["axes.unicode_minus"] = False

    a = res.loc[group]
    x = np.arange(len(a))
    top = dict(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, frameon=False)
    fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    bottom = np.zeros(len(a))
    for k in ALL:
        v = a[f"{k}_share"].values
        ax[0].bar(x, v, bottom=bottom, color=COLORS[k], label=NAMES[k], width=0.85)
        for xi, (b, h) in enumerate(zip(bottom, v)):
            if h >= 0.06:
                ax[0].text(xi, b + h / 2, f"{h:.0%}", ha="center", va="center", fontsize=7, color="white")
        bottom += v
    ax[0].set_ylim(0, 1)
    ax[0].set_title(f"SPXW 0DTE {group}: 시간대별 그릭 비중 (|기여| 합 = 100%)", pad=26)
    ax[0].legend(**top)

    for k in ALL:
        ax[1].plot(x, a[f"{k}_bp"], color=COLORS[k], marker="o", ms=3, label=NAMES[k])
    ax[1].plot(x, a["actual_bp"], color="black", ls="--", label="실제 손익")
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_title("롱 옵션 평균 손익 기여 (프리미엄 대비 bp / 분)", pad=26)
    ax[1].legend(**top)
    ax[1].set_xticks(x); ax[1].set_xticklabels(a.index, rotation=45)
    ax[1].set_xlabel("뉴욕 시간 (ET)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    plt.show()

