"""
engine_v2.py  ── v4  (Recalibrated Weights — Research-Aligned)
═══════════════════════════════════════════════════════════════════════════════
Cải tiến so với v3:

  [1] SCORE MOMENTUM — tái phân bổ trọng số theo nghiên cứu:
      r6m_adj   25%  (momentum bền vững — Jegadeesh & Titman)
      vol_ratio 30%  (smart money breakout — trừ điểm nếu > 4× để tránh phân phối đỉnh)
      ma_score  20%  (cấu trúc xu hướng — bắt buộc Price > MA50 & MA50 > MA200)
      rsi_score 10%  (timing / penalty khi RSI > 80 — giảm vì đa cộng tuyến với MA)
      macd_bull  5%  (giảm — tránh double-count khi đã tính MA)
      bb_pct     5%  (giảm — bám sát vol_ratio & momentum)
      adx_bull   5%  (giữ nhỏ — xác nhận cường độ xu hướng)

  [2] SCORE FUNDAMENTAL — tái phân bổ + nâng CFO:
      eps_yoy   20%  (EPS YoY > 20%)
      rev_yoy   15%  (Rev YoY > 15% — xác nhận tăng trưởng thực)
      eps_accel 15%  (gia tốc EPS — catalyst mạnh nhất, CANSLIM)
      cfo_pos   15%  (CFO dương ≥ 2 quý liên tiếp — Sloan 1996 Accrual Anomaly)
      gm_trend  10%  (biên lãi gộp — pricing power)
      roe_now   10%  (hiệu suất vốn so với trung bình ngành)
      ni_pos4q   5%  (baseline: lợi nhuận dương ≥ 3/4 quý)
      de_ratio   5%  (giảm — đòn bẩy cao không xấu nếu CFO khỏe)

  [3] RSI SCORE — vùng tối ưu 50-65, trừ điểm mạnh nếu > 80

  [4] VOL_RATIO PENALTY — trừ điểm nếu vol_ratio > 4× (tránh phiên phân phối đỉnh)

  [5] MA_SCORE — thêm điều kiện bắt buộc Price > MA50 AND MA50 > MA200

  [6] CFO_POS — nâng lên 2 quý liên tiếp dương (thay vì chỉ quý gần nhất)

Các thay đổi khác giữ nguyên (ADX Wilder, RSI Wilder, ATR, calibrated weights).
API giữ nguyên — chỉ cần thay file này, không đổi main.py hay Backtest.py.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE      = Path(__file__).parent
DATA_DIR  = BASE / "data"
RAW_DIR   = DATA_DIR / "raw"        # data/raw/   — file xlsx gốc
PROC_DIR  = DATA_DIR / "processed"  # data/processed/ — parquet cache

# Raw source files (data/raw/)
PRICE_FILE = RAW_DIR / "prices.xlsx"
BCTC_FILE  = RAW_DIR / "BCTC_THEO_QUÝ.xlsx"
INFO_FILE  = RAW_DIR / "HISTORICAL_PRICES.xlsx"
INDEX_FILE = DATA_DIR / "INDEX.xlsx"   # INDEX.xlsx nằm thẳng trong data/

# Parquet cache (data/processed/) — được tạo tự động nếu chưa có
PRICE_PQ = PROC_DIR / "prices.parquet"
BCTC_PQ  = PROC_DIR / "financials.parquet"
INFO_PQ  = PROC_DIR / "info.parquet"
INDEX_PQ = PROC_DIR / "index.parquet"

# Weights & ML model
WEIGHTS_FILE = DATA_DIR / "calibrated_weights.json"

# Tạo thư mục processed/ nếu chưa có
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Default weights (dùng khi chưa calibrate)
DEFAULT_WEIGHTS = {
    "score_momentum":    0.30,
    "score_rs":          0.20,
    "score_fundamental": 0.25,
    "score_seasonal":    0.25,
}


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD CALIBRATED WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def load_scoring_weights() -> dict:
    """
    Load weights từ calibrated_weights.json nếu tồn tại.
    Fallback về DEFAULT_WEIGHTS.
    """
    if WEIGHTS_FILE.exists():
        try:
            with open(WEIGHTS_FILE, encoding="utf-8") as f:
                w = json.load(f)
            keys = ["score_momentum", "score_rs", "score_fundamental", "score_seasonal"]
            if all(k in w for k in keys):
                total = sum(w[k] for k in keys)
                result = {k: w[k] / total for k in keys}  # re-normalise
                method = w.get("calibration_method", "unknown")
                print(f"✓ Calibrated weights loaded (method: {method}): "
                      f"mom={result['score_momentum']:.2f} "
                      f"rs={result['score_rs']:.2f} "
                      f"fund={result['score_fundamental']:.2f} "
                      f"sea={result['score_seasonal']:.2f}")
                return result
        except Exception as e:
            print(f"⚠ Không load được calibrated_weights.json: {e} — dùng default")
    return DEFAULT_WEIGHTS.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA  (Auto-Cache Parquet)
# ══════════════════════════════════════════════════════════════════════════════

def load_prices() -> pd.DataFrame:
    # Ưu tiên parquet đã có (nhanh hơn 10×)
    if PRICE_PQ.exists():
        xlsx_mtime = PRICE_FILE.stat().st_mtime if PRICE_FILE.exists() else 0
        if PRICE_PQ.stat().st_mtime >= xlsx_mtime:
            return pd.read_parquet(PRICE_PQ)
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy {PRICE_FILE}. Chạy fetch_prices.py trước.")
    df = pd.read_excel(PRICE_FILE, sheet_name="prices")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker_clean", "date"]).reset_index(drop=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRICE_PQ, index=False)
    return df


def load_index() -> pd.DataFrame:
    if INDEX_PQ.exists():
        xlsx_mtime = INDEX_FILE.stat().st_mtime if INDEX_FILE.exists() else 0
        if INDEX_PQ.stat().st_mtime >= xlsx_mtime:
            return pd.read_parquet(INDEX_PQ)
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy {INDEX_FILE}.")
    df = pd.read_excel(INDEX_FILE, sheet_name="Table Data")
    df = df.iloc[:, :2]
    df.columns = ["date", "vni"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "vni"]).sort_values("date").reset_index(drop=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INDEX_PQ, index=False)
    return df


def load_company_info() -> pd.DataFrame:
    if INFO_PQ.exists() and INFO_FILE.exists() and INFO_PQ.stat().st_mtime > INFO_FILE.stat().st_mtime:
        return pd.read_parquet(INFO_PQ)
    if INFO_PQ.exists() and not INFO_FILE.exists():
        # Dùng parquet cache nếu xlsx gốc không tìm thấy
        return pd.read_parquet(INFO_PQ)

    if INFO_FILE.exists():
        # HISTORICAL_PRICES.xlsx — Sheet1 có 8 cột
        df = pd.read_excel(INFO_FILE, sheet_name="Sheet1")
        # Lấy 6 cột đầu tiên (bỏ cột date dư)
        df = df.iloc[:, :7]
        df.columns = ["ticker", "company_name", "country", "industry", "sector",
                      "ipo_date", "sector_vn"]
        df["ticker_clean"] = df["ticker"].str.split(".").str[0]
        df["exchange"]     = df["ticker"].str.split(".").str[1]
        df = df[["ticker_clean", "company_name", "industry", "sector",
                 "sector_vn", "exchange"]].drop_duplicates("ticker_clean")
    else:
        # Fallback: dùng COMP_INFO.csv nếu không có HISTORICAL_PRICES.xlsx
        comp_csv = DATA_DIR / "COMP_INFO.csv"
        if not comp_csv.exists():
            comp_csv = DATA_DIR.parent / "COMP_INFO.csv"
        if comp_csv.exists():
            raw = pd.read_csv(comp_csv)
            df  = pd.DataFrame({
                "ticker_clean": raw["symbol"],
                "company_name": raw.get("en_organ_name", raw.get("organ_name", "")),
                "industry":     raw.get("type", ""),
                "sector":       raw.get("exchange", "Unknown"),
                "sector_vn":    raw.get("exchange", "Unknown"),
                "exchange":     raw.get("exchange", ""),
            }).drop_duplicates("ticker_clean")
        else:
            print("⚠ Không tìm thấy HISTORICAL_PRICES.xlsx hay COMP_INFO.csv — dùng info rỗng")
            df = pd.DataFrame(columns=["ticker_clean","company_name","industry",
                                       "sector","sector_vn","exchange"])

    df.to_parquet(INFO_PQ, index=False)
    return df


def _read_bctc_sheet(sheet: str, rename: dict) -> pd.DataFrame:
    df = pd.read_excel(BCTC_FILE, sheet_name=sheet)
    df = df.rename(columns={"Unnamed: 0": "ticker", "Unnamed: 1": "date"})
    df["date"]         = pd.to_datetime(df["date"], errors="coerce")
    df["ticker_clean"] = df["ticker"].str.split(".").str[0]
    keep = ["ticker_clean", "date"] + [c for c in rename if c in df.columns]
    return df[keep].rename(columns=rename)


def load_financials() -> pd.DataFrame:
    if BCTC_PQ.exists():
        xlsx_mtime = BCTC_FILE.stat().st_mtime if BCTC_FILE.exists() else 0
        if BCTC_PQ.stat().st_mtime >= xlsx_mtime:
            return pd.read_parquet(BCTC_PQ)
    is2 = _read_bctc_sheet("IS2", {
        "Cost of Revenues - Total":                    "cogs",
        "Gross Profit - Industrials/Property - Total": "gross_profit",
        "Net Income after Tax":                        "net_income",
        "Income before Taxes":                         "ebt",
    })
    is3 = _read_bctc_sheet("IS3", {
        "EPS - Basic - incl Extraordinary Items, Common - Total": "eps",
        "DPS - Common - Gross - Issue - By Announcement Date":    "dps",
    })
    is1 = _read_bctc_sheet("IS1", {"Revenue from Business Activities - Total": "revenue"})
    bs6 = _read_bctc_sheet("BS6", {
        "Total Liabilities":         "total_liabilities",
        "Common Equity - Total":     "equity",
        "Total Liabilities & Equity":"total_assets",
    })
    bs8 = _read_bctc_sheet("BS8", {"Net Debt": "net_debt"})
    cf1 = _read_bctc_sheet("CF1", {
        "Net Cash Flow from Operating Activities": "cfo",
        "Capital Expenditures - Total":            "capex",
    })
    cf3 = _read_bctc_sheet("CF3", {"Free Cash Flow to Equity": "fcfe"})

    dfs    = [is1, is2, is3, bs6, bs8, cf1, cf3]
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["ticker_clean", "date"], how="outer")

    merged = merged.dropna(subset=["ticker_clean", "date"])
    merged = merged.sort_values(["ticker_clean", "date"]).reset_index(drop=True)
    merged["gross_margin"] = np.where(merged["revenue"] > 0,
                                      merged["gross_profit"] / merged["revenue"].replace(0, np.nan),
                                      np.nan)
    merged["roe"]    = np.where(merged["equity"] > 0,
                                merged["net_income"] / merged["equity"].replace(0, np.nan), np.nan)
    merged["de_ratio"] = np.where(merged["equity"] > 0,
                                  merged["total_liabilities"] / merged["equity"].replace(0, np.nan), np.nan)
    merged.to_parquet(BCTC_PQ, index=False)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: np.ndarray, n: int) -> np.ndarray:
    k, out = 2 / (n + 1), np.full(len(series), np.nan)
    start  = next((i for i, v in enumerate(series) if not np.isnan(v)), None)
    if start is None or len(series) - start < n:
        return out
    out[start] = series[start]
    for i in range(start + 1, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def _wilder_smooth(series: np.ndarray, n: int) -> np.ndarray:
    """
    Wilder smoothing (= EMA với alpha = 1/n).
    Chuẩn được dùng trong ATR, ADX, RSI Wilder.
    """
    out   = np.full(len(series), np.nan)
    start = next((i for i, v in enumerate(series) if not np.isnan(v)), None)
    if start is None or len(series) - start < n:
        return out
    # Seed = SMA đầu tiên
    out[start + n - 1] = np.nanmean(series[start: start + n])
    for i in range(start + n, len(series)):
        if not np.isnan(series[i]):
            out[i] = out[i - 1] * (n - 1) / n + series[i] / n
    return out


def calc_adx_wilder(high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, n: int = 14):
    """
    ADX chuẩn Wilder (fix từ rolling mean trong v2).
    Returns (adx, plus_di, minus_di).
    """
    length = len(close)
    tr    = np.full(length, np.nan)
    pdm   = np.full(length, 0.0)
    mdm   = np.full(length, 0.0)

    for i in range(1, length):
        hl  = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

        up   = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm[i] = up   if (up > down and up > 0)   else 0.0
        mdm[i] = down if (down > up and down > 0) else 0.0

    atr_w  = _wilder_smooth(tr,  n)
    pdm_w  = _wilder_smooth(pdm, n)
    mdm_w  = _wilder_smooth(mdm, n)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di  = np.where(atr_w > 0, 100 * pdm_w / atr_w, 0.0)
        minus_di = np.where(atr_w > 0, 100 * mdm_w / atr_w, 0.0)
        dx       = np.where((plus_di + minus_di) > 0,
                            100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9),
                            0.0)

    adx = _wilder_smooth(dx, n)
    return adx, plus_di, minus_di


def _rsi_wilder(close: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI dùng Wilder smoothing (chuẩn hơn rolling mean)."""
    delta = np.diff(close.astype(float), prepend=np.nan)
    gains = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    avg_gain = _wilder_smooth(gains, n)
    avg_loss = _wilder_smooth(loss,  n)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs  = np.where(avg_loss > 0, avg_gain / avg_loss, np.inf)
        rsi = 100 - 100 / (1 + rs)
    rsi[:n] = np.nan
    return rsi


def calc_indicators(grp: pd.DataFrame) -> pd.DataFrame:
    c   = grp["close"].values.astype(float)
    h   = grp["high"].values.astype(float)  if "high"   in grp.columns else c.copy()
    l   = grp["low"].values.astype(float)   if "low"    in grp.columns else c.copy()
    v   = grp["volume"].values.astype(float)
    n   = len(c)
    out = grp.copy()

    # MAs
    for p in [10, 20, 50, 100, 200]:
        out[f"ma{p}"] = pd.Series(c).rolling(p).mean().values
    out["ema20"] = _ema(c, 20)
    out["ema50"] = _ema(c, 50)

    # MACD
    ema12  = _ema(c, 12)
    ema26  = _ema(c, 26)
    macd   = ema12 - ema26
    signal = _ema(np.where(np.isnan(macd), np.nan, macd), 9)
    out["macd"]       = macd
    out["macd_signal"] = signal
    out["macd_hist"]   = macd - signal
    out["macd_cross"]  = ((macd > signal) & (np.roll(macd, 1) <= np.roll(signal, 1))).astype(int)

    # [FIX] RSI Wilder smoothing (thay rolling mean)
    out["rsi"] = _rsi_wilder(c, 14)

    # Bollinger Bands
    bb_mid = pd.Series(c).rolling(20).mean()
    bb_std = pd.Series(c).rolling(20).std()
    out["bb_upper"] = (bb_mid + 2 * bb_std).values
    out["bb_lower"] = (bb_mid - 2 * bb_std).values
    out["bb_mid"]   = bb_mid.values
    out["bb_pct"]   = ((pd.Series(c) - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)).clip(0, 1).values

    # Stochastic
    low14  = pd.Series(l).rolling(14).min()
    high14 = pd.Series(h).rolling(14).max()
    k_raw  = 100 * (pd.Series(c) - low14) / (high14 - low14 + 1e-9)
    out["stoch_k"] = k_raw.values
    out["stoch_d"] = k_raw.rolling(3).mean().values

    # ATR (Wilder)
    tr_arr = np.full(n, np.nan)
    for i in range(1, n):
        tr_arr[i] = max(h[i] - l[i],
                        abs(h[i] - c[i - 1]),
                        abs(l[i] - c[i - 1]))
    out["atr"]     = _wilder_smooth(tr_arr, 14)
    out["atr_pct"] = np.where(c > 0, out["atr"] / c * 100, np.nan)

    # [FIX] ADX Wilder smoothing (thay rolling mean)
    adx, plus_di, minus_di = calc_adx_wilder(h, l, c, 14)
    out["adx"]      = adx
    out["plus_di"]  = plus_di
    out["minus_di"] = minus_di

    # Volume
    out["vol_ma20"]  = pd.Series(v).rolling(20).mean().values
    out["vol_ma60"]  = pd.Series(v).rolling(60).mean().values
    out["vol_ratio"] = np.where(out["vol_ma60"] > 0, v / out["vol_ma60"], np.nan)
    out["turnover_ma20"] = pd.Series(v * c).rolling(20).mean().values
    # OBV
    obv = np.zeros(n)
    for i in range(1, n):
        obv[i] = obv[i-1] + v[i] if c[i] > c[i-1] else obv[i-1] - v[i] if c[i] < c[i-1] else obv[i-1]
    out["obv"] = obv

    # Returns
    out["ret_1d"]  = pd.Series(c).pct_change(1).values
    out["ret_5d"]  = pd.Series(c).pct_change(5).values
    out["ret_21d"] = pd.Series(c).pct_change(21).values
    out["ret_63d"] = pd.Series(c).pct_change(63).values

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SEASONAL (no look-ahead)
# ══════════════════════════════════════════════════════════════════════════════

def compute_seasonal(prices: pd.DataFrame, ticker: str,
                     ref_date: pd.Timestamp = None) -> pd.DataFrame:
    grp = prices[prices["ticker_clean"] == ticker].copy()
    if ref_date is not None:
        grp = grp[grp["date"] <= ref_date]
    if len(grp) < 60:
        return pd.DataFrame(columns=["month", "avg_return", "win_rate", "count", "edge"])
    grp = grp.sort_values("date")
    grp["month"]     = grp["date"].dt.month
    grp["year"]      = grp["date"].dt.year
    grp["ret_month"] = grp["close"].pct_change(21)
    monthly = grp.groupby(["year", "month"])["ret_month"].last().reset_index()
    stats   = monthly.groupby("month")["ret_month"].agg(
        avg_return="mean",
        win_rate=lambda x: (x > 0).mean(),
        count="count",
    ).reset_index()
    stats["edge"] = stats["avg_return"] - stats["avg_return"].mean()
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  MARKET REGIME
# ══════════════════════════════════════════════════════════════════════════════

def compute_market_regime(index_df: pd.DataFrame,
                          ref_date: pd.Timestamp) -> dict:
    idx = index_df[index_df["date"] <= ref_date].sort_values("date")
    if len(idx) < 60:
        return {"regime": "unknown", "min_score_adjust": 0, "rs_adjust": 0}

    c    = idx["vni"].values
    ma20 = pd.Series(c).rolling(20).mean().iloc[-1]
    ma50 = pd.Series(c).rolling(50).mean().iloc[-1]
    ma200= pd.Series(c).rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
    last = c[-1]
    r63  = (last / c[-64] - 1) if len(c) >= 64 else 0

    if last > ma20 > ma50 and r63 > 0.05:
        regime, min_adj, rs_adj = "bull",     -5, 0
    elif last < ma20 < ma50 and r63 < -0.05:
        regime, min_adj, rs_adj = "bear",     +10, +5
    else:
        regime, min_adj, rs_adj = "sideways", +5,  0

    breadth = 1.0 if last > ma50 else 0.5 if not np.isnan(ma200) and last > ma200 else 0.0

    return {
        "regime":           regime,
        "vni_last":         float(last),
        "vni_ma20":         float(ma20),
        "vni_ma50":         float(ma50),
        "vni_r63":          float(r63),
        "breadth":          breadth,
        "min_score_adjust": min_adj,
        "rs_adjust":        rs_adj,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SUPPORT / RESISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def compute_support_resistance(prices_grp: pd.DataFrame,
                               current_price: float,
                               lookback: int = 252,
                               tolerance_pct: float = 0.02) -> dict:
    df = prices_grp.tail(lookback).copy()
    _empty = {
        "nearest_support_pct": np.nan, "nearest_resist_pct": np.nan,
        "at_support": 0, "at_resistance": 0,
        "support_strength": 0.5, "resistance_strength": 0.5,
    }
    if len(df) < 30:
        return _empty

    c, h, l = df["close"].values, df["high"].values, df["low"].values
    window  = 5
    swing_highs, swing_lows = [], []
    for i in range(window, len(c) - window):
        if h[i] == max(h[i - window: i + window + 1]):
            swing_highs.append(h[i])
        if l[i] == min(l[i - window: i + window + 1]):
            swing_lows.append(l[i])

    def cluster(levels, tol):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for price in levels[1:]:
            if (price - clusters[-1][-1]) / clusters[-1][-1] < tol:
                clusters[-1].append(price)
            else:
                clusters.append([price])
        return [np.mean(cl) for cl in clusters]

    resist_lvl  = cluster(swing_highs, tolerance_pct)
    support_lvl = cluster(swing_lows,  tolerance_pct)

    resist_pcts  = [(lv - current_price) / current_price for lv in resist_lvl if lv > current_price]
    support_pcts = [(lv - current_price) / current_price for lv in support_lvl if lv < current_price]

    nearest_resist  = min(resist_pcts,  default=np.nan)
    nearest_support = max(support_pcts, default=np.nan)

    at_support    = int(not np.isnan(nearest_support) and abs(nearest_support) <= 0.02)
    at_resistance = int(not np.isnan(nearest_resist)  and nearest_resist <= 0.02)

    def level_strength(level_price, price_arr, tol=0.015):
        touches = []
        for i in range(1, len(price_arr) - 1):
            if abs(price_arr[i] - level_price) / level_price <= tol:
                if (price_arr[i] - price_arr[i-1]) * (price_arr[i+1] - price_arr[i]) < 0:
                    touches.append(1)
                else:
                    touches.append(0)
        return np.mean(touches) if len(touches) >= 2 else 0.5

    sup_str = level_strength(current_price * (1 + nearest_support), c) if not np.isnan(nearest_support) else 0.5
    res_str = level_strength(current_price * (1 + nearest_resist),  c) if not np.isnan(nearest_resist)  else 0.5

    return {
        "nearest_support_pct":  float(nearest_support) if not np.isnan(nearest_support) else np.nan,
        "nearest_resist_pct":   float(nearest_resist)  if not np.isnan(nearest_resist)  else np.nan,
        "at_support":           at_support,
        "at_resistance":        at_resistance,
        "support_strength":     float(sup_str),
        "resistance_strength":  float(res_str),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PRICE PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

def compute_price_patterns(c: np.ndarray, v: np.ndarray,
                            vol_ma20: np.ndarray) -> dict:
    if len(c) < 10:
        return {k: 0 for k in ["streak_up", "streak_down", "streak_magnitude",
                                "vol_trend_in_streak", "exhaustion_signal",
                                "vol_breakout", "inside_bars"]}
    streak_up = streak_down = 0
    for i in range(len(c) - 1, max(len(c) - 20, 0), -1):
        if c[i] > c[i - 1]:
            if streak_down == 0: streak_up   += 1
            else: break
        elif c[i] < c[i - 1]:
            if streak_up == 0:   streak_down += 1
            else: break
        else:
            break

    streak_len   = max(streak_up, streak_down)
    start_idx    = max(len(c) - 1 - streak_len, 0)
    streak_mag   = (c[-1] / c[start_idx] - 1) if streak_len > 0 else 0.0

    vol_trend    = 1.0
    if streak_len >= 4:
        half      = streak_len // 2
        vol_first = np.mean(v[-streak_len: -half]) if half > 0 else 1
        vol_last  = np.mean(v[-half:])
        vol_trend = vol_last / vol_first if vol_first > 0 else 1.0

    exhaustion   = int(streak_len >= 5 and vol_trend < 0.8)
    vol_breakout = int(np.any(v[-3:] / (vol_ma20[-3:] + 1e-9) > 2.0))

    inside_bars = 0
    for i in range(len(c) - 1, max(len(c) - 5, 1), -1):
        if abs(c[i] / c[i - 1] - 1) < 0.005:
            inside_bars += 1
        else:
            break

    return {
        "streak_up":           int(streak_up),
        "streak_down":         int(streak_down),
        "streak_magnitude":    float(streak_mag),
        "vol_trend_in_streak": float(vol_trend),
        "exhaustion_signal":   int(exhaustion),
        "vol_breakout":        int(vol_breakout),
        "inside_bars":         int(inside_bars),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR ROTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_sector_rotation(scores_df: pd.DataFrame,
                             lookback_col: str = "rs_vs_mkt") -> pd.DataFrame:
    if scores_df.empty or "sector" not in scores_df.columns:
        return scores_df
    sector_rs = scores_df.groupby("sector")[lookback_col].mean().reset_index()
    sector_rs.columns = ["sector", "sector_avg_rs"]
    sector_rs["sector_rank"] = sector_rs["sector_avg_rs"].rank(ascending=False,
                                                                na_option="bottom").astype(int)
    scores_df = scores_df.merge(sector_rs[["sector", "sector_avg_rs", "sector_rank"]],
                                on="sector", how="left")
    top3 = sector_rs.nsmallest(3, "sector_rank")["sector"].tolist()
    scores_df["in_leading_sector"] = scores_df["sector"].isin(top3).astype(int)
    return scores_df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_scores(prices: pd.DataFrame,
                       financials: pd.DataFrame,
                       index_df: pd.DataFrame,
                       info: pd.DataFrame,
                       target_month: int,
                       ref_date: pd.Timestamp = None) -> pd.DataFrame:
    if ref_date is None:
        ref_date = prices["date"].max()

    regime_info = compute_market_regime(index_df, ref_date)
    regime      = regime_info["regime"]

    idx_sub = index_df[index_df["date"] <= ref_date].sort_values("date")
    vni_r3m = (idx_sub["vni"].iloc[-1] / idx_sub["vni"].iloc[-64] - 1) \
              if len(idx_sub) >= 64 else 0.0
    vni_r1m = (idx_sub["vni"].iloc[-1] / idx_sub["vni"].iloc[-22] - 1) \
              if len(idx_sub) >= 22 else 0.0

    # [NEW] Load calibrated weights mỗi lần tính (tự pick up file mới)
    W = load_scoring_weights(regime)

    tickers = prices["ticker_clean"].unique()
    rows    = []

    for ticker in tickers:
        grp = prices[prices["ticker_clean"] == ticker].sort_values("date")
        grp = grp[grp["date"] <= ref_date]
        if len(grp) < 60:
            continue

        grp = calc_indicators(grp)  # ← kết quả assign vào grp
        last = grp.iloc[-1]  # ← last lấy từ grp mới
        turnover_ma20 = float(last.get("turnover_ma20", 0))
        if turnover_ma20 < 3_000_000_000:
            continue
        c = grp["close"].values
        v = grp["volume"].values

        r1m     = c[-1] / c[-22] - 1  if len(c) >= 22  else np.nan
        r3m     = c[-1] / c[-64] - 1  if len(c) >= 64  else np.nan
        r6m     = c[-1] / c[-127] - 1 if len(c) >= 127 else np.nan
        r6m_adj = (c[-1] / c[-127] - 1) - (c[-1] / c[-22] - 1) if len(c) >= 127 else np.nan

        vol_ratio = float(last.get("vol_ratio", np.nan))
        ma_score  = sum([
            1 if not np.isnan(last.get(f"ma{p}", np.nan)) and c[-1] > last[f"ma{p}"] else 0
            for p in [20, 50, 100, 200]
        ]) + (1 if not np.isnan(last.get("ema20", np.nan)) and
                   not np.isnan(last.get("ema50", np.nan)) and
                   last["ema20"] > last["ema50"] else 0)

        rsi       = float(last.get("rsi", np.nan))
        macd_bull = int(float(last.get("macd_hist", 0)) > 0)
        macd_hist = float(last.get("macd_hist", 0))
        bb_pct    = float(last.get("bb_pct", 0.5))
        stoch_k   = float(last.get("stoch_k", 50))

        # RSI score — vùng tối ưu 50-65 (timing entry), penalty mạnh khi > 80
        rsi_score = (
            90 if (not np.isnan(rsi) and 50 <= rsi <= 65) else   # Vùng lý tưởng
            70 if (not np.isnan(rsi) and 65 < rsi <= 75)  else   # Ổn, hơi căng
            75 if (not np.isnan(rsi) and 40 <= rsi < 50)  else   # Sắp bứt phá
            60 if (not np.isnan(rsi) and rsi < 40)        else   # Oversold nhẹ
            15 if (not np.isnan(rsi) and rsi > 80)        else 50 # Penalty: quá căng
        )

        rs_vs_mkt = r3m - vni_r3m if not np.isnan(r3m) else np.nan
        rs_trend  = np.nan
        if len(c) >= 43:
            r1m_prev = c[-22] / c[-43] - 1
            rs_trend = (r1m - vni_r1m) - (r1m_prev - vni_r1m) if not np.isnan(r1m_prev) else np.nan

        # Fundamentals (30-day reporting lag)
        fin_g = financials[financials["ticker_clean"] == ticker].sort_values("date")
        fin_g = fin_g[(fin_g["date"] + pd.DateOffset(days=30)) <= ref_date]

        eps_yoy = eps_accel = rev_yoy = gm_trend = roe_now = np.nan
        cfo_pos = ni_pos4q = 0
        de_ratio = np.nan

        if len(fin_g) >= 4:
            last_fin   = fin_g.iloc[-1]
            eps_now    = last_fin.get("eps", np.nan)
            eps_4ago   = fin_g.iloc[-5]["eps"] if len(fin_g) >= 5 else np.nan
            eps_yoy    = (eps_now / eps_4ago - 1) if (eps_4ago and eps_4ago != 0) else np.nan
            eps_prev   = fin_g.iloc[-2]["eps"] if len(fin_g) >= 2 else np.nan
            eps_8ago   = fin_g.iloc[-6]["eps"] if len(fin_g) >= 6 else np.nan
            eps_pv_yoy = (eps_prev / eps_8ago - 1) if (eps_8ago and eps_8ago != 0) else np.nan
            eps_accel  = (eps_yoy - eps_pv_yoy) if not (np.isnan(eps_yoy) or np.isnan(eps_pv_yoy)) else np.nan
            rev_now    = last_fin.get("revenue", np.nan)
            rev_4ago   = fin_g.iloc[-5]["revenue"] if len(fin_g) >= 5 else np.nan
            rev_yoy    = (rev_now / rev_4ago - 1) if (rev_4ago and rev_4ago != 0) else np.nan
            gm_arr     = fin_g["gross_margin"].dropna().tail(4).values
            gm_trend   = gm_arr[-1] - gm_arr[-3] if len(gm_arr) >= 3 else np.nan
            roe_now    = last_fin.get("roe", np.nan)
            de_ratio   = last_fin.get("de_ratio", np.nan)
            cfo_vals   = fin_g["cfo"].tail(2).values
            # CFO dương ≥ 2 quý liên tiếp (Sloan 1996 — accrual anomaly)
            cfo_pos    = 1 if (len(cfo_vals) >= 2 and
                               all(not np.isnan(v) and v > 0 for v in cfo_vals[-2:])) else 0
            ni_vals    = fin_g["net_income"].tail(4).values
            ni_pos4q   = int(np.sum(ni_vals > 0)) if len(ni_vals) > 0 else 0

        sea     = compute_seasonal(prices, ticker, ref_date=ref_date)
        sea_row = sea[sea["month"] == target_month]
        seasonal_avg  = sea_row["avg_return"].values[0] if len(sea_row) > 0 else np.nan
        seasonal_wr   = sea_row["win_rate"].values[0]   if len(sea_row) > 0 else np.nan
        overall_avg   = sea["avg_return"].mean()
        seasonal_edge = seasonal_avg - overall_avg if not np.isnan(seasonal_avg) else np.nan

        sr  = compute_support_resistance(grp, float(c[-1]))
        pat = compute_price_patterns(c, v, grp["vol_ma20"].values)

        # [NEW] ADX signal — xu hướng mạnh khi ADX > 25
        adx_val  = float(last.get("adx", np.nan))
        adx_bull = int(not np.isnan(adx_val) and adx_val > 25 and
                       float(last.get("plus_di", 0)) > float(last.get("minus_di", 0)))

        # [NEW] ATR % — dùng cho risk-adjusted signal
        atr_pct = float(last.get("atr_pct", np.nan))

        rows.append({
            "ticker":       ticker,
            "price":        float(c[-1]),
            "r1m": r1m, "r3m": r3m, "r6m": r6m, "r6m_adj": r6m_adj,
            "vol_ratio": vol_ratio, "ma_score": ma_score,
            "ma20": float(last.get("ma20", np.nan)),
            "ma50": float(last.get("ma50", np.nan)),
            "rsi": rsi, "rsi_score": rsi_score,
            "macd_bull": macd_bull, "macd_hist": macd_hist,
            "bb_pct": bb_pct, "stoch_k": stoch_k,
            "adx": adx_val, "adx_bull": adx_bull,
            "atr_pct": atr_pct,
            "turnover_ma20": turnover_ma20,  # ← thêm dòng này
            "rs_vs_mkt": rs_vs_mkt, "rs_trend": rs_trend,
            "eps_yoy": eps_yoy, "eps_accel": eps_accel,
            "rev_yoy": rev_yoy, "gm_trend": gm_trend,
            "roe_now": roe_now, "de_ratio": de_ratio,
            "cfo_pos": cfo_pos, "ni_pos4q": ni_pos4q,
            "seasonal_avg": seasonal_avg, "seasonal_wr": seasonal_wr,
            "seasonal_edge": seasonal_edge,
            "nearest_support_pct":  sr["nearest_support_pct"],
            "nearest_resist_pct":   sr["nearest_resist_pct"],
            "at_support":           sr["at_support"],
            "at_resistance":        sr["at_resistance"],
            "support_strength":     sr["support_strength"],
            "resistance_strength":  sr["resistance_strength"],
            "streak_up":            pat["streak_up"],
            "streak_down":          pat["streak_down"],
            "streak_magnitude":     pat["streak_magnitude"],
            "vol_trend_in_streak":  pat["vol_trend_in_streak"],
            "exhaustion_signal":    pat["exhaustion_signal"],
            "vol_breakout":         pat["vol_breakout"],
            "inside_bars":          pat["inside_bars"],
            "market_regime":        regime,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.merge(
        info[["ticker_clean", "company_name", "sector", "sector_vn", "industry", "exchange"]],
        left_on="ticker", right_on="ticker_clean", how="left"
    )
    df["sector"] = df["sector"].fillna("Unknown")

    # ── Sub-scores (sector-relative percentile) ───────────────────────────────
    def sector_pct(col, ascending=True):
        result = pd.Series(np.nan, index=df.index)
        for sec, g in df.groupby("sector"):
            r = g[col].rank(pct=True, na_option="keep", ascending=ascending) * 100
            result.loc[g.index] = r
        return result

    df["s_r6m_adj"] = sector_pct("r6m_adj")
    df["s_vol"]     = sector_pct("vol_ratio")
    df["s_ma"]      = df["ma_score"] / 5 * 100
    df["s_rsi"]     = df["rsi_score"]
    df["s_macd"]    = df["macd_bull"] * 100
    df["s_bb"]      = df["bb_pct"].clip(0, 1) * 100
    df["s_adx"]     = df["adx_bull"] * 100   # ADX Wilder

    # Vol ratio penalty — trừ điểm khi vol_ratio > 4× (tránh phiên phân phối đỉnh)
    vol_penalty = np.where(df["vol_ratio"] > 4.0, (df["vol_ratio"] - 4.0) * 10, 0)
    vol_low_penalty = np.where(df["vol_ratio"] < 0.8, (0.8 - df["vol_ratio"]) * 50, 0)
    df["s_vol_adj"] = (df["s_vol"] - vol_penalty - vol_low_penalty).clip(0, 100)
    # MA score mandatory condition: Price > MA50 AND MA50 > MA200
    # Nếu không thỏa → ma_score bị cap ở 40 (phản ánh xu hướng yếu)
    ma50_ok  = (~df["ma50"].isna())  & (df["price"] > df["ma50"])
    ma200_ok = (~df["ma50"].isna())  # proxy: nếu không có ma200, chỉ cần ma50
    # Lấy MA200 từ last row — cần join lại; dùng ma_score trực tiếp với penalty
    df["s_ma_adj"] = np.where(
        df["ma_score"] >= 3,          # Price > MA50 implied khi ma_score >= 3 (vì MA50 là 1 trong 4 thành phần)
        df["s_ma"],
        df["s_ma"] * 0.5              # Giảm 50% nếu cấu trúc MA yếu
    )

    # score_momentum — trọng số nghiên cứu: r6m_adj 25%, vol 30%, ma 20%, rsi 10%, macd 5%, bb 5%, adx 5%
    df["score_momentum"] = (
        df["s_r6m_adj"]  * 0.25 +
        df["s_vol_adj"]  * 0.30 +
        df["s_ma_adj"]   * 0.20 +
        df["s_rsi"]      * 0.10 +
        df["s_macd"]     * 0.05 +
        df["s_bb"]       * 0.05 +
        df["s_adx"]      * 0.05
    ).clip(0, 100)

    df["s_rs_mkt"]   = sector_pct("rs_vs_mkt")
    df["s_rs_trend"] = sector_pct("rs_trend")
    df["score_rs"]   = (df["s_rs_mkt"] * 0.65 + df["s_rs_trend"] * 0.35).clip(0, 100)

    df["s_eps_yoy"]   = sector_pct("eps_yoy")
    df["s_eps_accel"] = sector_pct("eps_accel")
    df["s_rev_yoy"]   = sector_pct("rev_yoy")
    df["s_gm_trend"]  = sector_pct("gm_trend")
    df["s_roe"]       = sector_pct("roe_now")
    df["s_de"]        = sector_pct("de_ratio", ascending=False)
    df["s_cfo"]       = df["cfo_pos"] * 100
    df["s_ni4q"]      = df["ni_pos4q"] / 4 * 100

    # score_fundamental — eps_yoy 20%, rev_yoy 15%, eps_accel 15%, cfo_pos 15%,
    #                     gm_trend 10%, roe_now 10%, ni_pos4q 5%, de_ratio 5%
    df["score_fundamental"] = (
        df["s_eps_yoy"]   * 0.20 +
        df["s_rev_yoy"]   * 0.15 +
        df["s_eps_accel"] * 0.15 +
        df["s_cfo"]       * 0.15 +
        df["s_gm_trend"]  * 0.10 +
        df["s_roe"]       * 0.10 +
        df["s_ni4q"]      * 0.05 +
        df["s_de"]        * 0.05
    ).clip(0, 100)

    df["s_sea_avg"]  = sector_pct("seasonal_avg")
    df["s_sea_wr"]   = df["seasonal_wr"].fillna(0.5) * 100
    df["s_sea_edge"] = sector_pct("seasonal_edge")

    df["score_seasonal"] = (
        df["s_sea_avg"] * 0.40 + df["s_sea_wr"] * 0.35 + df["s_sea_edge"] * 0.25
    ).clip(0, 100)

    # ── Adjustment terms ──────────────────────────────────────────────────────
    df["sr_adjust"] = (
        df["at_support"]    * df["support_strength"]    * 10 +
        df["at_resistance"] * df["resistance_strength"] * (-15)
    )

    # [FIX] Clip score_momentum TRƯỚC khi áp dụng exhaustion penalty
    # Cap score_momentum nếu vol_ratio quá thấp
    df["score_momentum"] = np.where(
        df["vol_ratio"] < 0.5,
        df["score_momentum"].clip(0, 50),
        df["score_momentum"]
    )
    df["vol_confirm_bonus"] = df["vol_breakout"].fillna(0) * 5
    # [FIX] confluence_bonus được clip độc lập, không cộng dồn không kiểm soát
    trend_strength         = ((df["rs_vs_mkt"] > 0) & (df["vol_ratio"] > 1.2)).astype(int)
    df["confluence_bonus"] = (trend_strength * 15).clip(0, 15)

    # ── Final Score (dùng calibrated weights W) ───────────────────────────────
    regime_adj = regime_info["min_score_adjust"]
    tier_thresholds = {
        "Mua mạnh": 70 + regime_adj,
        "Mua":       55 + regime_adj,
        "Trung lập": 35 + regime_adj,
    }

    df["final_score"] = (
        df["score_momentum"]    * W["score_momentum"]    +
        df["score_rs"]          * W["score_rs"]          +
        df["score_fundamental"] * W["score_fundamental"] +
        df["score_seasonal"]    * W["score_seasonal"]    +
        df["sr_adjust"]         +
        df["vol_confirm_bonus"] +
        df["confluence_bonus"]
    ).clip(0, 100)

    def assign_tier(score):
        if score >= tier_thresholds["Mua mạnh"]: return "Mua mạnh"
        if score >= tier_thresholds["Mua"]:      return "Mua"
        if score >= tier_thresholds["Trung lập"]: return "Trung lập"
        return "Tránh"

    df["tier"]          = df["final_score"].apply(assign_tier)
    df["market_regime"] = regime
    # Loại mã thanh khoản thấp < 10 tỷ VNĐ/ngày
    LIQUIDITY_MIN = 10_000_000_000
    df = df[df["turnover_ma20"] >= LIQUIDITY_MIN].reset_index(drop=True)
    df = compute_sector_rotation(df, lookback_col="rs_vs_mkt")

    def _signals(row):
        sigs = []
        if row.get("vol_ratio", 0)        > 1.5:  sigs.append("Volume ↑")
        if row.get("ma_score", 0)         >= 4:   sigs.append("MA Align")
        if row.get("rsi", 50)             < 40:   sigs.append("RSI Oversold")
        if row.get("macd_bull", 0)        == 1:   sigs.append("MACD Bull")
        if row.get("adx_bull", 0)         == 1:   sigs.append(f"ADX {row.get('adx', 0):.0f}")
        if row.get("eps_yoy", 0)          > 0.10: sigs.append("EPS +YoY")
        if row.get("eps_accel", 0)        > 0:    sigs.append("EPS Accel")
        if row.get("rs_vs_mkt", 0)        > 0:    sigs.append("RS>Index")
        if row.get("seasonal_wr", 0)      > 0.6:  sigs.append(f"Mùa vụ {int(row.get('seasonal_wr', 0)*100)}%")
        if row.get("at_support", 0)       == 1:   sigs.append(f"Hỗ trợ {row.get('support_strength', 0):.0%}")
        if row.get("exhaustion_signal", 0) == 1:  sigs.append("⚠ Exhaustion")
        if row.get("vol_breakout", 0)     == 1:   sigs.append("Vol Spike")
        if row.get("in_leading_sector", 0) == 1:  sigs.append("🔥 Ngành dẫn")
        if row.get("at_resistance", 0)    == 1:   sigs.append("⚠ Kháng cự")
        # [NEW] ATR warning cho mã quá volatile
        atr = row.get("atr_pct", np.nan)
        if not np.isnan(atr) and atr > 4:         sigs.append(f"⚠ ATR {atr:.1f}%")
        return " · ".join(sigs) if sigs else "—"

    df["signals"] = df.apply(_signals, axis=1)

    return df.sort_values("final_score", ascending=False).reset_index(drop=True)