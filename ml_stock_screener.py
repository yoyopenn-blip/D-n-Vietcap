"""
ml_stock_screener.py
════════════════════
Định nghĩa 4 class chính cho hệ thống lọc cổ phiếu ML VN:
  - FeatureEngineer   : Tính 36+ chỉ báo kỹ thuật & cơ bản
  - RuleBasedScorer   : Chấm điểm rule-based (Momentum, RS, Fundamental)
  - MLPredictor       : Huấn luyện XGBoost walk-forward
  - PortfolioOptimizer: Lọc danh mục tối ưu

Schema dữ liệu đầu vào (từ convert_to_parquet.py):
  df_price  : [date, ticker_clean, open, high, low, close, volume]
  df_index  : [date, vni]
  df_fund   : [ticker_clean, date, revenue, net_income, eps, equity,
               total_assets, total_liabilities, cfo, capex, ...]
  df_info   : [ticker_clean, sector, ...]
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  xgboost không được cài. Chạy: pip install xgboost")

try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_div(a, b, fill=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result


def _rolling_safe(series: pd.Series, window: int, func: str, min_p: int = None):
    mp = min_p or max(1, window // 2)
    return getattr(series.rolling(window, min_periods=mp), func)()


# ══════════════════════════════════════════════════════════════════════════════
#  1. FEATURE ENGINEER
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Tính toán các chỉ báo kỹ thuật & cơ bản từ dữ liệu thô.

    Parameters
    ----------
    df_price : DataFrame  [date, ticker_clean, open, high, low, close, volume]
    df_index : DataFrame  [date, vni]
    df_fund  : DataFrame  [ticker_clean, date, revenue, net_income, eps, ...]
    df_info  : DataFrame  [ticker_clean, sector, ...]
    """

    TECH_FEATURES = [
        "ret_1d", "ret_5d", "ret_20d", "ret_60d",
        "ma20", "ma50", "ma200",
        "price_vs_ma20", "price_vs_ma50", "price_vs_ma200",
        "rsi_14", "rsi_28",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_pct",
        "atr_14", "atr_pct",
        "vol_ratio_20", "vol_ratio_60",
        "high_52w", "low_52w", "dist_from_52w_high",
        "rs_20d", "rs_60d",
        "obv_slope",
        "close_std_20", "close_std_60",
        # Volume quality (VN-specific)
        "vol_consistency",   # tính ổn định volume 20 phiên (CV thấp = ổn định)
        "vol_trend_slope",   # xu hướng volume 10 phiên (tăng dần = tích cực)
        "price_vol_confirm", # giá tăng kèm volume tăng (1=xác nhận, 0=phân kỳ)
        "distribution_day",  # phiên phân phối: vol cao nhưng giá không tăng
        "vol_spike_isolated",# vol đột biến đơn lẻ (1=bất thường, 0=bình thường)
    ]

    FUND_FEATURES = [
        "roe", "roa", "gross_margin", "net_margin",
        "debt_to_equity", "current_ratio",
        "revenue_growth", "eps_growth",
        "cfo_to_assets",
    ]

    def __init__(self, df_price, df_index, df_fund=None, df_info=None):
        self.df_price = df_price.copy()
        self.df_index = df_index.copy() if df_index is not None and not df_index.empty else pd.DataFrame()
        self.df_fund  = df_fund.copy()  if df_fund  is not None and not df_fund.empty  else pd.DataFrame()
        self.df_info  = df_info.copy()  if df_info  is not None and not df_info.empty  else pd.DataFrame()
        self.has_fundamentals = not self.df_fund.empty

        # Chuẩn hoá tên cột
        self._normalize_price_cols()

    def _normalize_price_cols(self):
        col_map = {}
        for c in self.df_price.columns:
            cl = c.lower().strip()
            if cl in ("date", "tradingdate"):
                col_map[c] = "Date"
            elif cl in ("ticker_clean", "ticker", "symbol", "code"):
                col_map[c] = "Ticker"
            elif cl in ("close", "price close", "adjclose"):
                col_map[c] = "Close"
            elif cl in ("open", "price open"):
                col_map[c] = "Open"
            elif cl in ("high", "price high"):
                col_map[c] = "High"
            elif cl in ("low", "price low"):
                col_map[c] = "Low"
            elif cl in ("volume", "vol"):
                col_map[c] = "Volume"
        self.df_price = self.df_price.rename(columns=col_map)

        # Đảm bảo cột Date là datetime
        self.df_price["Date"] = pd.to_datetime(self.df_price["Date"], errors="coerce")
        self.df_price = self.df_price.dropna(subset=["Date", "Ticker", "Close"])
        self.df_price = self.df_price.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # ── Chỉ báo kỹ thuật ──────────────────────────────────────────────────────

    def _calc_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_g = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_l = loss.ewm(com=window - 1, min_periods=window).mean()
        rs = _safe_div(avg_g.values, avg_l.values, fill=100.0)
        return pd.Series(100 - (100 / (1 + rs)), index=series.index)

    def _calc_macd(self, series: pd.Series):
        ema12 = series.ewm(span=12, min_periods=12).mean()
        ema26 = series.ewm(span=26, min_periods=26).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, min_periods=9).mean()
        return macd, signal, macd - signal

    def _calc_bb(self, series: pd.Series, window: int = 20):
        ma  = _rolling_safe(series, window, "mean")
        std = _rolling_safe(series, window, "std")
        upper = ma + 2 * std
        lower = ma - 2 * std
        pct   = _safe_div((series.values - lower.values), (upper.values - lower.values).clip(1e-8))
        return upper, lower, pd.Series(pct, index=series.index)

    def _calc_atr(self, high, low, close, window: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=window, min_periods=window).mean()

    def _calc_obv_slope(self, close, volume, window: int = 20) -> pd.Series:
        direction = np.sign(close.diff().fillna(0))
        obv = (direction * volume).cumsum()
        return obv.diff(window) / window

    def _calc_vol_quality(self, close: pd.Series, volume: pd.Series) -> dict:
        """
        Tính các chỉ báo chất lượng volume đặc thù thị trường VN.

        vol_consistency  : CV (std/mean) của volume 20 phiên — thấp = ổn định
                           VN hay bị spike ảo 1-2 phiên → CV cao là cờ đỏ
        vol_trend_slope  : Hồi quy tuyến tính volume 10 phiên (chuẩn hoá)
                           Dương = dòng tiền đang tích lũy dần
        price_vol_confirm: Tương quan giá-volume 10 phiên
                           Dương = breakout có xác nhận, Âm = phân kỳ nguy hiểm
        distribution_day : Vol > 1.2× TB20 + giá không tăng (close <= open)
                           Tích lũy 3/5 phiên gần nhất → đang bị xả hàng
        vol_spike_isolated: Vol > 2× TB20 nhưng phiên kề không cao
                            Spike đơn lẻ thường là làm giá / tạo sóng ảo
        """
        v   = volume.fillna(0)
        c   = close

        # vol_consistency: CV của 20 phiên (thấp = tốt, clip để dễ dùng)
        vol_ma20  = _rolling_safe(v, 20, "mean").replace(0, np.nan)
        vol_std20 = _rolling_safe(v, 20, "std")
        cv = _safe_div(vol_std20.values, vol_ma20.values, fill=1.0)
        # Đảo chiều: consistency cao = CV thấp → dùng 1 - clip(CV, 0, 2)/2
        vol_consistency = 1.0 - pd.Series(cv, index=v.index).clip(0, 2) / 2

        # vol_trend_slope: slope tuyến tính 10 phiên (chuẩn hoá theo MA)
        vol_slope = pd.Series(np.nan, index=v.index)
        for i in range(9, len(v)):
            window_v = v.iloc[i-9:i+1].values
            if window_v.mean() > 0:
                x = np.arange(10)
                slope = np.polyfit(x, window_v, 1)[0]
                vol_slope.iloc[i] = slope / (window_v.mean() + 1e-9)
        vol_trend_slope = vol_slope.clip(-2, 2)

        # price_vol_confirm: rolling correlation(ret_1d, vol_ratio) 10 phiên
        ret_1d    = c.pct_change()
        vol_ratio = _safe_div(v.values, vol_ma20.values, fill=1.0)
        vol_ratio_s = pd.Series(vol_ratio, index=v.index)
        price_vol_confirm = ret_1d.rolling(10, min_periods=5).corr(vol_ratio_s).fillna(0)

        # distribution_day: vol > 1.2× TB20 AND close <= open (nến đỏ/doji)
        open_col = pd.Series(np.nan, index=v.index)  # placeholder nếu không có open
        vol_high = (v > vol_ma20 * 1.2).astype(int)
        # Dùng ret_1d <= 0 làm proxy cho close <= open
        price_weak = (ret_1d <= 0).astype(int)
        dist_signal = (vol_high * price_weak)
        # Tích luỹ: tỷ lệ phiên phân phối trong 5 phiên gần nhất
        distribution_day = dist_signal.rolling(5, min_periods=3).mean().fillna(0)

        # vol_spike_isolated: vol > 2× TB20, nhưng TB(vol_-2:-1, vol_+1:+2) < 1.5× TB20
        spike    = (v > vol_ma20 * 2).astype(float)
        neighbor = (
            v.shift(1).fillna(0) + v.shift(2).fillna(0) +
            v.shift(-1).fillna(0) + v.shift(-2).fillna(0)
        ) / 4
        neighbor_ratio = _safe_div(neighbor.values, vol_ma20.values, fill=1.0)
        isolated = spike * (pd.Series(neighbor_ratio, index=v.index) < 1.5).astype(float)
        vol_spike_isolated = isolated.rolling(3, min_periods=1).max().fillna(0)

        return {
            "vol_consistency":    vol_consistency,
            "vol_trend_slope":    vol_trend_slope,
            "price_vol_confirm":  price_vol_confirm,
            "distribution_day":   distribution_day,
            "vol_spike_isolated": vol_spike_isolated,
        }

    def _tech_for_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        h = df.get("High", c)
        l = df.get("Low",  c)
        v = df.get("Volume", pd.Series(np.nan, index=df.index))

        # Returns
        df["ret_1d"]  = c.pct_change(1)
        df["ret_5d"]  = c.pct_change(5)
        df["ret_20d"] = c.pct_change(20)
        df["ret_60d"] = c.pct_change(60)

        # Moving averages
        df["ma20"]  = _rolling_safe(c, 20,  "mean")
        df["ma50"]  = _rolling_safe(c, 50,  "mean")
        df["ma200"] = _rolling_safe(c, 200, "mean")
        df["price_vs_ma20"]  = _safe_div(c.values, df["ma20"].values)  - 1
        df["price_vs_ma50"]  = _safe_div(c.values, df["ma50"].values)  - 1
        df["price_vs_ma200"] = _safe_div(c.values, df["ma200"].values) - 1

        # RSI
        df["rsi_14"] = self._calc_rsi(c, 14)
        df["rsi_28"] = self._calc_rsi(c, 28)

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = self._calc_macd(c)

        # Bollinger Bands
        df["bb_upper"], df["bb_lower"], df["bb_pct"] = self._calc_bb(c)

        # ATR
        atr = self._calc_atr(h, l, c)
        df["atr_14"]  = atr
        df["atr_pct"] = _safe_div(atr.values, c.values)

        # Volume ratio
        vol_ma20 = _rolling_safe(v, 20, "mean")
        vol_ma60 = _rolling_safe(v, 60, "mean")
        df["vol_ratio_20"] = _safe_div(v.values, vol_ma20.values)
        df["vol_ratio_60"] = _safe_div(v.values, vol_ma60.values)

        # 52-week high/low
        df["high_52w"] = _rolling_safe(c, 252, "max")
        df["low_52w"]  = _rolling_safe(c, 252, "min")
        df["dist_from_52w_high"] = _safe_div(c.values, df["high_52w"].values) - 1

        # Volatility
        df["close_std_20"] = _rolling_safe(c, 20, "std")
        df["close_std_60"] = _rolling_safe(c, 60, "std")

        # OBV slope
        df["obv_slope"] = self._calc_obv_slope(c, v)

        # Volume quality (VN-specific)
        vq = self._calc_vol_quality(c, v)
        df["vol_consistency"]    = vq["vol_consistency"]
        df["vol_trend_slope"]    = vq["vol_trend_slope"]
        df["price_vol_confirm"]  = vq["price_vol_confirm"]
        df["distribution_day"]   = vq["distribution_day"]
        df["vol_spike_isolated"] = vq["vol_spike_isolated"]

        return df

    def _add_relative_strength(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """RS so với VN-Index"""
        if self.df_index.empty:
            df_all["rs_20d"] = np.nan
            df_all["rs_60d"] = np.nan
            return df_all

        # Chuẩn hoá index
        idx = self.df_index.copy()
        date_col = [c for c in idx.columns if "date" in c.lower()][0]
        vni_col  = [c for c in idx.columns if c.lower() in ("vni", "close", "index")][0]
        idx = idx.rename(columns={date_col: "Date", vni_col: "vni"})
        idx["Date"] = pd.to_datetime(idx["Date"])
        idx = idx.sort_values("Date").drop_duplicates("Date")
        idx["vni_ret_20d"] = idx["vni"].pct_change(20)
        idx["vni_ret_60d"] = idx["vni"].pct_change(60)

        df_all = df_all.merge(idx[["Date", "vni_ret_20d", "vni_ret_60d"]], on="Date", how="left")
        df_all["rs_20d"] = df_all["ret_20d"] - df_all["vni_ret_20d"]
        df_all["rs_60d"] = df_all["ret_60d"] - df_all["vni_ret_60d"]
        df_all = df_all.drop(columns=["vni_ret_20d", "vni_ret_60d"], errors="ignore")
        return df_all

    def _add_fundamentals(self, df_all: pd.DataFrame) -> pd.DataFrame:
        if self.df_fund.empty:
            for f in self.FUND_FEATURES:
                df_all[f] = 0.0
            return df_all

        fund = self.df_fund.copy()
        # Chuẩn hoá tên cột
        fund.columns = [c.lower().strip() for c in fund.columns]
        for col_var in ("ticker_clean", "ticker"):
            if col_var in fund.columns:
                fund = fund.rename(columns={col_var: "Ticker"})
                break
        for col_var in ("date",):
            if col_var in fund.columns:
                fund["Date"] = pd.to_datetime(fund[col_var])

        fund = fund.sort_values(["Ticker", "Date"])

        # Tính các chỉ số
        if "net_income" in fund.columns and "equity" in fund.columns:
            fund["roe"] = _safe_div(fund["net_income"].values, fund["equity"].values)
        if "net_income" in fund.columns and "total_assets" in fund.columns:
            fund["roa"] = _safe_div(fund["net_income"].values, fund["total_assets"].values)
        if "revenue" in fund.columns and "net_income" in fund.columns:
            fund["net_margin"] = _safe_div(fund["net_income"].values, fund["revenue"].values)
        if "total_liabilities" in fund.columns and "equity" in fund.columns:
            fund["debt_to_equity"] = _safe_div(fund["total_liabilities"].values, fund["equity"].fillna(1).values)
        if "cfo" in fund.columns and "total_assets" in fund.columns:
            fund["cfo_to_assets"] = _safe_div(fund["cfo"].values, fund["total_assets"].values)
        if "revenue" in fund.columns:
            fund["revenue_growth"] = fund.groupby("Ticker")["revenue"].pct_change(4)
        if "eps" in fund.columns:
            fund["eps_growth"] = fund.groupby("Ticker")["eps"].pct_change(4)

        for f in self.FUND_FEATURES:
            if f not in fund.columns:
                fund[f] = 0.0

        # Merge as-of (dùng dữ liệu fundamental gần nhất với mỗi ngày giao dịch)
        df_all = df_all.sort_values(["Ticker", "Date"])
        fund_subset = fund[["Ticker", "Date"] + self.FUND_FEATURES].dropna(subset=["Ticker", "Date"])

        merged_parts = []
        for ticker, grp in df_all.groupby("Ticker"):
            f_grp = fund_subset[fund_subset["Ticker"] == ticker].sort_values("Date")
            if f_grp.empty:
                for f in self.FUND_FEATURES:
                    grp[f] = 0.0
            else:
                grp = pd.merge_asof(grp, f_grp, on="Date", by="Ticker",
                                    direction="backward", suffixes=("", "_fund"))
                # Lấp các cột chưa có
                for f in self.FUND_FEATURES:
                    if f not in grp.columns:
                        grp[f] = 0.0
            merged_parts.append(grp)

        df_all = pd.concat(merged_parts, ignore_index=True)
        return df_all

    def _add_sector(self, df_all: pd.DataFrame) -> pd.DataFrame:
        if self.df_info.empty:
            df_all["Sector"] = "Unknown"
            return df_all
        info = self.df_info.copy()
        info.columns = [c.lower().strip() for c in info.columns]
        for c in info.columns:
            if "ticker" in c:
                info = info.rename(columns={c: "Ticker"})
                break
        sector_col = next((c for c in info.columns if "sector" in c.lower() or "nganh" in c.lower()), None)
        if sector_col:
            info = info.rename(columns={sector_col: "Sector"})
            df_all = df_all.merge(info[["Ticker", "Sector"]].drop_duplicates("Ticker"),
                                  on="Ticker", how="left")
        if "Sector" not in df_all.columns:
            df_all["Sector"] = "Unknown"
        df_all["Sector"] = df_all["Sector"].fillna("Unknown")
        return df_all

    def _add_market_regime(self, df_all: pd.DataFrame) -> pd.DataFrame:
        if self.df_index.empty:
            df_all["market_regime"] = "neutral"
            return df_all
        idx = self.df_index.copy()
        date_col = [c for c in idx.columns if "date" in c.lower()][0]
        vni_col  = [c for c in idx.columns if c.lower() in ("vni", "close", "index")][0]
        idx = idx.rename(columns={date_col: "Date", vni_col: "vni"})
        idx["Date"] = pd.to_datetime(idx["Date"])
        idx = idx.sort_values("Date").drop_duplicates("Date")
        idx["vni_ma50"]  = idx["vni"].rolling(50,  min_periods=20).mean()
        idx["vni_ma200"] = idx["vni"].rolling(200, min_periods=50).mean()
        idx["market_regime"] = np.where(
            idx["vni"] > idx["vni_ma50"],
            np.where(idx["vni_ma50"] > idx["vni_ma200"], "bull", "neutral"),
            "bear"
        )
        df_all = df_all.merge(idx[["Date", "market_regime"]], on="Date", how="left")
        df_all["market_regime"] = df_all["market_regime"].fillna("neutral")
        return df_all

    def run_pipeline(self) -> pd.DataFrame:
        print("  → Tính chỉ báo kỹ thuật...")
        parts = []
        for ticker, grp in self.df_price.groupby("Ticker"):
            grp = grp.sort_values("Date").copy()
            if len(grp) < 30:
                continue
            grp = self._tech_for_ticker(grp)
            parts.append(grp)

        if not parts:
            print("  ❌ Không có dữ liệu sau khi tính chỉ báo.")
            return pd.DataFrame()

        df_all = pd.concat(parts, ignore_index=True)

        print("  → Tính Relative Strength vs VN-Index...")
        df_all = self._add_relative_strength(df_all)

        print("  → Gắn Sector...")
        df_all = self._add_sector(df_all)

        print("  → Tính chỉ số cơ bản...")
        df_all = self._add_fundamentals(df_all)

        print("  → Xác định Market Regime...")
        df_all = self._add_market_regime(df_all)

        print(f"  ✅ Feature engineering xong: {len(df_all):,} dòng | {df_all['Ticker'].nunique()} mã")
        return df_all


# ══════════════════════════════════════════════════════════════════════════════
#  2. RULE-BASED SCORER
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedScorer:
    """
    Chấm điểm rule-based theo 3 nhóm:
      - Score_Momentum  (0-40)
      - Score_RS        (0-30)
      - Score_Fundamental (0-30, nếu có dữ liệu)
    Tổng: Total_Score (0-100)
    """

    def __init__(self, df_features: pd.DataFrame, has_fundamentals: bool = True):
        self.df = df_features.copy()
        self.has_fundamentals = has_fundamentals

    def _score_momentum(self) -> pd.Series:
        df = self.df
        score = pd.Series(0.0, index=df.index)

        # Trend (giá > MA): tối đa 15 điểm
        score += np.where(df.get("price_vs_ma20",  pd.Series(0.0, index=df.index)) > 0, 5,  0)
        score += np.where(df.get("price_vs_ma50",  pd.Series(0.0, index=df.index)) > 0, 5,  0)
        score += np.where(df.get("price_vs_ma200", pd.Series(0.0, index=df.index)) > 0, 5,  0)

        # RSI hợp lý (40-75): tối đa 5 điểm
        rsi = df.get("rsi_14", pd.Series(50.0, index=df.index))
        score += np.where((rsi >= 40) & (rsi <= 75), 5, 0)

        # MACD bullish: tối đa 5 điểm
        macd_hist = df.get("macd_hist", pd.Series(0.0, index=df.index))
        score += np.where(macd_hist > 0, 5, 0)

        # Breakout từ 52w high (gần đỉnh): tối đa 5 điểm
        dist = df.get("dist_from_52w_high", pd.Series(-0.5, index=df.index))
        score += np.where(dist > -0.05, 5, 0)

        # Volume surge: tối đa 5 điểm
        vol_r = df.get("vol_ratio_20", pd.Series(1.0, index=df.index))
        score += np.where(vol_r > 1.5, 5, 0)

        # Volume quality bonus/penalty (VN-specific): net tối đa +5, tối thiểu -10
        vol_cons  = df.get("vol_consistency",    pd.Series(0.5, index=df.index))
        vol_trend = df.get("vol_trend_slope",    pd.Series(0.0, index=df.index))
        pv_conf   = df.get("price_vol_confirm",  pd.Series(0.0, index=df.index))
        dist_day  = df.get("distribution_day",   pd.Series(0.0, index=df.index))
        spike_iso = df.get("vol_spike_isolated", pd.Series(0.0, index=df.index))

        # Bonus: volume ổn định + xu hướng tăng + giá-vol xác nhận nhau
        vol_quality_bonus = (
            np.where(vol_cons  > 0.65, 3, 0) +   # volume đều đặn
            np.where(vol_trend > 0.10, 2, 0) +   # dòng tiền tích lũy dần
            np.where(pv_conf   > 0.30, 2, 0)     # giá tăng cùng chiều volume
        )
        # Penalty: phiên phân phối + spike đơn lẻ (dấu hiệu làm giá)
        vol_quality_penalty = (
            np.where(dist_day  > 0.4, -8, np.where(dist_day  > 0.2, -4, 0)) +
            np.where(spike_iso > 0,   -5, 0)
        )
        score = score + vol_quality_bonus + vol_quality_penalty

        # Return momentum: tối đa 5 điểm
        ret20 = df.get("ret_20d", pd.Series(0.0, index=df.index))
        score += np.where(ret20 > 0.05, 5, 0)

        return score.clip(0, 40)

    def _score_rs(self) -> pd.Series:
        df = self.df
        score = pd.Series(0.0, index=df.index)

        rs20 = df.get("rs_20d", pd.Series(0.0, index=df.index))
        rs60 = df.get("rs_60d", pd.Series(0.0, index=df.index))

        # RS20 > 0 (tốt hơn index gần đây): 10 điểm
        score += np.where(rs20 > 0, 10, 0)
        # RS20 > 5%: thêm 5 điểm
        score += np.where(rs20 > 0.05, 5, 0)
        # RS60 > 0: 10 điểm
        score += np.where(rs60 > 0, 10, 0)
        # RS60 > 10%: thêm 5 điểm
        score += np.where(rs60 > 0.10, 5, 0)

        return score.clip(0, 30)

    def _score_fundamental(self) -> pd.Series:
        df = self.df
        score = pd.Series(0.0, index=df.index)

        if not self.has_fundamentals:
            return score

        roe = df.get("roe", pd.Series(0.0, index=df.index))
        roa = df.get("roa", pd.Series(0.0, index=df.index))
        net_margin = df.get("net_margin", pd.Series(0.0, index=df.index))
        d2e = df.get("debt_to_equity", pd.Series(1.0, index=df.index))
        rev_g = df.get("revenue_growth", pd.Series(0.0, index=df.index))
        eps_g = df.get("eps_growth",     pd.Series(0.0, index=df.index))
        cfo   = df.get("cfo_to_assets",  pd.Series(0.0, index=df.index))

        score += np.where(roe > 0.15, 8, np.where(roe > 0.08, 4, 0))
        score += np.where(roa > 0.05, 5, 0)
        score += np.where(net_margin > 0.10, 5, 0)
        score += np.where(d2e < 1.0, 5, np.where(d2e < 2.0, 2, 0))
        score += np.where(rev_g > 0.10, 4, np.where(rev_g > 0, 2, 0))
        score += np.where(eps_g > 0.15, 3, 0)
        score += np.where(cfo > 0, 0, 0)  # placeholder

        return score.clip(0, 30)

    def _vol_quality_flag(self) -> pd.Series:
        """
        Hard filter volume — trả về:
          0  = volume ổn, không bị phạt thêm
         -15 = volume đáng ngờ (1 tiêu chí xấu)
         -30 = volume rất xấu (2+ tiêu chí xấu) → gần như chắc bị loại khỏi Mua/Mua Mạnh

        Ba tiêu chí độc lập:
          A. vol_consistency < 0.40  → volume quá bất thường (spike ảo thường xuyên)
          B. distribution_day > 0.40 → hơn 40% phiên gần đây là phân phối
          C. vol_spike_isolated > 0  → có spike đơn lẻ trong 3 phiên gần nhất
        """
        df = self.df
        flag = pd.Series(0, index=df.index)

        vol_cons  = df.get("vol_consistency",    pd.Series(0.7, index=df.index))
        dist_day  = df.get("distribution_day",   pd.Series(0.0, index=df.index))
        spike_iso = df.get("vol_spike_isolated", pd.Series(0.0, index=df.index))

        bad_consistency  = (vol_cons  < 0.40).astype(int)
        bad_distribution = (dist_day  > 0.40).astype(int)
        bad_spike        = (spike_iso > 0   ).astype(int)

        n_bad = bad_consistency + bad_distribution + bad_spike
        flag = np.where(n_bad >= 2, -30, np.where(n_bad == 1, -15, 0))

        n_bad2  = (n_bad >= 2).sum()
        n_bad1  = (n_bad == 1).sum()
        print(f"  🚩 Volume filter: {n_bad2} mã rất xấu (-30đ) | {n_bad1} mã đáng ngờ (-15đ)")
        return pd.Series(flag, index=df.index)

    def calculate_total_score(self) -> pd.DataFrame:
        print("  → Tính Score_Momentum...")
        self.df["Score_Momentum"] = self._score_momentum()
        print("  → Tính Score_RS...")
        self.df["Score_RS"] = self._score_rs()
        print("  → Tính Score_Fundamental...")
        self.df["Score_Fundamental"] = self._score_fundamental()
        print("  → Kiểm tra chất lượng Volume...")
        self.df["Vol_Flag"] = self._vol_quality_flag()

        self.df["Total_Score"] = (
            self.df["Score_Momentum"] +
            self.df["Score_RS"] +
            self.df["Score_Fundamental"] +
            self.df["Vol_Flag"]          # hard penalty, KHÔNG clip trước
        ).clip(0, 100)

        print(f"  ✅ Chấm điểm xong. Score trung bình: {self.df['Total_Score'].mean():.1f}")
        return self.df


# ══════════════════════════════════════════════════════════════════════════════
#  3. ML PREDICTOR (XGBoost Walk-Forward)
# ══════════════════════════════════════════════════════════════════════════════

class MLPredictor:
    """
    Huấn luyện XGBoost với walk-forward validation để dự báo
    xác suất cổ phiếu tăng > 5% trong 20 phiên tới (ml_prob).
    """

    FEATURE_COLS = [
        "ret_1d", "ret_5d", "ret_20d", "ret_60d",
        "price_vs_ma20", "price_vs_ma50", "price_vs_ma200",
        "rsi_14", "rsi_28",
        "macd_hist",
        "bb_pct", "atr_pct",
        "vol_ratio_20", "vol_ratio_60",
        "dist_from_52w_high",
        "rs_20d", "rs_60d",
        "close_std_20",
        "Score_Momentum", "Score_RS", "Score_Fundamental",
        # Volume quality
        "vol_consistency", "vol_trend_slope", "price_vol_confirm",
        "distribution_day", "vol_spike_isolated",
    ]

    XGB_PARAMS = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    def __init__(self, df_scored: pd.DataFrame, forward_days: int = 20, target_return: float = 0.05):
        self.df = df_scored.copy()
        self.forward_days = forward_days
        self.target_return = target_return

    def _build_label(self) -> pd.DataFrame:
        """Label = 1 nếu giá tăng > target_return trong forward_days tới."""
        df = self.df.copy()
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        df["future_ret"] = (
            df.groupby("Ticker")["Close"]
            .transform(lambda x: x.shift(-self.forward_days) / x - 1)
        )
        df["label"] = (df["future_ret"] > self.target_return).astype(int)
        return df

    def _get_feature_matrix(self, df: pd.DataFrame):
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[available].copy()
        # Fill NaN bằng median của từng cột
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        X = X.replace([np.inf, -np.inf], 0)
        return X


    # ── Threshold Calibration ────────────────────────────────────────────────
    @staticmethod
    def calibrate_threshold(
        df_test_all: pd.DataFrame,
        min_precision: float = 0.55,
        min_coverage: float = 0.05,
    ) -> dict:
        """
        Sweep qua các ngưỡng ml_prob để tìm ngưỡng tối ưu dựa trên test set labels.

        Tiêu chí:
          - Precision >= min_precision  (khi lọc ra → phải đúng >= 55%)
          - Coverage  >= min_coverage   (phải lọc được >= 5% tổng số mã)
          - Tối đa hóa F1-score trong vùng thỏa điều kiện trên

        Returns dict:
          {
            "threshold_strong": float,   # ngưỡng Mua Mạnh (precision cao hơn)
            "threshold_buy":    float,   # ngưỡng Mua
            "precision_strong": float,
            "precision_buy":    float,
            "coverage_strong":  float,
            "coverage_buy":     float,
            "n_eval":           int,     # số mã có label thực để evaluate
          }
        """
        df = df_test_all.dropna(subset=["ml_prob", "label"]).copy()
        n_total = len(df)
        if n_total < 50:
            print("  ⚠️  Không đủ dữ liệu calibrate — dùng ngưỡng mặc định p75/p90.")
            p75 = df["ml_prob"].quantile(0.75) if not df.empty else 0.40
            p90 = df["ml_prob"].quantile(0.90) if not df.empty else 0.50
            return {
                "threshold_strong": p90, "threshold_buy": p75,
                "precision_strong": None, "precision_buy": None,
                "coverage_strong": 0.10, "coverage_buy": 0.25,
                "n_eval": n_total,
            }

        thresholds = np.arange(0.25, 0.85, 0.01)
        results = []
        for t in thresholds:
            mask = df["ml_prob"] >= t
            n_selected = mask.sum()
            coverage = n_selected / n_total
            if n_selected < 5:
                continue
            preds = mask.astype(int)
            precision = precision_score(df["label"], preds, zero_division=0)
            recall    = recall_score(df["label"], preds, zero_division=0)
            f1        = f1_score(df["label"], preds, zero_division=0)
            results.append({
                "threshold": t,
                "precision": precision,
                "recall":    recall,
                "f1":        f1,
                "coverage":  coverage,
                "n_selected": n_selected,
            })

        if not results:
            p75 = df["ml_prob"].quantile(0.75)
            p90 = df["ml_prob"].quantile(0.90)
            return {
                "threshold_strong": p90, "threshold_buy": p75,
                "precision_strong": None, "precision_buy": None,
                "coverage_strong": 0.10, "coverage_buy": 0.25,
                "n_eval": n_total,
            }

        res_df = pd.DataFrame(results)

        # Vùng thỏa mãn điều kiện cơ bản
        valid = res_df[(res_df["precision"] >= min_precision) &
                       (res_df["coverage"]  >= min_coverage)]

        if valid.empty:
            # Nới lỏng min_precision nếu không tìm được
            valid = res_df[res_df["coverage"] >= min_coverage]

        if valid.empty:
            valid = res_df

        # Ngưỡng "Mua" — F1 tốt nhất trong vùng valid
        best_buy = valid.loc[valid["f1"].idxmax()]

        # Ngưỡng "Mua Mạnh" — precision cao hơn ngưỡng buy, coverage >= 3%
        strong_candidates = res_df[
            (res_df["threshold"] > best_buy["threshold"]) &
            (res_df["coverage"]  >= 0.03)
        ]
        if not strong_candidates.empty:
            best_strong = strong_candidates.loc[strong_candidates["f1"].idxmax()]
        else:
            # fallback: p90
            t_p90 = df["ml_prob"].quantile(0.90)
            row = res_df.iloc[(res_df["threshold"] - t_p90).abs().argsort()[:1]]
            best_strong = row.iloc[0]

        print(
            f"  🎯 Threshold calibrated từ {n_total:,} mẫu:\n"
            f"     Mua      : prob >= {best_buy['threshold']:.3f}  "
            f"| precision={best_buy['precision']:.2%} "
            f"| coverage={best_buy['coverage']:.1%} "
            f"| F1={best_buy['f1']:.3f}\n"
            f"     Mua Mạnh : prob >= {best_strong['threshold']:.3f}  "
            f"| precision={best_strong['precision']:.2%} "
            f"| coverage={best_strong['coverage']:.1%} "
            f"| F1={best_strong['f1']:.3f}"
        )

        return {
            "threshold_strong": round(float(best_strong["threshold"]), 3),
            "threshold_buy":    round(float(best_buy["threshold"]), 3),
            "precision_strong": round(float(best_strong["precision"]), 4),
            "precision_buy":    round(float(best_buy["precision"]), 4),
            "coverage_strong":  round(float(best_strong["coverage"]), 4),
            "coverage_buy":     round(float(best_buy["coverage"]), 4),
            "n_eval":           n_total,
        }

    def train_walk_forward(self, n_splits: int = 5) -> pd.DataFrame:
        if not HAS_XGB:
            print("  ⚠️  XGBoost không có. Dùng score trực tiếp làm proxy ml_prob.")
            self.df["ml_prob"] = self.df["Total_Score"] / 100.0
            return self.df

        print("  → Tạo nhãn dự báo...")
        df = self._build_label()
        df = df.dropna(subset=["label", "Close"])

        all_dates = sorted(df["Date"].unique())
        n = len(all_dates)
        if n < 100:
            print("  ⚠️  Quá ít dữ liệu cho walk-forward. Dùng single train/test.")
            n_splits = 1

        split_size = n // (n_splits + 1)
        result_parts = []

        print(f"  → Walk-forward training ({n_splits} folds)...")
        for fold in range(n_splits):
            train_end_idx = split_size * (fold + 2)
            test_end_idx  = min(train_end_idx + split_size, n)

            if train_end_idx >= n:
                break

            train_dates = all_dates[:train_end_idx]
            test_dates  = all_dates[train_end_idx:test_end_idx]

            df_train = df[df["Date"].isin(train_dates)].dropna(subset=["label"])
            df_test  = df[df["Date"].isin(test_dates)]

            if len(df_train) < 200 or df_test.empty:
                continue

            X_train = self._get_feature_matrix(df_train)
            y_train = df_train["label"]
            X_test  = self._get_feature_matrix(df_test)

            # Chỉ dùng cột có trong cả train lẫn test
            common_cols = [c for c in X_train.columns if c in X_test.columns]
            X_train, X_test = X_train[common_cols], X_test[common_cols]

            model = xgb.XGBClassifier(**self.XGB_PARAMS)
            model.fit(X_train, y_train, verbose=False)

            probs = model.predict_proba(X_test)[:, 1]
            df_test = df_test.copy()
            df_test["ml_prob"] = probs
            result_parts.append(df_test)

            if HAS_SKLEARN:
                try:
                    auc = roc_auc_score(df_test["label"].dropna(), probs[:len(df_test["label"].dropna())])
                    print(f"    Fold {fold+1}: AUC={auc:.3f} | train={len(df_train):,} test={len(df_test):,}")
                except Exception:
                    print(f"    Fold {fold+1}: train={len(df_train):,} test={len(df_test):,}")
            else:
                print(f"    Fold {fold+1}: train={len(df_train):,} test={len(df_test):,}")

        if not result_parts:
            print("  ⚠️  Walk-forward không tạo được kết quả. Dùng Total_Score làm proxy.")
            df["ml_prob"] = df["Total_Score"] / 100.0
            return df

        df_out = pd.concat(result_parts, ignore_index=True)

        # Merge ml_prob vào df gốc (các ngày chưa được predict giữ NaN hoặc score proxy)
        df_merged = df.merge(df_out[["Date", "Ticker", "ml_prob"]].drop_duplicates(),
                             on=["Date", "Ticker"], how="left")
        df_merged["ml_prob"] = df_merged["ml_prob"].fillna(df_merged["Total_Score"] / 100.0)
        df_merged["ml_prob"] = df_merged["ml_prob"].clip(0, 1)

        # ── Auto-calibrate thresholds từ test set labels ────────────────────
        df_eval = pd.concat(result_parts, ignore_index=True)
        calibrated = MLPredictor.calibrate_threshold(df_eval)
        df_merged.attrs["threshold_strong"] = calibrated["threshold_strong"]
        df_merged.attrs["threshold_buy"]    = calibrated["threshold_buy"]
        df_merged.attrs["calibrated"]       = True

        print(f"  ✅ ML training xong. ml_prob trung bình: {df_merged['ml_prob'].mean():.3f}")
        return df_merged


# ══════════════════════════════════════════════════════════════════════════════
#  4. PORTFOLIO OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioOptimizer:
    """Lọc và xếp hạng danh mục tối ưu cho một ngày cụ thể."""

    @staticmethod
    def select_top_stocks(
        df_final: pd.DataFrame,
        target_date,
        top_n: int = 10,
        max_per_sector: int = 3,
        min_prob_threshold: float = 0.50,
        min_score: float = 40.0,
    ) -> pd.DataFrame:

        target_date = pd.to_datetime(target_date)

        # Lấy dữ liệu tại ngày gần nhất
        df_day = df_final[df_final["Date"] <= target_date].copy()
        if df_day.empty:
            return pd.DataFrame()

        # Lấy ngày mới nhất có trong dữ liệu
        latest = df_day.groupby("Ticker")["Date"].max().reset_index()
        latest.columns = ["Ticker", "LatestDate"]
        df_day = df_day.merge(latest, on="Ticker")
        df_day = df_day[df_day["Date"] == df_day["LatestDate"]].drop(columns=["LatestDate"])

        # Lọc theo ngưỡng
        df_day = df_day[df_day["ml_prob"]     >= min_prob_threshold]
        df_day = df_day[df_day["Total_Score"] >= min_score]

        # Loại bỏ bear market hoàn toàn nếu cần
        regime = df_day["market_regime"].mode()
        if len(regime) > 0 and regime.iloc[0] == "bear":
            top_n = max(3, top_n // 2)

        # Sắp xếp theo composite score
        df_day["composite"] = (
            df_day["ml_prob"]      * 0.5 +
            (df_day["Total_Score"] / 100.0) * 0.3 +
            df_day.get("rs_20d", pd.Series(0.0, index=df_day.index)).clip(-0.5, 0.5) * 0.2
        )
        df_day = df_day.sort_values("composite", ascending=False)

        # Giới hạn số mã mỗi ngành
        selected = []
        sector_count = {}
        for _, row in df_day.iterrows():
            sec = row.get("Sector", "Unknown")
            if sector_count.get(sec, 0) < max_per_sector:
                selected.append(row)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected) >= top_n:
                break

        if not selected:
            return pd.DataFrame()

        result = pd.DataFrame(selected).drop(columns=["composite"], errors="ignore")
        result = result.sort_values("ml_prob", ascending=False).reset_index(drop=True)
        return result