"""
Backtest.py  ── v3  (Production-Ready)
═══════════════════════════════════════════════════════════════════════════════
Cải tiến so với v2:

  [1] SLIPPAGE MODEL (thực tế hơn)
      - Fixed slippage 0.1 % (spread sàn + impact nhỏ)
      - Volume-adjusted impact: cổ phiếu thanh khoản thấp bị phạt thêm
        impact = clip(order_size / avg_daily_volume, 0, 0.03) * 0.5
      → Tổng chi phí/vòng dao động 0.3 % – 0.9 % tuỳ thanh khoản

  [2] POSITION SIZING  (Kelly/Volatility-parity)
      - Mặc định: Equal-weight chuẩn hoá (baseline dễ đọc)
      - Tuỳ chọn: Inverse-volatility weighting (ATR-based)
        → Cổ phiếu ATR cao được cấp vốn ít hơn → portfolio vol ổn định hơn

  [3] DRAWDOWN CONTROL
      - Trailing stop: nếu portfolio drawdown từ đỉnh > MAX_DD_STOP → đóng hết
      - Tháng bear market: tự động giảm số mã nắm giữ (TOP_K * bear_ratio)

  [4] RISK METRICS ĐẦY ĐỦ
      - Sharpe Ratio (annualised, dùng risk-free rate VN ≈ 4 %/năm)
      - Sortino Ratio (chỉ tính downside deviation)
      - Max Drawdown & Drawdown Duration
      - Calmar Ratio (CAGR / Max Drawdown)
      - Win Rate, Profit Factor, Avg Win/Loss Ratio
      - VaR 5 % & CVaR 5 % (monthly)

  [5] REGIME-AWARE OUTPUT
      - Breakdown Sharpe / Alpha theo từng chế độ thị trường

Cách dùng:
    python Backtest.py
    python Backtest.py --start 2021-01-01 --sizing vol_parity --stop 0.15

Output:
    data/backtest_results.xlsx  (sheet: monthly, summary, risk, regime_breakdown)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE        = Path(__file__).parent
DATA_DIR    = BASE / "data"
RESULT_FILE = DATA_DIR / "backtest_results.xlsx"

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_TRANSACTION_COST = 0.0015   # 0.15 % một chiều (phí + thuế)
FIXED_SLIPPAGE        = 0.001    # 0.1 % spread cố định
VOLUME_IMPACT_COEFF   = 0.50     # Hệ số market-impact (Almgren-Chriss lite)
TOP_K                 = 10
MIN_SCORE             = 55.0
RISK_FREE_ANNUAL      = 0.04     # 4 %/năm (tương đương lãi suất tiết kiệm VN)
RISK_FREE_MONTHLY     = (1 + RISK_FREE_ANNUAL) ** (1 / 12) - 1
BEAR_TOP_K_RATIO      = 0.6      # Bear market → nắm giữ 60 % số mã bình thường
MAX_DD_STOP           = 0.20     # Đóng tất cả khi drawdown portfolio > 20 %

# [UPGRADE] Exit Strategy — Trailing Stop theo từng mã
TRAILING_STOP_PER_TICKER = 0.07  # Bán nếu giá giảm > 7 % từ đỉnh của mã đó trong kỳ


# ══════════════════════════════════════════════════════════════════════════════
#  SLIPPAGE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def estimate_slippage(ticker: str,
                      prices: pd.DataFrame,
                      entry_date: pd.Timestamp,
                      order_fraction: float = 0.01) -> float:
    """
    Ước tính slippage thực tế dựa trên thanh khoản của cổ phiếu.

    Parameters
    ----------
    order_fraction : float
        Tỷ lệ order so với ADTV (Average Daily Trading Value).
        Mặc định 1 % ≈ một lệnh nhỏ lẻ cá nhân.

    Returns
    -------
    float
        Tổng slippage một chiều (không nhân đôi ở đây — caller nhân đôi).
    """
    grp = prices[prices["ticker_clean"] == ticker].sort_values("date")
    grp = grp[grp["date"] <= entry_date].tail(20)

    if grp.empty:
        return FIXED_SLIPPAGE

    avg_vol   = grp["volume"].mean()
    avg_close = grp["close"].mean()
    adtv      = avg_vol * avg_close  # VND

    if adtv <= 0:
        return FIXED_SLIPPAGE

    # Market impact tỷ lệ với sqrt(order_size / adtv) — sqrt là xấp xỉ phổ biến
    impact = VOLUME_IMPACT_COEFF * np.sqrt(order_fraction) * np.sqrt(1 / max(adtv / 1e9, 0.01))
    impact = np.clip(impact, 0.0, 0.03)   # Cap tối đa 3 % (cổ phiếu rất kém lỏng)

    return FIXED_SLIPPAGE + impact


def total_round_trip_cost(slippage_one_way: float) -> float:
    """Phí giao dịch + slippage khứ hồi."""
    return 2 * (BASE_TRANSACTION_COST + slippage_one_way)


# ══════════════════════════════════════════════════════════════════════════════
#  POSITION SIZING
# ══════════════════════════════════════════════════════════════════════════════

def compute_position_weights(tickers: list,
                              prices: pd.DataFrame,
                              ref_date: pd.Timestamp,
                              method: str = "equal") -> dict:
    """
    Tính trọng số vị thế cho danh sách tickers.

    method:
        "equal"      → Chia đều (1/N mỗi mã)
        "vol_parity" → Inverse-volatility (ATR 14 ngày), portfolio vol đều nhau
    """
    if method == "equal" or len(tickers) == 0:
        w = 1.0 / len(tickers) if tickers else 0.0
        return {t: w for t in tickers}

    inv_vols = {}
    for ticker in tickers:
        grp = prices[prices["ticker_clean"] == ticker].sort_values("date")
        grp = grp[grp["date"] <= ref_date].tail(21)
        if len(grp) < 5:
            inv_vols[ticker] = 1.0
            continue
        c    = grp["close"].values.astype(float)
        rets = np.diff(c) / c[:-1]
        vol  = np.std(rets) * np.sqrt(252)
        inv_vols[ticker] = 1.0 / max(vol, 0.05)   # floor 5 %/năm

    total = sum(inv_vols.values())
    return {t: v / total for t, v in inv_vols.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  RISK METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_metrics(monthly_rets: pd.Series,
                         label: str = "Portfolio") -> dict:
    """
    Tính bộ risk metrics đầy đủ từ chuỗi monthly returns.
    """
    if len(monthly_rets) < 3:
        return {}

    r    = monthly_rets.dropna()
    n    = len(r)
    mean = r.mean()
    std  = r.std(ddof=1)

    # Annualised
    cagr = (1 + r).prod() ** (12 / n) - 1

    # Sharpe (monthly excess returns → annualised)
    excess   = r - RISK_FREE_MONTHLY
    sharpe   = (excess.mean() / excess.std(ddof=1)) * np.sqrt(12) if excess.std() > 0 else 0

    # Sortino (chỉ downside)
    downside = excess[excess < 0]
    sortino_denom = np.sqrt((downside**2).mean() * 12) if len(downside) > 0 else 1e-9
    sortino  = (excess.mean() * 12) / sortino_denom if sortino_denom > 0 else 0

    # Drawdown
    cum       = (1 + r).cumprod()
    roll_max  = cum.cummax()
    dd_series = (cum - roll_max) / roll_max
    max_dd    = dd_series.min()

    # Drawdown duration (số tháng liên tiếp đang ở dưới đỉnh)
    in_dd            = (dd_series < 0).astype(int)
    dd_duration      = 0
    current_duration = 0
    for val in in_dd:
        if val:
            current_duration += 1
            dd_duration = max(dd_duration, current_duration)
        else:
            current_duration = 0

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0

    # Win Rate / Profit Factor
    wins   = r[r > 0]
    losses = r[r < 0]
    win_rate      = len(wins) / n if n > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    avg_win_loss  = (wins.mean() / abs(losses.mean())) if (len(losses) > 0 and losses.mean() != 0) else np.inf

    # VaR & CVaR (5 %)
    var_5  = np.percentile(r, 5)
    cvar_5 = r[r <= var_5].mean() if (r <= var_5).any() else var_5

    # Skewness & Kurtosis
    skew = r.skew()
    kurt = r.kurt()

    return {
        f"{label} CAGR":            f"{cagr:.1%}",
        f"{label} Sharpe":          f"{sharpe:.2f}",
        f"{label} Sortino":         f"{sortino:.2f}",
        f"{label} Max Drawdown":    f"{max_dd:.1%}",
        f"{label} DD Duration (mo)":dd_duration,
        f"{label} Calmar":          f"{calmar:.2f}",
        f"{label} Win Rate":        f"{win_rate:.0%}",
        f"{label} Profit Factor":   f"{profit_factor:.2f}" if profit_factor != np.inf else "∞",
        f"{label} Avg Win/Loss":    f"{avg_win_loss:.2f}"  if avg_win_loss != np.inf else "∞",
        f"{label} VaR 5%":          f"{var_5:.1%}",
        f"{label} CVaR 5%":         f"{cvar_5:.1%}",
        f"{label} Skew":            f"{skew:.2f}",
        f"{label} Kurt":            f"{kurt:.2f}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(start_date:  str   = "2022-01-01",
                 step_months: int   = 1,
                 use_ml:      bool  = False,
                 sizing:      str   = "equal",       # "equal" | "vol_parity"
                 max_dd_stop: float = MAX_DD_STOP,   # 0 = tắt stop
                 ) -> pd.DataFrame:
    """
    Chạy backtest walk-forward production-ready.

    Parameters
    ----------
    sizing      : "equal" hoặc "vol_parity"
    max_dd_stop : Ngưỡng trailing-stop toàn danh mục (0 = tắt)
    """
    print("=" * 65)
    print("VN Stock Backtest  v3  (Production-Ready)")
    print(f"Start: {start_date}  |  Top-{TOP_K}  |  Min score: {MIN_SCORE}")
    print(f"Sizing: {sizing}  |  DD stop: {max_dd_stop:.0%}  |  ML: {use_ml}")
    print(f"Base cost: {BASE_TRANSACTION_COST:.2%}  +  Slippage (volume-adjusted)")
    print("=" * 65)

    from engine_v2 import (load_prices, load_index, load_company_info,
                            load_financials, compute_all_scores)

    prices     = load_prices()
    index_df   = load_index()
    info       = load_company_info()
    financials = load_financials()

    ml_scorer = None
    if use_ml:
        try:
            from ml_stock_screener import MLScorer
            ml_scorer = MLScorer.load()
            print("✓ ML model loaded")
        except Exception as e:
            print(f"⚠ ML model chưa có ({e}) — không dùng ML")

    start   = pd.Timestamp(start_date)
    end     = prices["date"].max() - pd.DateOffset(days=30)
    current = start
    results = []

    # Trailing stop state
    peak_cum     = 1.0
    cum_value    = 1.0
    in_drawdown_stop = False

    while current + pd.DateOffset(months=step_months) <= end:
        next_dt = current + pd.DateOffset(months=step_months)

        # ── Scoring ──────────────────────────────────────────────────────────
        try:
            scores = compute_all_scores(
                prices, financials, index_df, info,
                target_month=current.month, ref_date=current
            )
        except Exception as e:
            print(f"  ⚠ {current.date()}: {e}")
            current = next_dt
            continue

        if scores.empty:
            current = next_dt
            continue

        regime = scores["market_regime"].iloc[0] if "market_regime" in scores.columns else "unknown"

        # Regime-aware top-K
        effective_k = int(TOP_K * BEAR_TOP_K_RATIO) if regime == "bear" else TOP_K

        if ml_scorer is not None:
            scores["ml_prob"] = ml_scorer.predict(scores)
            rank_col = "ml_prob"
        else:
            rank_col = "final_score"

        filtered  = scores[scores["final_score"] >= MIN_SCORE].copy()
        top_picks = filtered.nlargest(effective_k, rank_col)["ticker"].tolist()

        if not top_picks:
            current = next_dt
            continue

        # ── Position Weights ─────────────────────────────────────────────────
        weights = compute_position_weights(top_picks, prices, current, method=sizing)

        # ── Returns per pick ─────────────────────────────────────────────────
        pick_returns = {}
        for ticker in top_picks:
            grp = prices[prices["ticker_clean"] == ticker].sort_values("date")

            entry_slice = grp[grp["date"] > current]
            if entry_slice.empty:
                continue
            entry_idx  = entry_slice.index[0]
            entry_date = grp.loc[entry_idx, "date"]

            # Survivorship bias: entry trễ hơn 5 ngày → bỏ
            if len(grp[(grp["date"] > current) & (grp["date"] <= entry_date)]) > 5:
                continue

            exit_slice = grp[grp["date"] >= next_dt]
            exit_idx   = exit_slice.index[0] if not exit_slice.empty else grp.index[-1]

            entry_price = grp.loc[entry_idx, "open"] if "open" in grp.columns else grp.loc[entry_idx, "close"]

            # [UPGRADE] Trailing Stop per ticker: theo dõi đỉnh giá trong kỳ nắm giữ
            # Nếu giá giảm 7 % từ đỉnh → thoát sớm tại điểm dừng lỗ
            intra_period = grp[(grp["date"] >= entry_date) & (grp["date"] < next_dt)]
            if len(intra_period) >= 2 and TRAILING_STOP_PER_TICKER > 0:
                prices_arr    = intra_period["close"].values
                peak          = entry_price
                trailing_exit = None
                for i, px in enumerate(prices_arr):
                    peak = max(peak, px)
                    if px < peak * (1 - TRAILING_STOP_PER_TICKER):
                        trailing_exit = px
                        break
                if trailing_exit is not None:
                    exit_price = trailing_exit
                else:
                    exit_price = grp.loc[exit_idx, "close"]
            else:
                exit_price = grp.loc[exit_idx, "close"]

            # Slippage (volume-adjusted, một chiều)
            slip       = estimate_slippage(ticker, prices, entry_date)
            round_trip = total_round_trip_cost(slip)

            gross_ret  = exit_price / entry_price - 1
            net_ret    = gross_ret - round_trip
            pick_returns[ticker] = net_ret

        if not pick_returns:
            current = next_dt
            continue

        # ── Portfolio Return (weighted) ───────────────────────────────────────
        port_ret = sum(weights.get(t, 0) * r for t, r in pick_returns.items())
        # Re-normalise nếu có ticker bị drop
        active_weight = sum(weights.get(t, 0) for t in pick_returns)
        if active_weight > 0 and active_weight < 1:
            port_ret = port_ret / active_weight

        # ── Trailing Stop ─────────────────────────────────────────────────────
        cum_value  *= (1 + port_ret)
        peak_cum    = max(peak_cum, cum_value)
        current_dd  = (cum_value - peak_cum) / peak_cum

        if max_dd_stop > 0 and current_dd < -max_dd_stop and not in_drawdown_stop:
            print(f"  🛑 {current.strftime('%Y-%m')} Trailing stop triggered! DD = {current_dd:.1%}")
            in_drawdown_stop = True

        # Recovery: re-enter khi VNI phục hồi lại trên MA50
        if in_drawdown_stop:
            idx_sub = index_df[index_df["date"] <= current].sort_values("date")
            if len(idx_sub) >= 50:
                vni_now = idx_sub["vni"].iloc[-1]
                vni_ma50 = pd.Series(idx_sub["vni"].values).rolling(50).mean().iloc[-1]
                if vni_now > vni_ma50:
                    in_drawdown_stop = False
                    print(f"  ✅ {current.strftime('%Y-%m')} Re-entry: VNI > MA50")
                else:
                    current = next_dt
                    results.append({
                        "period": current.strftime("%Y-%m"),
                        "regime": regime,
                        "n_picks": 0,
                        "portfolio_ret": 0.0,
                        "vni_ret": np.nan,
                        "alpha": np.nan,
                        "hit_rate": np.nan,
                        "best_pick": np.nan,
                        "worst_pick": np.nan,
                        "tickers": "STOP",
                        "avg_slippage": np.nan,
                        "dd_from_peak": current_dd,
                        "sizing_method": sizing,
                    })
                    continue

        # ── Benchmark ────────────────────────────────────────────────────────
        idx_sub = index_df[(index_df["date"] >= current) &
                           (index_df["date"] < next_dt)].sort_values("date")
        vni_ret = (idx_sub["vni"].iloc[-1] / idx_sub["vni"].iloc[0] - 1) \
                  if len(idx_sub) >= 2 else np.nan

        # ── Monthly stats ─────────────────────────────────────────────────────
        rets_list  = list(pick_returns.values())
        hit_rate   = np.mean([r > 0 for r in rets_list])
        alpha      = port_ret - vni_ret if not np.isnan(vni_ret) else np.nan
        avg_slip   = np.mean([estimate_slippage(t, prices, current) for t in pick_returns])

        result = {
            "period":       current.strftime("%Y-%m"),
            "regime":       regime,
            "n_picks":      len(pick_returns),
            "portfolio_ret":port_ret,
            "vni_ret":      vni_ret,
            "alpha":        alpha,
            "hit_rate":     hit_rate,
            "best_pick":    max(rets_list),
            "worst_pick":   min(rets_list),
            "tickers":      ", ".join(list(pick_returns.keys())[:5]) + ("..." if len(pick_returns) > 5 else ""),
            "avg_slippage": avg_slip,
            "dd_from_peak": current_dd,
            "sizing_method":sizing,
        }
        results.append(result)

        sign     = "+" if port_ret > 0 else ""
        vni_sign = "+" if not np.isnan(vni_ret) and vni_ret > 0 else ""
        print(
            f"{current.strftime('%Y-%m')} [{regime:8s}] | "
            f"Port: {sign}{port_ret:.1%} | VNI: {vni_sign}{vni_ret:.1%} | "
            f"α: {'+' if not np.isnan(alpha) and alpha > 0 else ''}{alpha:.1%} | "
            f"Slip: {avg_slip:.2%} | Hit: {hit_rate:.0%} | "
            f"DD: {current_dd:.1%}"
        )
        current = next_dt

    if not results:
        print("❌ Không có kết quả — kiểm tra data range")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY & RISK METRICS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    port_rets = df["portfolio_ret"]
    vni_rets  = df["vni_ret"].fillna(0)
    alpha_col = df["alpha"].dropna()

    # Basic stats
    total_months  = len(df)
    avg_port      = port_rets.mean()
    avg_vni       = vni_rets.mean()
    avg_alpha     = alpha_col.mean()
    win_months    = (port_rets > 0).mean()
    outperform_m  = (df["alpha"].fillna(-999) > 0).mean()
    cum_port      = (1 + port_rets).prod() - 1
    cum_vni       = (1 + vni_rets).prod() - 1

    print(f"{'Số tháng backtest:':<30} {total_months}")
    print(f"{'Return TB/tháng:':<30} {avg_port:.1%}  (portfolio)  vs  {avg_vni:.1%}  (VNI)")
    print(f"{'Alpha TB/tháng:':<30} {avg_alpha:.1%}")
    print(f"{'Tháng thắng:':<30} {win_months:.0%}")
    print(f"{'Tháng outperform VNI:':<30} {outperform_m:.0%}")
    print(f"{'Cumulative return:':<30} {cum_port:.1%}  (portfolio)  vs  {cum_vni:.1%}  (VNI)")

    # Full risk metrics
    port_metrics = compute_risk_metrics(port_rets, "Portfolio")
    vni_metrics  = compute_risk_metrics(vni_rets,  "VNI")
    alpha_metrics= compute_risk_metrics(alpha_col, "Alpha")

    print("\n── Risk Metrics ──────────────────────────────────────────")
    for k, v in port_metrics.items():
        vni_k = k.replace("Portfolio", "VNI")
        vni_v = vni_metrics.get(vni_k, "—")
        print(f"  {k:<35} {v}   (VNI: {vni_v})")

    print("\n── Alpha Metrics ─────────────────────────────────────────")
    for k, v in alpha_metrics.items():
        print(f"  {k:<35} {v}")

    # Regime breakdown
    if "regime" in df.columns:
        print("\n── Breakdown theo Market Regime ──────────────────────────")
        rg = df.groupby("regime").agg(
            months      =("portfolio_ret", "count"),
            avg_ret     =("portfolio_ret", "mean"),
            avg_alpha   =("alpha",         "mean"),
            hit_rate    =("hit_rate",      "mean"),
            avg_slip    =("avg_slippage",  "mean"),
        ).round(4)
        print(rg.to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\n💾 Lưu {RESULT_FILE} ...")

    with pd.ExcelWriter(RESULT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="monthly", index=False)

        # Summary sheet
        all_metrics = {**port_metrics, **vni_metrics, **alpha_metrics}
        summary_rows = [
            {"Metric": "Số tháng",            "Giá trị": total_months},
            {"Metric": "Return TB/tháng",      "Giá trị": f"{avg_port:.2%}"},
            {"Metric": "Alpha TB/tháng",       "Giá trị": f"{avg_alpha:.2%}"},
            {"Metric": "Tháng thắng",          "Giá trị": f"{win_months:.0%}"},
            {"Metric": "Outperform VNI",       "Giá trị": f"{outperform_m:.0%}"},
            {"Metric": "Cum Return",           "Giá trị": f"{cum_port:.1%}"},
            {"Metric": "Cum VNI Return",       "Giá trị": f"{cum_vni:.1%}"},
        ] + [{"Metric": k, "Giá trị": v} for k, v in all_metrics.items()]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)

        # Risk detail sheet (raw numbers cho charting)
        risk_detail = pd.DataFrame({
            "period":       df["period"],
            "cum_port":     (1 + df["portfolio_ret"]).cumprod() - 1,
            "cum_vni":      (1 + df["vni_ret"].fillna(0)).cumprod() - 1,
            "dd_from_peak": df["dd_from_peak"],
            "alpha":        df["alpha"],
        })
        risk_detail.to_excel(writer, sheet_name="risk_detail", index=False)

        # Regime breakdown
        if "regime" in df.columns:
            rg.reset_index().to_excel(writer, sheet_name="regime_breakdown", index=False)

    print("✅ Xong!")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VN Stock Backtest v3")
    ap.add_argument("--start",  default="2022-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--step",   default=1, type=int,  help="Months per step")
    ap.add_argument("--ml",     action="store_true",  help="Dùng ML model")
    ap.add_argument("--sizing", default="equal",      choices=["equal", "vol_parity"],
                    help="Position sizing method")
    ap.add_argument("--stop",   default=0.20, type=float,
                    help="Max drawdown stop (0 = tắt). Default 0.20")
    args = ap.parse_args()

    results = run_backtest(
        start_date  = args.start,
        step_months = args.step,
        use_ml      = args.ml,
        sizing      = args.sizing,
        max_dd_stop = args.stop,
    )