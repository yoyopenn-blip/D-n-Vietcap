"""
debug_ml.py
───────────
Chạy script này để chẩn đoán lỗi ml_stock_screener.py trước khi train.

Cách dùng:
    python debug_ml.py
"""

import pandas as pd
import numpy as np
import traceback
from pathlib import Path

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"


def check_file(path, label):
    if path.exists():
        size = path.stat().st_size / 1024
        print(f"  ✅ {label}: {path.name} ({size:.0f} KB)")
        return True
    else:
        print(f"  ❌ {label}: {path.name} — KHÔNG TỒN TẠI")
        return False


print("=" * 60)
print("DEBUG: VN Stock ML Diagnostics")
print("=" * 60)

# ── 1. Kiểm tra file tồn tại ──────────────────────────────────────────────────
print("\n[1] Kiểm tra file data/")
files = {
    "prices.parquet":     DATA_DIR / "prices.parquet",
    "prices.xlsx":        DATA_DIR / "prices.xlsx",
    "financials.parquet": DATA_DIR / "financials.parquet",
    "index.parquet":      DATA_DIR / "index.parquet",
    "INDEX.xlsx":         DATA_DIR / "INDEX.xlsx",
    "info.parquet":       DATA_DIR / "info.parquet",
    "HISTORICAL_PRICES":  DATA_DIR / "HISTORICAL_PRICES.xlsx",
    "BCTC (optional)":    DATA_DIR / "BCTC_THEO_QUÝ.xlsx",
}
for label, path in files.items():
    check_file(path, label)

# ── 2. Load và kiểm tra từng DataFrame ───────────────────────────────────────
print("\n[2] Load data qua engine_v2")
try:
    from engine_v2 import load_prices, load_index, load_company_info, load_financials
except Exception as e:
    print(f"  ❌ Import engine_v2 thất bại: {e}")
    raise SystemExit(1)

# prices
print("\n  --- prices ---")
try:
    prices = load_prices()
    print(f"  ✅ Shape: {prices.shape}")
    print(f"  Columns: {list(prices.columns)}")
    print(f"  date range: {prices['date'].min().date()} → {prices['date'].max().date()}")
    print(f"  Số mã: {prices['ticker_clean'].nunique()}")
    print(f"  NaN close: {prices['close'].isna().sum()}")
except Exception as e:
    print(f"  ❌ load_prices() thất bại: {e}")
    traceback.print_exc()

# index
print("\n  --- index ---")
try:
    index_df = load_index()
    print(f"  ✅ Shape: {index_df.shape}")
    print(f"  Columns: {list(index_df.columns)}")
    print(f"  date range: {index_df['date'].min().date()} → {index_df['date'].max().date()}")
    print(f"  NaN vni: {index_df['vni'].isna().sum()}")
except Exception as e:
    print(f"  ❌ load_index() thất bại: {e}")
    traceback.print_exc()

# info
print("\n  --- company info ---")
try:
    info = load_company_info()
    print(f"  ✅ Shape: {info.shape}")
    print(f"  Columns: {list(info.columns)}")
    print(f"  Sample ticker_clean: {info['ticker_clean'].head(5).tolist()}")
except Exception as e:
    print(f"  ❌ load_company_info() thất bại: {e}")
    traceback.print_exc()

# financials
print("\n  --- financials ---")
try:
    financials = load_financials()
    print(f"  ✅ Shape: {financials.shape}")
    print(f"  Columns: {list(financials.columns)}")
    if not financials.empty:
        print(f"  date range: {financials['date'].min().date()} → {financials['date'].max().date()}")
        print(f"  Số mã: {financials['ticker_clean'].nunique()}")
    else:
        print(f"  ⚠️  financials RỖNG — thiếu file BCTC_THEO_QUÝ.xlsx")
        print(f"  → Scores vẫn chạy được nhưng phần fundamental = 0")
except Exception as e:
    print(f"  ❌ load_financials() thất bại: {e}")
    traceback.print_exc()
    financials = pd.DataFrame()

# ── 3. Test compute_all_scores trên 1 tháng cụ thể ───────────────────────────
print("\n[3] Test compute_all_scores (1 tháng)")
try:
    from engine_v2 import compute_all_scores
    ref = pd.Timestamp("2024-06-01")
    print(f"  ref_date = {ref.date()}, target_month = 6")
    sc = compute_all_scores(prices, financials, index_df, info,
                            target_month=6, ref_date=ref)
    if sc.empty:
        print(f"  ❌ compute_all_scores trả về RỖNG tại {ref.date()}")
        print(f"     → Kiểm tra: prices có data trước {ref.date()} không?")
        n_before = (prices["date"] <= ref).sum()
        print(f"     Rows trong prices trước ref_date: {n_before}")
        tickers_60 = prices[prices["date"] <= ref].groupby("ticker_clean").size()
        ok = (tickers_60 >= 60).sum()
        print(f"     Mã có >= 60 phiên trước ref_date: {ok}/{len(tickers_60)}")
    else:
        print(f"  ✅ OK — {len(sc)} mã | columns: {list(sc.columns[:8])}...")
except Exception as e:
    print(f"  ❌ compute_all_scores exception: {e}")
    traceback.print_exc()

# ── 4. Test với ref_date sớm hơn (2021) ──────────────────────────────────────
print("\n[4] Test compute_all_scores tại 2021-06-01 (ngày xa nhất)")
try:
    ref_early = pd.Timestamp("2021-06-01")
    n_before_early = (prices["date"] <= ref_early).sum()
    tickers_early  = prices[prices["date"] <= ref_early].groupby("ticker_clean").size()
    ok_early = (tickers_early >= 60).sum()
    print(f"  Rows trong prices trước {ref_early.date()}: {n_before_early}")
    print(f"  Mã có >= 60 phiên: {ok_early}/{len(tickers_early)}")

    if ok_early == 0:
        min_date = prices["date"].min()
        print(f"\n  ⚠️  PHÁT HIỆN VẤN ĐỀ CHÍNH:")
        print(f"  prices chỉ bắt đầu từ {min_date.date()}")
        print(f"  Cần ít nhất 60 phiên (~3 tháng) trước start_date của ml_model")
        need_start = min_date + pd.DateOffset(days=90)
        print(f"\n  → FIX: Đổi start_date trong ml_stock_screener.py từ '2021-01-01' thành '{need_start.date()}'")
        print(f"  → Hoặc chạy: python fetch_prices.py --days 1825  (để tải đủ 5 năm)")
except Exception as e:
    print(f"  ❌ {e}")
    traceback.print_exc()

# ── 5. Kiểm tra parquet cache có bị stale không ───────────────────────────────
print("\n[5] Kiểm tra parquet cache timestamps")
cache_files = [
    ("prices.parquet", DATA_DIR/"prices.parquet", DATA_DIR/"prices.xlsx"),
    ("index.parquet",  DATA_DIR/"index.parquet",  DATA_DIR/"INDEX.xlsx"),
    ("info.parquet",   DATA_DIR/"info.parquet",   DATA_DIR/"HISTORICAL_PRICES.xlsx"),
]
for name, pq_path, src_path in cache_files:
    if pq_path.exists() and src_path.exists():
        pq_mtime  = pq_path.stat().st_mtime
        src_mtime = src_path.stat().st_mtime
        if pq_mtime < src_mtime:
            print(f"  ⚠️  {name}: PARQUET CŨ HƠN SOURCE — cần xóa cache!")
            print(f"     Xóa: del data\\{pq_path.name}")
        else:
            print(f"  ✅ {name}: cache mới hơn source, OK")
    elif pq_path.exists():
        print(f"  ℹ️  {name}: chỉ có parquet (không có xlsx source)")

print("\n" + "=" * 60)
print("Chẩn đoán hoàn tất. Xem thông báo ⚠️ và ❌ ở trên để fix.")
print("=" * 60)