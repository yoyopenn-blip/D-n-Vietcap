"""
convert_to_parquet.py  ──  VN Stock Data Converter
═══════════════════════════════════════════════════════════════════════════════
Chuyển đổi raw Excel → Parquet tối ưu cho engine_v2.py

Cấu trúc thư mục:
    data/
    ├── raw/
    │   ├── prices.xlsx              (sheet: prices)
    │   ├── BCTC_THEO_QUÝ.xlsx      (sheets: BS1-BS9, IS1-IS4, CF1-CF3, COMP)
    │   ├── BCTC_THEO_NĂM.xlsx      (cùng cấu trúc)
    │   └── HISTORICAL_PRICES.xlsx  (sheet: Sheet1)
    ├── INDEX.xlsx                   (sheet: Table Data)
    └── processed/
        ├── prices.parquet
        ├── financials.parquet       ← engine_v2 dùng (từ BCTC QUÝ)
        ├── financial_yearly.parquet ← BCTC NĂM (cho phân tích dài hạn)
        ├── info.parquet
        └── index.parquet

Cách chạy:
    python convert_to_parquet.py           # chuyển tất cả
    python convert_to_parquet.py --prices  # chỉ chuyển giá
    python convert_to_parquet.py --bctc    # chỉ chuyển BCTC
    python convert_to_parquet.py --force   # ghi đè dù đã có parquet
"""

import argparse
import gc
import os
import re
import sys
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
RAW_DIR   = BASE_DIR / "data" / "raw"
PROC_DIR  = BASE_DIR / "data" / "processed"
INDEX_XLS = BASE_DIR / "data" / "INDEX.xlsx"

PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Tên file raw ───────────────────────────────────────────────────────────────
FILES = {
    "prices":     RAW_DIR / "prices.xlsx",
    "bctc_quy":   RAW_DIR / "BCTC_THEO_QUÝ.xlsx",
    "bctc_nam":   RAW_DIR / "BCTC_THEO_NĂM.xlsx",
    "hist_price": RAW_DIR / "HISTORICAL_PRICES.xlsx",
    "index":      INDEX_XLS,
}

# Sheets cần đọc trong BCTC (bỏ qua sheet mã hoá base64 và microsoft.com:*)
VALID_PREFIXES = ("BS", "IS", "CF")

# ── Cột cần lấy theo từng sheet (engine_v2.py dùng) ───────────────────────────
# Chỉ đọc đúng cột cần → tiết kiệm RAM ~70%
SHEET_COLS = {
    # Income Statement
    "IS1": {"Revenue from Business Activities - Total": "revenue"},
    "IS2": {
        "Cost of Revenues - Total":                    "cogs",
        "Gross Profit - Industrials/Property - Total": "gross_profit",
        "Net Income after Tax":                        "net_income",
        "Income before Taxes":                         "ebt",
    },
    "IS3": {
        "EPS - Basic - incl Extraordinary Items, Common - Total": "eps",
        "DPS - Common - Gross - Issue - By Announcement Date":    "dps",
    },
    # Balance Sheet
    "BS6": {
        "Total Liabilities":          "total_liabilities",
        "Common Equity - Total":      "equity",
        "Total Liabilities & Equity": "total_assets",
    },
    "BS8": {"Net Debt": "net_debt"},
    # Cash Flow
    "CF1": {
        "Net Cash Flow from Operating Activities": "cfo",
        "Capital Expenditures - Total":            "capex",
    },
    "CF3": {"Free Cash Flow to Equity": "fcfe"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_sheet_names(xlsx_path: Path) -> list[str]:
    """Lấy tên sheet bằng cách đọc zip (không load data → rất nhanh)."""
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            xml = z.read("xl/workbook.xml").decode("utf-8")
        return re.findall(r'name="([^"]+)"', xml)
    except Exception as e:
        print(f"  ⚠ Không đọc được sheet names từ {xlsx_path.name}: {e}")
        return []


def _is_valid_sheet(name: str) -> bool:
    """Sheet hợp lệ: BS1-BS9, IS1-IS4, CF1-CF3 (bỏ mã hoá và microsoft.com)."""
    return any(name.upper().startswith(p) for p in VALID_PREFIXES)


def _read_sheet_targeted(xlsx_path: Path, sheet: str,
                          col_map: dict) -> pd.DataFrame | None:
    """
    Đọc một sheet, chỉ lấy cột ticker/date + các cột trong col_map.
    Trả về None nếu sheet không có cột nào khớp.
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=1)
        available = {c: v for c, v in col_map.items() if c in df.columns}
        if not available:
            return None

        use_cols = [0, 1] + [
            i for i, c in enumerate(df.columns) if c in available
        ]
        df = pd.read_excel(xlsx_path, sheet_name=sheet,
                           usecols=use_cols)
    except Exception as e:
        print(f"    ⚠ Lỗi đọc {sheet}: {e}")
        return None

    df = df.rename(columns={"Unnamed: 0": "ticker", "Unnamed: 1": "date"})
    df["date"]         = pd.to_datetime(df["date"], errors="coerce")
    df["ticker_clean"] = df["ticker"].astype(str).str.split(".").str[0]
    df = df.dropna(subset=["ticker_clean", "date"])
    df = df.drop_duplicates(subset=["ticker_clean", "date"], keep="last")

    # Đổi tên cột theo col_map
    df = df.rename(columns=available)
    keep = ["ticker_clean", "date"] + list(available.values())
    return df[[c for c in keep if c in df.columns]]


def _read_sheet_full(xlsx_path: Path, sheet: str) -> pd.DataFrame | None:
    """
    Đọc toàn bộ cột của một sheet (dùng cho sheet không trong SHEET_COLS).
    Giới hạn 60 cột để tránh OutOfMemory với file lớn.
    """
    MAX_COLS = 60
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=0)
        use_cols = list(range(min(len(df.columns), MAX_COLS)))
        df = pd.read_excel(xlsx_path, sheet_name=sheet, usecols=use_cols)
    except Exception as e:
        print(f"    ⚠ Lỗi đọc {sheet}: {e}")
        return None

    if len(df.columns) < 3:
        return None

    # Chuẩn hoá 2 cột đầu → ticker, date
    cols = list(df.columns)
    cols[0] = "ticker"
    cols[1] = "date"
    df.columns = cols

    # Xoá cột Unnamed thừa
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    df["date"]         = pd.to_datetime(df["date"], errors="coerce")
    df["ticker_clean"] = df["ticker"].astype(str).str.split(".").str[0]
    df = df.dropna(subset=["ticker_clean", "date"])
    df = df.drop_duplicates(subset=["ticker_clean", "date"], keep="last")

    keep = ["ticker_clean", "date"] + [
        c for c in df.columns if c not in ("ticker", "ticker_clean", "date")
    ]
    return df[[c for c in keep if c in df.columns]]


# ══════════════════════════════════════════════════════════════════════════════
#  1. PRICES
# ══════════════════════════════════════════════════════════════════════════════

def convert_prices(force: bool = False):
    out = PROC_DIR / "prices.parquet"
    src = FILES["prices"]

    if out.exists() and not force:
        if src.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            print("✅ prices.parquet đã cập nhật. Bỏ qua (dùng --force để ghi đè).")
            return
    if not src.exists():
        print(f"❌ Không tìm thấy {src}. Bỏ qua.")
        return

    print(f"⏳ Đang chuyển đổi prices.xlsx → prices.parquet ...")
    t0 = datetime.now()

    df = pd.read_excel(src, sheet_name="prices")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])

    # Đảm bảo có cột ticker_clean
    if "ticker_clean" not in df.columns:
        if "ticker" in df.columns:
            df["ticker_clean"] = df["ticker"].astype(str).str.split(".").str[0]
        else:
            print("  ❌ Không tìm thấy cột ticker/ticker_clean trong prices.xlsx")
            return

    # Chuẩn hoá kiểu dữ liệu
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

    df = df.sort_values(["ticker_clean", "date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["ticker_clean", "date"], keep="last")

    df.to_parquet(out, index=False)
    elapsed = (datetime.now() - t0).seconds
    print(f"  ✅ prices.parquet — {df['ticker_clean'].nunique()} mã | "
          f"{len(df):,} dòng | {out.stat().st_size/1024/1024:.1f} MB | {elapsed}s")


# ══════════════════════════════════════════════════════════════════════════════
#  2. COMPANY INFO
# ══════════════════════════════════════════════════════════════════════════════

def convert_info(force: bool = False):
    out = PROC_DIR / "info.parquet"
    src = FILES["hist_price"]

    if out.exists() and not force:
        if src.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            print("✅ info.parquet đã cập nhật. Bỏ qua.")
            return
    if not src.exists():
        print(f"⚠ Không tìm thấy {src}. Thử dùng COMP_INFO.csv ...")
        _convert_info_from_csv(out)
        return

    print(f"⏳ Đang chuyển đổi HISTORICAL_PRICES.xlsx → info.parquet ...")
    t0 = datetime.now()

    df = pd.read_excel(src, sheet_name="Sheet1")
    # Lấy 7 cột đầu (bỏ cột date thừa ở cuối)
    df = df.iloc[:, :7]
    df.columns = ["ticker", "company_name", "country", "industry",
                  "sector", "ipo_date", "sector_vn"]
    df["ticker_clean"] = df["ticker"].astype(str).str.split(".").str[0]
    df["exchange"]     = df["ticker"].astype(str).str.split(".").str[1]
    df = df[["ticker_clean", "company_name", "industry", "sector",
             "sector_vn", "exchange"]].drop_duplicates("ticker_clean")
    df = df[df["ticker_clean"].notna() & (df["ticker_clean"] != "nan")]

    df.to_parquet(out, index=False)
    elapsed = (datetime.now() - t0).seconds
    print(f"  ✅ info.parquet — {len(df)} công ty | {elapsed}s")


def _convert_info_from_csv(out: Path):
    csv = BASE_DIR / "data" / "COMP_INFO.csv"
    if not csv.exists():
        print("  ❌ Không có HISTORICAL_PRICES.xlsx lẫn COMP_INFO.csv.")
        return
    df = pd.read_csv(csv)
    result = pd.DataFrame({
        "ticker_clean": df["symbol"].astype(str),
        "company_name": df.get("en_organ_name", df.get("organ_name", "")),
        "industry":     df.get("type", ""),
        "sector":       df.get("exchange", "Unknown"),
        "sector_vn":    df.get("exchange", "Unknown"),
        "exchange":     df.get("exchange", ""),
    }).drop_duplicates("ticker_clean")
    result.to_parquet(out, index=False)
    print(f"  ✅ info.parquet (từ COMP_INFO.csv) — {len(result)} công ty")


# ══════════════════════════════════════════════════════════════════════════════
#  3. INDEX
# ══════════════════════════════════════════════════════════════════════════════

def convert_index(force: bool = False):
    out = PROC_DIR / "index.parquet"
    src = FILES["index"]

    if out.exists() and not force:
        if src.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            print("✅ index.parquet đã cập nhật. Bỏ qua.")
            return
    if not src.exists():
        print(f"❌ Không tìm thấy {src}. Bỏ qua.")
        return

    print(f"⏳ Đang chuyển đổi INDEX.xlsx → index.parquet ...")
    df = pd.read_excel(src, sheet_name="Table Data")
    df = df.iloc[:, :2]
    df.columns = ["date", "vni"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "vni"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    df.to_parquet(out, index=False)
    print(f"  ✅ index.parquet — {len(df):,} ngày")


# ══════════════════════════════════════════════════════════════════════════════
#  4. BCTC (QUARTERLY / YEARLY)  ←  phần nặng nhất
# ══════════════════════════════════════════════════════════════════════════════

def _merge_bctc_sheets(xlsx_path: Path, label: str) -> pd.DataFrame | None:
    """
    Đọc tuần tự từng sheet BS/IS/CF, merge outer theo [ticker_clean, date].
    Dùng SHEET_COLS để chỉ đọc cột cần thiết → giảm RAM đáng kể.
    """
    all_sheets = _get_sheet_names(xlsx_path)
    valid = [s for s in all_sheets if _is_valid_sheet(s)]
    print(f"  Sheets hợp lệ: {len(valid)} / {len(all_sheets)} "
          f"({', '.join(valid[:10])}{'...' if len(valid)>10 else ''})")

    master: pd.DataFrame | None = None
    processed = 0

    for sheet in valid:
        print(f"    [{processed+1:>2}/{len(valid)}] {sheet:<6}", end="  ", flush=True)

        # Dùng targeted read nếu sheet có trong SHEET_COLS, ngược lại đọc full
        if sheet in SHEET_COLS:
            df_s = _read_sheet_targeted(xlsx_path, sheet, SHEET_COLS[sheet])
        else:
            df_s = _read_sheet_full(xlsx_path, sheet)

        if df_s is None or df_s.empty:
            print("(trống — bỏ qua)")
            continue

        rows, cols = len(df_s), len(df_s.columns) - 2
        print(f"{rows:>6,} dòng | {cols} cột tài chính")

        if master is None:
            master = df_s
        else:
            master = pd.merge(master, df_s, on=["ticker_clean", "date"], how="outer")
            # Giải phóng ngay sau merge
            del df_s
            gc.collect()

            # Chống bùng nổ dòng sau merge
            if master.duplicated(subset=["ticker_clean", "date"]).any():
                before = len(master)
                master = master.drop_duplicates(subset=["ticker_clean", "date"], keep="last")
                print(f"    ⚠ Dedup sau merge: {before:,} → {len(master):,}")

        processed += 1

    if master is None:
        print(f"  ❌ Không đọc được sheet nào từ {xlsx_path.name}")
        return None

    # Thêm gross_margin và roe tính sẵn
    if "gross_profit" in master.columns and "revenue" in master.columns:
        master["gross_margin"] = np.where(
            master["revenue"] > 0,
            master["gross_profit"] / master["revenue"].replace(0, np.nan),
            np.nan,
        )
    if "net_income" in master.columns and "equity" in master.columns:
        master["roe"] = np.where(
            master["equity"] > 0,
            master["net_income"] / master["equity"].replace(0, np.nan),
            np.nan,
        )
    if "total_liabilities" in master.columns and "equity" in master.columns:
        master["de_ratio"] = np.where(
            master["equity"] > 0,
            master["total_liabilities"] / master["equity"].replace(0, np.nan),
            np.nan,
        )

    master = master.sort_values(["ticker_clean", "date"]).reset_index(drop=True)
    print(f"\n  ✔ {label}: {master['ticker_clean'].nunique()} mã | "
          f"{len(master):,} dòng | {len(master.columns)} cột")
    return master


def convert_bctc_quarterly(force: bool = False):
    out = PROC_DIR / "financials.parquet"
    src = FILES["bctc_quy"]

    if out.exists() and not force:
        if src.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            print("✅ financials.parquet đã cập nhật. Bỏ qua.")
            return
    if not src.exists():
        print(f"❌ Không tìm thấy {src}. Bỏ qua.")
        return

    print(f"\n⏳ Đang chuyển đổi BCTC_THEO_QUÝ.xlsx → financials.parquet ...")
    t0 = datetime.now()

    df = _merge_bctc_sheets(src, "BCTC QUÝ")
    if df is None:
        return

    df.to_parquet(out, index=False)
    elapsed = (datetime.now() - t0).seconds
    mb = out.stat().st_size / 1024 / 1024
    print(f"  ✅ financials.parquet — {mb:.1f} MB | {elapsed}s")


def convert_bctc_yearly(force: bool = False):
    out = PROC_DIR / "financial_yearly.parquet"
    src = FILES["bctc_nam"]

    if out.exists() and not force:
        if src.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            print("✅ financial_yearly.parquet đã cập nhật. Bỏ qua.")
            return
    if not src.exists():
        print(f"❌ Không tìm thấy {src}. Bỏ qua.")
        return

    print(f"\n⏳ Đang chuyển đổi BCTC_THEO_NĂM.xlsx → financial_yearly.parquet ...")
    t0 = datetime.now()

    df = _merge_bctc_sheets(src, "BCTC NĂM")
    if df is None:
        return

    df.to_parquet(out, index=False)
    elapsed = (datetime.now() - t0).seconds
    mb = out.stat().st_size / 1024 / 1024
    print(f"  ✅ financial_yearly.parquet — {mb:.1f} MB | {elapsed}s")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Chuyển đổi raw Excel → Parquet cho VN Stock Screener"
    )
    ap.add_argument("--prices",   action="store_true", help="Chỉ chuyển giá")
    ap.add_argument("--info",     action="store_true", help="Chỉ chuyển company info")
    ap.add_argument("--index",    action="store_true", help="Chỉ chuyển VN-Index")
    ap.add_argument("--bctc",     action="store_true", help="Chỉ chuyển BCTC (cả quý lẫn năm)")
    ap.add_argument("--quarterly",action="store_true", help="Chỉ chuyển BCTC quý")
    ap.add_argument("--yearly",   action="store_true", help="Chỉ chuyển BCTC năm")
    ap.add_argument("--force",    action="store_true", help="Ghi đè dù đã có parquet")
    args = ap.parse_args()

    run_all = not any([args.prices, args.info, args.index,
                       args.bctc, args.quarterly, args.yearly])

    print("=" * 65)
    print("  VN Stock Data Converter  ──  raw Excel → Parquet")
    print(f"  RAW DIR  : {RAW_DIR}")
    print(f"  PROC DIR : {PROC_DIR}")
    print(f"  Force    : {args.force}")
    print("=" * 65)

    # Kiểm tra file raw
    missing = [k for k, p in FILES.items() if not p.exists()]
    if missing:
        print("\n⚠ File chưa có (sẽ bỏ qua):")
        for k in missing:
            print(f"   {FILES[k]}")

    print()
    t_total = datetime.now()

    if run_all or args.prices:
        convert_prices(force=args.force)
        print()

    if run_all or args.info:
        convert_info(force=args.force)
        print()

    if run_all or args.index:
        convert_index(force=args.force)
        print()

    if run_all or args.bctc or args.quarterly:
        convert_bctc_quarterly(force=args.force)

    if run_all or args.bctc or args.yearly:
        convert_bctc_yearly(force=args.force)

    elapsed = (datetime.now() - t_total).seconds
    print()
    print("=" * 65)
    print(f"🎉 Hoàn tất! Tổng thời gian: {elapsed // 60}m {elapsed % 60}s")
    print(f"   Parquet files trong: {PROC_DIR}")
    for f in sorted(PROC_DIR.glob("*.parquet")):
        mb = f.stat().st_size / 1024 / 1024
        print(f"   {f.name:<35} {mb:>6.1f} MB")
    print("=" * 65)
    print("\n➡  Bước tiếp theo: python main.py")


if __name__ == "__main__":
    main()