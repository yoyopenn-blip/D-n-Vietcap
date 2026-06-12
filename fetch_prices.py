import time
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Cấu hình ──────────────────────────────────────────────────────────────────
HISTORY_DAYS = 365 * 5  # 5 năm
API_SLEEP = 0.50  # Tăng nhẹ thời gian nghỉ để tránh bị block
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "prices.xlsx"
INDEX_FILE = OUTPUT_DIR / "INDEX.xlsx"

# ── Danh sách ~400 mã HOSE (Dùng luôn danh sách này làm mặc định, không sợ API sập) ──
TICKERS = [
    "AAA", "AAM", "AAT", "ABR", "ABS", "ABT", "ACB", "ACC", "ACG", "ACL", "ADG", "ADS", "AGG", "AGM", "AGR",
    "AMD", "ANV", "APC", "APG", "APH", "ASG", "ASM", "ASP", "AST", "BAF", "BAX", "BCE", "BCG", "BCM", "BFC",
    "BHN", "BIC", "BID", "BKG", "BMC", "BMI", "BMP", "BRC", "BSI", "BTP", "BTT", "BVH", "BWE", "C32", "C47",
    "CAV", "CCI", "CCL", "CDC", "CEV", "CHP", "CIG", "CII", "CKG", "CLC", "CLL", "CLW", "CMG", "CMV", "CNC",
    "CNG", "COM", "CRC", "CRE", "CSM", "CSV", "CTD", "CTF", "CTG", "CTI", "CTR", "CTS", "CVT", "D2D", "DAG",
    "DAH", "DAT", "DBC", "DBD", "DBT", "DC4", "DCL", "DCM", "DGC", "DGW", "DHA", "DHC", "DHG", "DHM", "DIG",
    "DLG", "DMC", "DPG", "DPM", "DPR", "DQC", "DRC", "DRL", "DS3", "DSN", "DTA", "DTL", "DTT", "DVP", "DXG",
    "DXS", "DXV", "EIB", "ELC", "EMC", "EVE", "EVF", "FBT", "FCM", "FCN", "FDC", "FIR", "FIT", "FLD", "FMC",
    "FPT", "FRT", "FTS", "GAS", "GDT", "GEG", "GEX", "GIL", "GMC", "GMD", "GSP", "GTA", "GTS", "GVR", "HAG",
    "HAH", "HAI", "HAP", "HAR", "HAS", "HAX", "HBC", "HCD", "HCM", "HDB", "HDC", "HDG", "HDP", "HHP", "HHS",
    "HIG", "HII", "HNG", "HOT", "HPG", "HQC", "HRC", "HSG", "HSL", "HT1", "HTI", "HTL", "HTN", "HTV", "HU1",
    "HU3", "HUB", "HVH", "HVN", "HVX", "IBC", "ICT", "IDC", "IDI", "IJC", "ILB", "IMP", "ITA", "ITC", "ITD",
    "JVC", "KBC", "KDC", "KDH", "KHG", "KHP", "KMR", "KOS", "KPF", "KSB", "L10", "LBM", "LCG", "LCW", "LDG",
    "LEC", "LGC", "LGL", "LHM", "LIX", "LM8", "LPB", "LSS", "MBB", "MCG", "MCP", "MDG", "MHC", "MIG", "MSB",
    "MSH", "MSN", "MWG", "NAA", "NAF", "NAV", "NBB", "NCT", "NHA", "NHH", "NHT", "NKG", "NLG", "NNC", "NSC",
    "NT2", "NTL", "NVL", "OCB", "OPC", "ORS", "PAC", "PAN", "PC1", "PDN", "PDR", "PET", "PGC", "PGD", "PGI",
    "PGV", "PHC", "PHR", "PIT", "PJT", "PLP", "PLX", "PME", "PNJ", "POM", "POW", "PPC", "PSH", "PTB", "PTC",
    "PTL", "PVD", "PVT", "QBS", "QCG", "RAL", "RDP", "REE", "S4A", "SAB", "SAM", "SAV", "SBA", "SBT", "SBV",
    "SC5", "SCD", "SCR", "SCS", "SFC", "SFI", "SGN", "SGR", "SHA", "SHB", "SHI", "SHP", "SJD", "SJF", "SJS",
    "SKG", "SMA", "SMB", "SMC", "SPM", "SRC", "SRF", "SSG", "SSI", "ST8", "STB", "STG", "STK", "SVC", "SVD",
    "SVI", "SVT", "SZC", "SZL", "TBC", "TCB", "TCD", "TCH", "TCL", "TCM", "TCO", "TCR", "TCT", "TDC", "TDG",
    "TDH", "TDM", "TDP", "TDW", "TEG", "THG", "THI", "TIP", "TIX", "TLG", "TLH", "TMP", "TMS", "TMT", "TN1",
    "TNA", "TNC", "TNH", "TNI", "TNT", "TPB", "TRC", "TSC", "TTA", "TTB", "TTE", "TTF", "TVB", "TV2", "TVT",
    "TYA", "UDC", "UIC", "VAF", "VCA", "VCB", "VCF", "VCG", "VCI", "VDP", "VDS", "VFG", "VGC", "VHC", "VHM",
    "VIB", "VIC", "VID", "VIP", "VIX", "VJC", "VMD", "VND", "VNE", "VNG", "VNL", "VNM", "VNS", "VOS", "VPB",
    "VPD", "VPG", "VPH", "VPI", "VPS", "VRC", "VRE", "VSC", "VSH", "VSI", "VTB", "VTO", "YBM", "YEG"
]

INDEX_SYMBOL = "VNINDEX"


# ═════════════════════════════════════════════════════════════════════════════
# API FETCHERS
# ═════════════════════════════════════════════════════════════════════════════

def fetch_ssi(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """SSI iBoard API — Chia nhỏ request mỗi 6 tháng để không bị cắt data"""
    url = "https://iboard-api.ssi.com.vn/statistics/company/ssmi/stock-info"
    headers = {
        "Accept": "application/json",
        "Connection": "keep-alive",
        "Origin": "https://iboard.ssi.com.vn",
        "Referer": "https://iboard.ssi.com.vn/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    all_chunks = []
    current_start = start_dt

    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=180), end_dt)
        params = {
            "symbol": symbol,
            "page": 1,
            "pageSize": 5000,
            "fromDate": current_start.strftime("%d/%m/%Y"),
            "toDate": current_end.strftime("%d/%m/%Y"),
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=8)
            if r.status_code == 200:
                d = r.json()
                if d.get("code") == "SUCCESS" and d.get("data"):
                    df_chunk = pd.DataFrame(d["data"])
                    if not df_chunk.empty:
                        df_chunk = df_chunk[["tradingDate", "open", "high", "low", "close", "volume"]].copy()
                        df_chunk.columns = ["date", "open", "high", "low", "close", "volume"]
                        all_chunks.append(df_chunk)
        except Exception:
            pass

        current_start = current_end + timedelta(days=1)
        time.sleep(0.1)

    if all_chunks:
        df = pd.concat(all_chunks, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return pd.DataFrame()


def fetch_vndirect(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """VNDirect Open API — Fallback"""
    url = "https://finfo-api.vndirect.com.vn/v4/stock_prices"

    all_chunks = []
    current_start = start_dt

    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=180), end_dt)
        q_str = (
            f"code:{symbol}"
            f"~date:gte:{current_start.strftime('%Y-%m-%d')}"
            f"~date:lte:{current_end.strftime('%Y-%m-%d')}"
        )
        try:
            r = requests.get(url, params={"sort": "date", "q": q_str, "size": 5000},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    df_chunk = pd.DataFrame(data)
                    df_chunk = df_chunk[["date", "adOpen", "adHigh", "adLow", "adClose", "nmVolume"]].copy()
                    df_chunk.columns = ["date", "open", "high", "low", "close", "volume"]
                    all_chunks.append(df_chunk)
        except Exception:
            pass

        current_start = current_end + timedelta(days=1)
        time.sleep(0.1)

    if all_chunks:
        df = pd.concat(all_chunks, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce") * 1000
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return pd.DataFrame()


def fetch_ticker(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    df = fetch_ssi(symbol, start_dt, end_dt)
    if df.empty:
        df = fetch_vndirect(symbol, start_dt, end_dt)
    if not df.empty:
        df["ticker_clean"] = symbol
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD & LƯU TRỮ
# ═════════════════════════════════════════════════════════════════════════════

def get_all_hose_tickers() -> list:
    """Kéo danh sách HOSE. Nếu sập mạng, trả về sẵn danh sách 396 mã HOSE ở trên."""
    url = "https://finfo-api.vndirect.com.vn/v4/stocks"
    params = {"q": "floor:HOSE~type:STOCK", "size": 1000}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json().get("data", [])
            tickers = [item["symbol"] for item in data if len(item["symbol"]) == 3]
            if len(tickers) > 200:
                print(f"\n✅ Tìm thấy {len(tickers)} mã cổ phiếu trên HOSE qua API.")
                return sorted(tickers)
    except:
        pass

    print(f"\n⚠ API Danh sách cổ phiếu bị lỗi mạng. Tự động dùng danh sách {len(TICKERS)} mã HOSE được nạp sẵn.")
    return TICKERS


def download_all(tickers: list, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    total = len(tickers)
    t0 = time.time()
    print(f"📡 Tải {total} mã | {start_dt.date()} → {end_dt.date()}")
    print(f"   Nguồn: SSI iBoard → VNDirect | ⚡ Song song 20 luồng\n")

    args_list = [(sym, start_dt, end_dt, i+1, total) for i, sym in enumerate(tickers)]
    all_dfs = []

    # MAX_WORKERS = 20 là điểm cân bằng tốt (không bị block IP)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_ticker_safe, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_dfs.append(result)

    elapsed = time.time() - t0
    success = len(all_dfs)
    print(f"\n   ✔ Xong: {success}/{total} mã | {elapsed:.0f}s ({elapsed/60:.1f} phút)")

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna(subset=["date", "close"])
    df = df.drop_duplicates(subset=["date", "ticker_clean"])
    return df.sort_values(["ticker_clean", "date"]).reset_index(drop=True)

def download_vnindex(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    print(f"\n📈 Tải VN-Index | {start_dt.date()} → {end_dt.date()} ...", end=" ", flush=True)
    df = fetch_ticker(INDEX_SYMBOL, start_dt, end_dt)
    if not df.empty:
        df = df.rename(columns={"close": "vni"})[["date", "vni"]]
        print(f"✅  {len(df)} ngày")
    else:
        print("❌  Không có data")
    return df


def save_excel(df_prices: pd.DataFrame, df_index: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    n = df_prices["ticker_clean"].nunique()
    if n == 0:
        print("❌ Không có dữ liệu để lưu.")
        return

    d1 = df_prices["date"].min().date()
    d2 = df_prices["date"].max().date()
    avg_rows = len(df_prices) / n
    print(f"\n💾 Lưu {OUTPUT_FILE} ...")
    print(f"   {len(df_prices):,} dòng | {n} mã | {d1} → {d2}")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_prices.to_excel(writer, sheet_name="prices", index=False)
        summary = (
            df_prices.groupby("ticker_clean")
            .agg(rows=("close", "count"),
                 date_start=("date", "min"),
                 date_end=("date", "max"),
                 last_close=("close", "last"))
            .reset_index()
        )
        summary.to_excel(writer, sheet_name="summary", index=False)
    print(f"   ✅ {OUTPUT_FILE.stat().st_size / 1024:.0f} KB")

    if not df_index.empty:
        with pd.ExcelWriter(INDEX_FILE, engine="openpyxl") as writer:
            df_index.to_excel(writer, sheet_name="Table Data", index=False)
        print(f"   ✅ VN-Index lưu: {len(df_index)} ngày")

    for pq in [OUTPUT_DIR / "prices.parquet", OUTPUT_DIR / "index.parquet"]:
        if pq.exists():
            pq.unlink()
            print(f"   🗑️  Đã xóa cache: {pq.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=HISTORY_DAYS)
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("  VN Stock Price Fetcher v3 | Tự động quét toàn bộ HOSE")
    print("=" * 60)

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=args.days)

    if args.tickers:
        tickers = args.tickers
    else:
        tickers = get_all_hose_tickers()

    if not args.update:
        if OUTPUT_FILE.exists():
            print(f"🗑️ Chế độ tải full 5 năm. Tự động xóa file cũ: {OUTPUT_FILE.name}")
            OUTPUT_FILE.unlink()
        if INDEX_FILE.exists() and not args.no_index:
            INDEX_FILE.unlink()

    df_final = download_all(tickers, start_dt, end_dt)
    if df_final.empty:
        print("\n❌ Không tải được data.")
        return

    df_index = pd.DataFrame()
    if not args.no_index:
        df_index = download_vnindex(start_dt, end_dt)

    def save_parquet(df_prices: pd.DataFrame, df_index: pd.DataFrame) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)

        # Lưu file prices
        prices_path = OUTPUT_DIR / "prices.parquet"
        df_prices.to_parquet(prices_path, compression='snappy')
        print(f"   ✅ Đã lưu dữ liệu giá vào: {prices_path}")

        # Lưu file index
        if not df_index.empty:
            index_path = OUTPUT_DIR / "INDEX.parquet"
            df_index.to_parquet(index_path, compression='snappy')
            print(f"   ✅ Đã lưu VN-Index vào: {index_path}")