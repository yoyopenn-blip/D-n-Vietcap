"""
src/backend/data_loader.py  —  OPTIMIZED EDITION v3
=================================================
Tối ưu so với v2:

  [9]  Cache DataFrame thay list[dict]  → tiết kiệm ~3x RAM (_snapshot_df)
       get_snapshot_df()  → trả DataFrame trực tiếp cho screener/heatmap
       get_latest_snapshot() → backward-compat wrapper (.to_dict("records"))

  [10] load_financial_data_nocache()    → đọc quarterly không giữ trong RAM
       Quarterly (~80-100MB) chỉ load khi mở tab chi tiết, GC sau dùng.

  [11] del df_price + gc.collect()      → giải phóng 1.5M dòng giá (~50MB)
       ngay sau khi trích df_latest, không chờ GC tự dọn.

  [12] get_filter_ranges() dùng _snapshot_df trong RAM thay vì re-read Parquet.

Giữ nguyên từ v2:
  [1] Parallel sheet reading   → ThreadPoolExecutor
  [2] Snapshot Parquet cache   → Tự phát hiện source thay đổi (mtime check)
  [3] In-memory singleton      → module-level, 0ms sau lần đầu
  [4] lru_cache(maxsize=4)     → yearly BCTC (dùng thường xuyên)
  [5] Concat thay merge chuỗi → O(n) vs O(n²)
  [6] float32 dtype            → cột giá tiết kiệm ~40% RAM
  [7] BATCH yfinance download  → 200 mã/request (~50x nhanh hơn)
  [8] Auto-update guard        → chỉ chạy 1 lần mỗi phiên
"""

import os
import gc
import time
import logging
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.constants import SECTOR_TRANSLATION

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CẤU HÌNH
# ──────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DEV_MODE    = False
SHEET_LIMIT = 5

# Global toggle to allow disabling automatic yfinance updates.
# Can be overridden via environment variable `IDX_AUTO_UPDATE` (0/false to disable).
AUTO_UPDATE = False # API yfinance thường xuyên bị lỗi, tạm disable để tránh crash khi load dữ liệu. Kích hoạt lại khi cần test auto-update.

FILES = {
    "yearly":              "BCTC THEO NĂM.xlsx",
    "quarterly":           "BCTC THEO QUÝ.xlsx",
    "price":               "HISTORICAL PRICES.xlsx",
    "index":               "INDEX.xlsx",
    "parquet_financial_y": "financial_yearly.parquet",
    "parquet_financial_q": "financial_quarterly.parquet",
    "parquet_price":       "market_prices.parquet",
    "parquet_index":       "index.parquet",
    "parquet_snapshot":    "snapshot_cache.parquet",
}

_PRICE_FLOAT_COLS = ["Price Open", "Price High", "Price Low", "Price Close"]

# ──────────────────────────────────────────────
# [3] IN-MEMORY SINGLETON
# [9] Lưu dạng DataFrame thay vì list[dict]
#     → tiết kiệm ~3x RAM (float64=8B vs Python float=24B/value)
# ──────────────────────────────────────────────
_snapshot_lock: threading.Lock = threading.Lock()

_snapshot_build_lock: threading.Lock = threading.Lock()   # ← THÊM DÒNG NÀY
_snapshot_df:   pd.DataFrame   = None   # None = chưa có trong RAM
# [M2 FIX] Cache BCTC quý — tránh đọc lại từ disk mỗi lần CANSLIM chạy
_quarterly_df:  pd.DataFrame   = None
_quarterly_lock: threading.Lock = threading.Lock()

# ── Cache cho filter ranges ──
_filter_ranges_cache = None
_filter_ranges_lock  = threading.Lock()

# [FIX 4] TTL cache cho market data — tránh đọc lại parquet ~50MB mỗi callback
_MARKET_CACHE: dict = {"data": None, "ts": 0.0}

# [8] Guard chặn auto-update chạy nhiều lần trong cùng 1 phiên
_auto_update_done = False
_auto_update_lock = threading.Lock()

# Thêm gần đầu file, sau các import
_data_cutoff_date: str = ""   # "dd/mm/yyyy" — gán khi build snapshot

def get_data_cutoff_date() -> str:
    """Trả về ngày dữ liệu cuối cùng, dạng dd/mm/yyyy."""
    return _data_cutoff_date

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _mtime(path):
    try:    return os.path.getmtime(path)
    except: return 0.0


def _read_one_sheet(xls, sheet_name):
    """Đọc 1 sheet, chuẩn hoá cột. Trả None nếu lỗi."""
    try:
        df = pd.read_excel(xls, sheet_name)
        if len(df.columns) < 3:
            return None
        cols       = list(df.columns)
        cols[0]    = "Ticker"
        cols[1]    = "Date"
        df.columns = cols
        df         = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Ticker"] = df["Ticker"].astype(str)
        df = df.dropna(subset=["Ticker", "Date"])
        return df if not df.empty else None
    except Exception as e:
        logger.debug(f"   Skip sheet '{sheet_name}': {e}")
        return None


# ──────────────────────────────────────────────
# CORE PROCESSING
# ──────────────────────────────────────────────

def _process_financial_file(file_path):
    """
    Đọc BCTC Excel.
    [1] ThreadPoolExecutor đọc BS/IS/CF song song.
    [5] Concat 1 lần thay vì merge chuỗi.
    """
    if not os.path.exists(file_path):
        logger.error(f"Khong tim thay: {file_path}")
        return pd.DataFrame()

    t0 = time.perf_counter()
    logger.info(f"Doc BCTC (DEV={DEV_MODE}): {os.path.basename(file_path)}")
    xls = pd.ExcelFile(file_path)

    # Sheet COMP
    df_comp = pd.DataFrame()
    try:
        df_comp = pd.read_excel(xls, "COMP")
        df_comp = df_comp.loc[:, ~df_comp.columns.str.contains(r"^Unnamed")]
        if len(df_comp.columns) > 0:
            df_comp.rename(columns={df_comp.columns[0]: "Ticker"}, inplace=True)
            # Tìm tên cột Sector thực tế (VN data có thể dùng tên khác)
            all_cols = list(df_comp.columns)
            sector_col_actual  = next((c for c in all_cols if 'gics sector'   in c.lower()), None)
            industry_col_actual= next((c for c in all_cols if 'gics industry' in c.lower() and 'sub' not in c.lower()), None)
            sub_col_actual     = next((c for c in all_cols if 'sub-industry'  in c.lower() or 'sub industry' in c.lower()), None)
            trbc_col_actual    = next((c for c in all_cols if 'trbc'          in c.lower()), None)

            # Đổi tên về chuẩn để quant_engine tìm thấy
            rename_map = {}
            if sector_col_actual   and sector_col_actual   != "GICS Sector Name":     rename_map[sector_col_actual]   = "GICS Sector Name"
            if industry_col_actual and industry_col_actual != "GICS Industry Name":   rename_map[industry_col_actual] = "GICS Industry Name"
            if sub_col_actual      and sub_col_actual      != "GICS Sub-Industry Name": rename_map[sub_col_actual]    = "GICS Sub-Industry Name"
            if trbc_col_actual     and trbc_col_actual     != "TRBC Industry Name":   rename_map[trbc_col_actual]     = "TRBC Industry Name"
            if rename_map:
                df_comp = df_comp.rename(columns=rename_map)
                logger.info(f"   [COMP] Đổi tên cột: {rename_map}")

            keep = [c for c in [
                "Ticker", "Company Common Name",
                "GICS Sector Name", "GICS Industry Name",
                "GICS Sub-Industry Name", "TRBC Industry Name",
                "Organization Founded Year", "Date Became Public", "Auditor Details",
            ] if c in df_comp.columns]
            df_comp = df_comp[keep]
            df_comp["Ticker"] = df_comp["Ticker"].astype(str)
    except Exception as e:
        logger.debug(f"COMP error: {e}")

    # [1] Đọc song song BS/IS/CF sheets
    valid = ("BS", "IS", "CF")
    sheet_names = [s for s in xls.sheet_names if s.upper().startswith(valid)]
    if DEV_MODE:
        sheet_names = sheet_names[:SHEET_LIMIT]

    logger.info(f"   Doc {len(sheet_names)} sheets song song...")
    all_sheets = []
    with ThreadPoolExecutor(max_workers=min(10, max(1, len(sheet_names)))) as ex:
        futs = {ex.submit(_read_one_sheet, xls, s): s for s in sheet_names}
        for fut in as_completed(futs):
            df_s = fut.result()
            if df_s is not None:
                all_sheets.append(df_s)

    if not all_sheets:
        return pd.DataFrame()

    # [5] Concat + groupby first (nhanh hơn merge chain)
    combined = pd.concat(all_sheets, ignore_index=True)
    master = (
        combined
        .sort_values("Date")
        .groupby(["Ticker", "Date"], sort=False)
        .first()
        .reset_index()
    )

    if not df_comp.empty:
        master = pd.merge(master, df_comp, on="Ticker", how="left")

    logger.info(f"   BCTC xong: {len(master):,} dong | {time.perf_counter()-t0:.1f}s")
    return master


def _process_price_file(file_path):
    """
    Đọc Excel giá (1 ticker/sheet).
    [1] ThreadPoolExecutor.
    [6] float32 để tiết kiệm RAM.
    """
    if not os.path.exists(file_path):
        return pd.DataFrame()

    t0 = time.perf_counter()
    logger.info(f"Doc File Gia (DEV={DEV_MODE}): {os.path.basename(file_path)}")
    xls = pd.ExcelFile(file_path)

    sheet_names = [s for s in xls.sheet_names if s != "Sheet1"]
    if DEV_MODE:
        sheet_names = sheet_names[:SHEET_LIMIT]

    logger.info(f"   Doc {len(sheet_names)} ticker sheets song song...")
    all_dfs = []
    with ThreadPoolExecutor(max_workers=min(16, max(1, len(sheet_names)))) as ex:
        futs = {ex.submit(_read_one_sheet, xls, s): s for s in sheet_names}
        for fut in as_completed(futs):
            df_s = fut.result()
            if df_s is not None:
                all_dfs.append(df_s)

    if not all_dfs:
        return pd.DataFrame()

    final = pd.concat(all_dfs, ignore_index=True)

    # [6] Giảm bộ nhớ ~40%
    for col in _PRICE_FLOAT_COLS:
        if col in final.columns:
            try:
                final[col] = pd.to_numeric(final[col], errors="coerce").astype("float32")
            except Exception:
                pass

    logger.info(f"   Gia xong: {len(final):,} dong | {time.perf_counter()-t0:.1f}s")
    return final

def _process_index_file(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path, header=0)
        df.columns = df.columns.str.strip()

        rename_map = {}
        for col in df.columns:
            cl = col.upper()
            if 'DATE' in cl:
                rename_map[col] = 'Date'
            elif 'VNI30' in cl:
                rename_map[col] = 'VN30_Close'
            elif 'VNI100' in cl:
                rename_map[col] = 'VN100_Close'
            elif 'VNI' in cl:
                rename_map[col] = 'VNINDEX_Close'
        df = df.rename(columns=rename_map)

        df["Date"]      = pd.to_datetime(df["Date"], errors="coerce")
        df["VNINDEX_Close"] = pd.to_numeric(df["VNINDEX_Close"], errors="coerce")
        df = df.dropna(subset=["Date", "VNINDEX_Close"])
        return df.drop_duplicates(subset=["Date"]).sort_values("Date")
    except Exception as e:
        logger.error(f"Index file error: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# [7] AUTO-UPDATE GIÁ TỪ YFINANCE — BATCH MODE
# ──────────────────────────────────────────────

def _auto_update_price_from_yfinance(df_existing: pd.DataFrame, parquet_path: str) -> pd.DataFrame:
    """
    Kiểm tra ngày cuối cùng trong parquet.
    Nếu thiếu dữ liệu so với hôm nay → tự động tải bù từ yfinance rồi merge vào.

    [7] Dùng batch download (200 mã/request) thay vì loop từng mã → nhanh ~50x.
    [8] Guard biến _auto_update_done: chỉ chạy 1 lần mỗi phiên.
    """
    global _auto_update_done

    with _auto_update_lock:
        if _auto_update_done:
            logger.info("[AutoUpdate] Da chay trong phien nay. Bo qua.")
            return df_existing

    try:
        import yfinance as yf
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yfinance', '-q'])
        import yfinance as yf

    today = pd.Timestamp.today().normalize()

    if today.weekday() >= 5:
        logger.info("[AutoUpdate] Cuoi tuan - bo qua.")
        with _auto_update_lock:
            _auto_update_done = True
        return df_existing

    if df_existing.empty:
        return df_existing

    df_existing['Date'] = pd.to_datetime(df_existing['Date'])
    last_date = df_existing['Date'].max()

    yesterday = today - pd.Timedelta(days=1)
    if last_date >= yesterday:
        logger.info(f"[AutoUpdate] Du lieu da cap nhat den {last_date.date()}. Bo qua.")
        with _auto_update_lock:
            _auto_update_done = True
        return df_existing

    start_str = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end_str   = (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    logger.info(f"[AutoUpdate] Thieu du lieu tu {start_str} den {today.date()}. Bat dau tai...")

    tickers    = df_existing['Ticker'].unique().tolist()
    jk_tickers = list(tickers)

    BATCH_SIZE = 200
    new_rows   = []

    for batch_start in range(0, len(jk_tickers), BATCH_SIZE):
        batch     = jk_tickers[batch_start: batch_start + BATCH_SIZE]
        batch_str = " ".join(batch)
        try:
            df_batch = yf.download(
                batch_str,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if df_batch.empty:
                continue

            if not isinstance(df_batch.columns, pd.MultiIndex):
                ticker_solo = batch[0]
                df_batch.columns = pd.MultiIndex.from_tuples(
                    [(ticker_solo, str(c)) for c in df_batch.columns]
                )

            for jk_ticker in batch:
                try:
                    level0 = df_batch.columns.get_level_values(0)
                    if jk_ticker not in level0:
                        continue

                    df_t = df_batch[jk_ticker].copy()
                    df_t = df_t.dropna(how="all")
                    if df_t.empty:
                        continue

                    df_t.columns = [str(c).strip().title() for c in df_t.columns]

                    needed  = ['Close', 'Open', 'High', 'Low', 'Volume']
                    missing = [c for c in needed if c not in df_t.columns]
                    if missing:
                        continue

                    df_t = df_t[needed].copy()
                    for col in ['Close', 'Open', 'High', 'Low']:
                        df_t[col] = pd.to_numeric(df_t[col], errors='coerce').round(0)
                    df_t['Volume'] = pd.to_numeric(df_t['Volume'], errors='coerce')

                    df_t.columns = ['Price Close', 'Price Open', 'Price High', 'Price Low', 'Volume']
                    df_t['Turnover'] = df_t['Volume'] * df_t['Price Close']

                    df_t.reset_index(inplace=True)
                    df_t['Date'] = pd.to_datetime(df_t['Date']).dt.tz_localize(None).dt.normalize()
                    df_t.insert(0, 'Ticker', jk_ticker)   # giữ nguyên, đã có đuôi sàn
                    new_rows.append(df_t)

                except Exception as e:
                    logger.warning(f"[AutoUpdate] Loi xu ly {jk_ticker}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"[AutoUpdate] Loi batch {batch_start}-{batch_start + BATCH_SIZE}: {e}")
            continue

    with _auto_update_lock:
        _auto_update_done = True

    if not new_rows:
        logger.info("[AutoUpdate] Khong co du lieu moi nao tu yfinance.")
        return df_existing

    df_new      = pd.concat(new_rows, ignore_index=True)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
    df_combined = df_combined.sort_values(['Ticker', 'Date'])

    df_combined.to_parquet(parquet_path, index=False)
    added = len(df_combined) - len(df_existing)
    logger.info(f"[AutoUpdate] Hoan tat! +{added:,} dong moi. Tong: {len(df_combined):,} dong.")
    return df_combined


# ──────────────────────────────────────────────
# PUBLIC LOADERS
# ──────────────────────────────────────────────

def _strip_exchange_suffix(df: pd.DataFrame) -> pd.DataFrame:
    """Strip .HM/.HN/.HNO khỏi cột Ticker và tạo cột Exchange nếu chưa có."""
    if 'Ticker' in df.columns:
        df = df.copy()
        # Bổ sung logic: Lưu thông tin sàn vào cột Exchange trước khi cắt đuôi
        if 'Exchange' not in df.columns:
            df['Exchange'] = ''
            df.loc[df['Ticker'].str.endswith('.HNO', na=False), 'Exchange'] = 'UPCOM'
            df.loc[df['Ticker'].str.endswith('.HN', na=False), 'Exchange'] = 'HNX'
            df.loc[df['Ticker'].str.endswith('.HM', na=False), 'Exchange'] = 'HOSE'

        df['Ticker'] = df['Ticker'].str.replace(r'\.(HNO|HN|HM)$', '', regex=True)
    return df

def load_market_data():
    # [FIX 4] TTL cache 5 phút — đủ cho phiên dùng liên tục
    now = time.time()
    if _MARKET_CACHE["data"] is not None and now - _MARKET_CACHE["ts"] < 300:
        logger.info(f"[MarketCache] HIT — trả từ RAM ({len(_MARKET_CACHE['data']):,} dòng)")
        return _MARKET_CACHE["data"]

    parquet = os.path.join(PROCESSED_DIR, FILES["parquet_price"])
    if os.path.exists(parquet) and not DEV_MODE:
        t0 = time.perf_counter()
        df = pd.read_parquet(parquet)
        logger.info(f"Gia from Parquet: {len(df):,} dong | {time.perf_counter()-t0:.2f}s")
        if AUTO_UPDATE:
            df = _auto_update_price_from_yfinance(df, parquet)
        else:
            logger.info("[AutoUpdate] Disabled by config. Skipping yfinance update.")
        # ── Strip đuôi sàn để Ticker khớp với snapshot ──
        df = _strip_exchange_suffix(df)
        if 'Exchange' in df.columns:
            exch_sample = df[['Ticker','Exchange']].drop_duplicates('Ticker').head(5)
            logger.info(f"[DEBUG load_market] Exchange sample:\n{exch_sample.to_string()}")
            logger.info(f"[DEBUG load_market] Exchange counts: {df['Exchange'].value_counts().to_dict()}")
        else:
            logger.warning("[DEBUG load_market] Cột Exchange KHÔNG tồn tại sau strip!")
        # Đảm bảo Ticker là string (tránh lỗi .str accessor)
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].astype(str)
        _MARKET_CACHE["data"] = df
        _MARKET_CACHE["ts"]   = time.time()
        return df

    df = _process_price_file(os.path.join(RAW_DIR, FILES["price"]))
    # Lưu parquet TRƯỚC khi strip — giữ đuôi .HM/.HN/.HNO trong file gốc
    # để _build_snapshot_df đọc lại được thông tin sàn
    if not df.empty and not DEV_MODE:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_parquet(parquet, index=False)
        logger.info("Saved Parquet gia (có đuôi sàn)")
    df = _strip_exchange_suffix(df)
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].astype(str)
    return df

@lru_cache(maxsize=4)
def load_financial_data(report_type="yearly"):
    """
    Load BCTC với lru_cache — dùng cho yearly (cần thường xuyên trong snapshot).
    Quarterly nên dùng load_financial_data_nocache() để tránh giữ ~100MB mãi trong RAM.
    """
    is_yearly = report_type == "yearly"
    parq_key  = "parquet_financial_y" if is_yearly else "parquet_financial_q"
    parquet   = os.path.join(PROCESSED_DIR, FILES[parq_key])

    if os.path.exists(parquet):
        t0 = time.perf_counter()
        df = pd.read_parquet(parquet)
        # DEBUG ngành
        if "GICS Sector Name" in df.columns:
            raw_sectors = df["GICS Sector Name"].dropna().unique()
            logger.info(f"[DEBUG sectors] Giá trị thô trong BCTC: {list(raw_sectors[:15])}")
        df = _strip_exchange_suffix(df)
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].astype(str)
        if "GICS Sector Name" in df.columns:
            df["GICS Sector Name"] = (
                df["GICS Sector Name"].map(SECTOR_TRANSLATION)
                .fillna(df["GICS Sector Name"])
            )
        logger.info(f"BCTC {report_type} from Parquet: {len(df):,} dong | {time.perf_counter()-t0:.2f}s")
        return df

    if not is_yearly and not DEV_MODE:
        logger.warning("[SKIP] Quarterly Excel qua nang. Can file Parquet.")
        return pd.DataFrame()

    excel_key = "yearly" if is_yearly else "quarterly"
    df = _process_financial_file(os.path.join(RAW_DIR, FILES[excel_key]))
    if not df.empty:
        if "GICS Sector Name" in df.columns:
            df["GICS Sector Name"] = (
                df["GICS Sector Name"].map(SECTOR_TRANSLATION)
                .fillna(df["GICS Sector Name"])
            )
        if not DEV_MODE:
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            df.to_parquet(parquet, index=False)
            logger.info(f"Saved Parquet BCTC {report_type}")
    return df

def load_financial_data_nocache(report_type: str = "quarterly") -> pd.DataFrame:
    """
    [M2 FIX] BCTC quý được cache vào RAM lần đầu, các lần sau trả về từ cache.
    Tránh đọc lại ~2GB parquet mỗi lần user chọn CANSLIM trên free-tier HF.
    """
    global _quarterly_df

    # Chỉ cache quarterly — yearly đã có lru_cache riêng
    if report_type == "yearly":
        return load_financial_data("yearly")

    with _quarterly_lock:
        if _quarterly_df is not None:
            logger.info(f"BCTC quarterly from RAM cache: {len(_quarterly_df):,} dòng")
            return _quarterly_df

    parquet = os.path.join(PROCESSED_DIR, FILES["parquet_financial_q"])
    if not os.path.exists(parquet):
        logger.warning(f"[nocache] Không tìm thấy {parquet}")
        return pd.DataFrame()

    try:
        t0 = time.perf_counter()
        df = pd.read_parquet(parquet)
        if "GICS Sector Name" in df.columns:
            df["GICS Sector Name"] = (
                df["GICS Sector Name"].map(SECTOR_TRANSLATION)
                .fillna(df["GICS Sector Name"])
            )
        df = _strip_exchange_suffix(df)
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].astype(str)

        with _quarterly_lock:
            _quarterly_df = df

        logger.info(f"BCTC quarterly loaded & cached: {len(df):,} dòng | {time.perf_counter()-t0:.2f}s")
        return df
    except Exception as e:
        logger.error(f"Lỗi đọc BCTC quarterly: {e}")
        return pd.DataFrame()

def load_index_data():
    parquet = os.path.join(PROCESSED_DIR, FILES["parquet_index"])
    if os.path.exists(parquet):
        t0 = time.perf_counter()
        df = pd.read_parquet(parquet)
        logger.info(f"Index from Parquet: {len(df):,} dong | {time.perf_counter()-t0:.2f}s")
        if AUTO_UPDATE:
            df = _auto_update_index_from_yfinance(df, parquet)
        else:
            logger.info("[AutoUpdate] Disabled by config. Skipping index yfinance update.")
        return df

    df = _process_index_file(os.path.join(RAW_DIR, FILES["index"]))
    if not df.empty and not DEV_MODE:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_parquet(parquet, index=False)
        logger.info("Saved Parquet index")
    return df


def _auto_update_index_from_yfinance(df_existing: pd.DataFrame, parquet_path: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        return df_existing

    today = pd.Timestamp.today().normalize()
    if today.weekday() >= 5:
        return df_existing

    df_existing['Date'] = pd.to_datetime(df_existing['Date'])
    last_date = df_existing['Date'].max()
    yesterday = today - pd.Timedelta(days=1)

    if last_date >= yesterday:
        logger.info(f"[IndexUpdate] Index đã cập nhật đến {last_date.date()}. Bỏ qua.")
        return df_existing

    start_str = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end_str   = (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        df_new = yf.download("^VNINDEX", start=start_str, end=end_str,
                             auto_adjust=True, progress=False)
        if df_new.empty:
            logger.info("[IndexUpdate] Không có dữ liệu mới.")
            return df_existing

        df_new = df_new.reset_index()
        df_new.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df_new.columns]
        df_new['Date'] = pd.to_datetime(df_new['Date']).dt.tz_localize(None).dt.normalize()

        col_map = {
            'Close': 'VNINDEX_Close', 'Open': 'VNINDEX_Open',
            'High': 'VNINDEX_High',   'Low': 'VNINDEX_Low',
            'Volume': 'VNINDEX_Volume',
        }
        df_new = df_new.rename(columns=col_map)
        keep = ['Date'] + [v for v in col_map.values() if v in df_new.columns]
        df_new = df_new[keep].dropna(subset=['VNINDEX_Close'])

        for col in ['VNINDEX_Open', 'VNINDEX_High', 'VNINDEX_Low', 'VNINDEX_Volume']:
            if col not in df_existing.columns:
                df_existing[col] = np.nan

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['Date'], keep='last')
        df_combined = df_combined.sort_values('Date').reset_index(drop=True)

        df_combined.to_parquet(parquet_path, index=False)
        added = len(df_combined) - len(df_existing)
        logger.info(f"[IndexUpdate] Xong! +{added} ngày mới. Tổng: {len(df_combined)} ngày.")
        return df_combined

    except Exception as e:
        logger.warning(f"[IndexUpdate] Lỗi: {e}")
        return df_existing


# ──────────────────────────────────────────────
# SNAPSHOT — DISK CACHE [2] + IN-MEMORY [3][9]
# ──────────────────────────────────────────────

def _snapshot_stale() -> bool:
    snap    = os.path.join(PROCESSED_DIR, FILES["parquet_snapshot"])
    snap_mt = _mtime(snap)
    if snap_mt == 0:
        logger.info("[STALE] snapshot không tồn tại → rebuild")
        return True
    sources = [
        os.path.join(PROCESSED_DIR, FILES["parquet_price"]),
        os.path.join(PROCESSED_DIR, FILES["parquet_financial_y"]),
        os.path.join(RAW_DIR, FILES["price"]),
        os.path.join(RAW_DIR, FILES["yearly"]),
        os.path.join(BASE_DIR, "src", "backend", "quant_engine.py"),
        os.path.join(BASE_DIR, "src", "backend", "technical_indicators.py"),
        os.path.join(BASE_DIR, "src", "backend", "data_loader.py"),  # ← rebuild khi sửa data_loader
    ]
    for src in sources:
        if _mtime(src) > snap_mt:
            logger.info(f"[STALE] '{os.path.basename(src)}' mới hơn snapshot → rebuild")
            return True
    logger.info(f"[FRESH] Snapshot còn mới")
    return False


def _build_snapshot_df() -> pd.DataFrame:
    """
    Hàm nội bộ: load giá → quant engine → trả DataFrame snapshot.
    """
    df_price = None
    try:
        df_price = load_market_data()
    except Exception as e:
        logger.error(f"Loi load gia: {e}")
        return pd.DataFrame()

    if df_price.empty:
        return pd.DataFrame()

    # Build exchange map trực tiếp từ df_price (đã được tạo cột Exchange ở hàm strip)
    # FIX: Đọc đuôi sàn từ file parquet GỐC (vì Ticker trong df_price đã bị strip đuôi
    # nhưng cột Exchange có thể rỗng nếu snapshot cũ được load từ disk cache).
    _exchange_map = {}
    try:
        _raw_price_path = os.path.join(PROCESSED_DIR, FILES["parquet_price"])
        if os.path.exists(_raw_price_path):
            # Đọc cột Ticker thô từ parquet (có đuôi .HM/.HN/.HNO)
            _ticker_raw = pd.read_parquet(_raw_price_path, columns=["Ticker"])
            _ticker_raw["_Exchange"] = ""
            _ticker_raw.loc[_ticker_raw["Ticker"].str.endswith(".HNO", na=False), "_Exchange"] = "UPCOM"
            _ticker_raw.loc[_ticker_raw["Ticker"].str.endswith(".HN",  na=False), "_Exchange"] = "HNX"
            _ticker_raw.loc[_ticker_raw["Ticker"].str.endswith(".HM",  na=False), "_Exchange"] = "HOSE"
            _ticker_raw["Ticker"] = _ticker_raw["Ticker"].str.replace(r"\.(HNO|HN|HM)$", "", regex=True)
            _ticker_raw = _ticker_raw[_ticker_raw["_Exchange"] != ""]
            _exchange_map = (
                _ticker_raw.drop_duplicates("Ticker")
                .set_index("Ticker")["_Exchange"]
                .to_dict()
            )
            logger.info(f"[Exchange] Đọc từ parquet gốc: {len(_exchange_map)} ticker có sàn | sample: {dict(list(_exchange_map.items())[:3])}")
    except Exception as _ex:
        logger.warning(f"[Exchange] Lỗi đọc từ parquet gốc, fallback về df_price: {_ex}")

    # Fallback: dùng cột Exchange đã có trong df_price
    if not _exchange_map and 'Exchange' in df_price.columns:
        _exchange_map = (
            df_price[df_price['Exchange'].astype(str).str.strip() != '']
            .drop_duplicates(subset=["Ticker"])
            .set_index("Ticker")["Exchange"]
            .to_dict()
        )
        logger.info(f"[Exchange] Fallback từ df_price: {len(_exchange_map)} ticker")

    try:

        global _data_cutoff_date
        try:
            _data_cutoff_date = df_price["Date"].max().strftime("%d/%m/%Y")
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Loi cat ngay: {e}")

    try:
        df_price = df_price.sort_values(["Ticker", "Date"])
        df_price["Avg_Vol_20D"] = (
            df_price.groupby("Ticker", sort=False)["Volume"]
            .transform(lambda x: x.rolling(20, min_periods=1).mean())
            .round(0).fillna(0)
        )
        # ADV20 theo giá trị VND (cần cho Liquidity Constraint của Markowitz)
        df_price["Avg_Vol_20D_VND"] = (
            df_price["Avg_Vol_20D"] * df_price["Price Close"]
        ).round(0).fillna(0)
    except Exception as e:
        logger.warning(f"Loi AvgVol: {e}")

    df_latest = df_price.drop_duplicates(subset=["Ticker"], keep="last").copy()

    # EPS Growth QoQ proxy — dùng YoY đã có nếu QoQ chưa tính riêng
    # Khi quant_engine tính xong sẽ override nếu có quarterly data
    if "EPS Growth YoY (%)" in df_latest.columns:
        df_latest["EPS_Growth_QoQ"] = df_latest["EPS Growth YoY (%)"]
    else:
        df_latest["EPS_Growth_QoQ"] = float("nan")

    del df_price
    gc.collect()

    df_fin = load_financial_data("yearly")

    try:
        from src.backend.quant_engine import calculate_all_scores
        df_final = calculate_all_scores(df_latest, df_fin)
    except Exception as e:
        logger.error(f"Loi quant engine: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

    del df_latest
    gc.collect()

    if df_final.empty:
        return pd.DataFrame()

    for _sec_col in ('Sector', 'GICS Sector Name', 'GICS Industry Name', 'GICS Sub-Industry Name'):
        if _sec_col in df_final.columns:
            df_final[_sec_col] = df_final[_sec_col].astype(str).str.strip()
            df_final.loc[df_final[_sec_col].isin(['nan', 'None', '', '0']), _sec_col] = 'Chưa phân loại'

    # Gán Exchange vào snapshot cuối
    if 'Ticker' in df_final.columns:
        df_final['Ticker'] = df_final['Ticker'].astype(str)
        if _exchange_map:
            df_final['Exchange'] = df_final['Ticker'].map(_exchange_map).fillna('')
            exch_counts = df_final['Exchange'].value_counts().to_dict()
            logger.info(f"[Exchange] Distribution: {exch_counts}")
        else:
            df_final['Exchange'] = ''

    return df_final

def get_snapshot_df() -> pd.DataFrame:
    """
    [9] Trả về snapshot dạng DataFrame — cache trực tiếp, tiết kiệm ~3x RAM.
    [C1 FIX] Double-checked locking: _snapshot_build_lock serialise rebuild,
             tránh 2 thread cùng chạy _build_snapshot_df() song song.
    """
    global _snapshot_df

    snap_path = os.path.join(PROCESSED_DIR, FILES["parquet_snapshot"])

    # ── FAST PATH: RAM cache ─────────────────────────────────────────────
    with _snapshot_lock:
        if _snapshot_df is not None:
            if not os.path.exists(snap_path):
                logger.info("Parquet bị xóa → clear RAM cache để rebuild")
                _snapshot_df = None
            else:
                logger.info(f"Snapshot from memory (DataFrame): {len(_snapshot_df)} ma")
                return _snapshot_df

    # ── DISK CACHE (không cần build lock vì chỉ đọc) ────────────────────
    if not _snapshot_stale():
        try:
            t0 = time.perf_counter()
            df = pd.read_parquet(snap_path)

            if 'Exchange' not in df.columns or df['Exchange'].astype(str).str.strip().eq('').all():
                logger.warning("[Exchange] Snapshot thiếu/rỗng cột Exchange → xóa cache để rebuild!")
                try:
                    os.remove(snap_path)
                except OSError:
                    pass
                raise ValueError("Exchange column missing or all-empty — forcing rebuild")

            with _snapshot_lock:
                _snapshot_df = df
            exch_dist = df['Exchange'].value_counts().to_dict()
            logger.info(f"Snapshot from Parquet: {len(df)} ma | Exchange={exch_dist} | {time.perf_counter()-t0:.2f}s")
            return df
        except Exception as e:
            logger.warning(f"Khong doc duoc snapshot Parquet: {e}")

    # ── REBUILD — chỉ 1 thread được chạy, các thread khác CHỜ ───────────
    logger.info("Chờ _snapshot_build_lock để rebuild snapshot...")
    with _snapshot_build_lock:
        # ── Double-check sau khi lấy được build lock ─────────────────────
        # Thread khác có thể đã rebuild xong trong lúc thread này chờ lock
        with _snapshot_lock:
            if _snapshot_df is not None:
                logger.info(f"[C1 FIX] Snapshot đã được thread khác build xong: {len(_snapshot_df)} ma")
                return _snapshot_df

        # Kiểm tra lại disk cache (thread khác có thể đã lưu parquet)
        if not _snapshot_stale():
            try:
                df = pd.read_parquet(snap_path)
                if 'Exchange' in df.columns and not df['Exchange'].astype(str).str.strip().eq('').all():
                    with _snapshot_lock:
                        _snapshot_df = df
                    logger.info(f"[C1 FIX] Snapshot load từ Parquet sau khi chờ: {len(df)} ma")
                    return df
            except Exception:
                pass

        # ── Thực sự rebuild ──────────────────────────────────────────────
        logger.info("Tinh lai Snapshot (full pipeline)...")
        t0 = time.perf_counter()

        df_final = _build_snapshot_df()
        if df_final.empty:
            return pd.DataFrame()

        # Lưu Parquet
        try:
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            df_save = df_final.copy()
            for col in df_save.select_dtypes(include=['object']).columns:
                df_save[col] = df_save[col].astype(str)
            df_save.to_parquet(snap_path, index=False)
            del df_save
            logger.info(f"Saved snapshot ({len(df_final)} ma)")
        except Exception as e:
            logger.warning(f"Khong luu duoc snapshot: {e}")

        with _snapshot_lock:
            _snapshot_df = df_final

        # Reset filter ranges cache
        global _filter_ranges_cache
        with _filter_ranges_lock:
            _filter_ranges_cache = None

        if not df_final.empty:
            perf_cols = sorted(c for c in df_final.columns if c.startswith("Perf_"))
            logger.info(f"[DEBUG] Perf columns: {perf_cols}")
            logger.info(f"[DEBUG] Total columns: {len(df_final.columns)}")

        logger.info(f"Snapshot xong: {len(df_final)} ma | {time.perf_counter()-t0:.1f}s")
        return df_final

def get_latest_snapshot(df_price=None) -> list:
    """
    Backward-compatible wrapper — trả list[dict] như mọi code cũ kỳ vọng.
    Ưu tiên dùng get_snapshot_df() trực tiếp ở các callback cần DataFrame.
    """
    df = get_snapshot_df()
    if df is None or df.empty:
        return []
    return df.to_dict("records")

def invalidate_snapshot_cache():
    """Gọi sau khi thay file dữ liệu mới để buộc tái tính."""
    global _snapshot_df, _filter_ranges_cache, _auto_update_done
    with _snapshot_lock:
        _snapshot_df = None
    with _quarterly_lock:                    # ← thêm block này
        _quarterly_df = None
    with _filter_ranges_lock:
        _filter_ranges_cache = None
    with _auto_update_lock:
        _auto_update_done = False
    snap = os.path.join(PROCESSED_DIR, FILES["parquet_snapshot"])
    try:
        if os.path.exists(snap):
            os.remove(snap)
            logger.info("Da xoa snapshot cache")
    except OSError as e:
        logger.warning(f"{e}")


# ──────────────────────────────────────────────
# FILTER RANGES — tính min/max thực tế từ snapshot
# ──────────────────────────────────────────────

FILTER_COL_MAP = {
    "filter-price":           "Price Close",
    "filter-volume":          "Volume",
    "filter-market-cap":      "Market Cap",
    "filter-eps":             "EPS",
    "filter-perf-1w":         "Perf_1W",
    "filter-perf-1m":         "Perf_1M",
    "filter-pe":              "P/E",
    "filter-pb":              "P/B",
    "filter-ps":              "P/S",
    "filter-ev-ebitda":       "EV/EBITDA",
    "filter-div-yield":       "Dividend Yield (%)",
    "filter-roe":             "ROE (%)",
    "filter-roa":             "ROA (%)",
    "filter-gross-margin":    "Gross Margin (%)",
    "filter-net-margin":      "Net Margin (%)",
    "filter-ebit-margin":     "EBIT Margin (%)",
    "filter-rev-growth-yoy":  "Revenue Growth YoY (%)",
    "filter-rev-cagr-5y":     "Revenue CAGR 5Y (%)",
    "filter-eps-growth-yoy":  "EPS Growth YoY (%)",
    "filter-eps-cagr-5y":     "EPS CAGR 5Y (%)",
    "filter-de":              "D/E",
    "filter-current-ratio":   "Current Ratio",
    "filter-net-cash-cap":    "Net Cash / Market Cap (%)",
    "filter-net-cash-assets": "Net Cash / Assets (%)",
    "filter-price-vs-sma5":   "Price_vs_SMA5",
    "filter-price-vs-sma10":  "Price_vs_SMA10",
    "filter-price-vs-sma20":  "Price_vs_SMA20",
    "filter-price-vs-sma50":  "Price_vs_SMA50",
    "filter-price-vs-sma100": "Price_vs_SMA100",
    "filter-price-vs-sma200": "Price_vs_SMA200",
    "filter-pct-from-high-1y":  "Pct_From_High_1Y",
    "filter-pct-from-low-1y":   "Pct_From_Low_1Y",
    "filter-pct-from-high-all": "Pct_From_High_All",
    "filter-pct-from-low-all":  "Pct_From_Low_All",
    "filter-rsi14":       "RSI_14",
    "filter-macd-hist":   "MACD_Histogram",
    "filter-bb-width":    "BB_Width",
    "filter-consec-up":   "Consec_Up",
    "filter-consec-down": "Consec_Down",
    "filter-beta":    "Beta",
    "filter-alpha":   "Alpha",
    "filter-rs-3d":   "RS_3D",
    "filter-rs-1m":   "RS_1M",
    "filter-rs-3m":   "RS_3M",
    "filter-rs-1y":   "RS_1Y",
    "filter-rs-avg":  "RS_Avg",
    "filter-vol-vs-sma5":  "Vol_vs_SMA5",
    "filter-vol-vs-sma10": "Vol_vs_SMA10",
    "filter-vol-vs-sma20": "Vol_vs_SMA20",
    "filter-vol-vs-sma50": "Vol_vs_SMA50",
    "filter-avg-vol-5d":   "Avg_Vol_5D",
    "filter-avg-vol-10d":  "Avg_Vol_10D",
    "filter-avg-vol-50d":  "Avg_Vol_50D",
    "filter-canslim": "CANSLIM Score",
}

_FALLBACK_RANGES = {
    "filter-price":           [0, 50000],
    "filter-volume":          [0, 10000000],
    "filter-market-cap":      [0, 999000000000000],
    "filter-eps":             [-1000, 10000],
    "filter-perf-1w":         [-30, 30],
    "filter-perf-1m":         [-50, 100],
    "filter-pe":              [0, 100],
    "filter-pb":              [0, 20],
    "filter-ps":              [0, 20],
    "filter-ev-ebitda":       [0, 50],
    "filter-div-yield":       [0, 25],
    "filter-roe":             [-50, 100],
    "filter-roa":             [-30, 50],
    "filter-gross-margin":    [-50, 100],
    "filter-net-margin":      [-50, 50],
    "filter-ebit-margin":     [-50, 50],
    "filter-rev-growth-yoy":  [-50, 300],
    "filter-rev-cagr-5y":     [-20, 100],
    "filter-eps-growth-yoy":  [-100, 500],
    "filter-eps-cagr-5y":     [-20, 100],
    "filter-de":              [0, 15],
    "filter-current-ratio":   [0, 10],
    "filter-net-cash-cap":    [-100, 100],
    "filter-net-cash-assets": [-100, 100],
    "filter-price-vs-sma5":   [-30, 50],
    "filter-price-vs-sma10":  [-30, 50],
    "filter-price-vs-sma20":  [-30, 50],
    "filter-price-vs-sma50":  [-50, 100],
    "filter-price-vs-sma100": [-50, 100],
    "filter-price-vs-sma200": [-50, 100],
    "filter-pct-from-high-1y":  [-80, 10],
    "filter-pct-from-low-1y":   [-10, 300],
    "filter-pct-from-high-all": [-90, 10],
    "filter-pct-from-low-all":  [-10, 500],
    "filter-rsi14":       [0, 100],
    "filter-macd-hist":   [-1000, 1000],
    "filter-bb-width":    [0, 50],
    "filter-consec-up":   [0, 20],
    "filter-consec-down": [0, 20],
    "filter-beta":    [-2, 4],
    "filter-alpha":   [-50, 100],
    "filter-rs-3d":   [-20, 20],
    "filter-rs-1m":   [-30, 50],
    "filter-rs-3m":   [-50, 100],
    "filter-rs-1y":   [-80, 200],
    "filter-rs-avg":  [-50, 100],
    "filter-vol-vs-sma5":  [0, 10],
    "filter-vol-vs-sma10": [0, 10],
    "filter-vol-vs-sma20": [0, 10],
    "filter-vol-vs-sma50": [0, 10],
    "filter-avg-vol-5d":   [0, 100000000],
    "filter-avg-vol-10d":  [0, 100000000],
    "filter-avg-vol-50d":  [0, 100000000],
    "filter-canslim": [0, 6],
    "filter-gtgd-1w":  [0, 100000000000],
    "filter-gtgd-10d": [0, 200000000000],
    "filter-gtgd-1m":  [0, 500000000000],
}


def _round_nice(val, is_min: bool) -> float:
    import math
    if val == 0:
        return 0.0
    abs_val = abs(val)
    if abs_val >= 1_000_000_000_000:   magnitude = 100_000_000_000
    elif abs_val >= 1_000_000_000:     magnitude = 1_000_000_000
    elif abs_val >= 1_000_000:         magnitude = 100_000
    elif abs_val >= 100_000:           magnitude = 10_000
    elif abs_val >= 10_000:            magnitude = 1_000
    elif abs_val >= 1_000:             magnitude = 100
    elif abs_val >= 100:               magnitude = 10
    elif abs_val >= 10:                magnitude = 1
    elif abs_val >= 1:                 magnitude = 0.5
    else:                              magnitude = 0.1
    if is_min:
        return round(math.floor(val / magnitude) * magnitude, 6)
    else:
        return round(math.ceil(val / magnitude) * magnitude, 6)


def get_filter_ranges() -> dict:
    """
    [12] Ưu tiên dùng _snapshot_df đang trong RAM thay vì re-read Parquet.
    [FIX] Dùng percentile clipping thay vì min/max thô để loại bỏ outlier
    cực đoan của thị trường IDX (nhiều mã penny/shell company làm lệch range).

    Chiến lược theo nhóm (dựa trên phân tích phân phối thực tế):
    - CLIP_P2_P98 : cột có outlier cực mạnh (ratio curr_max/p98 > 5×)
    - CLIP_P1_P99 : cột có outlier vừa (ratio 2–5×)
    - OK_AS_IS    : cột phân phối tương đối đẹp (ratio < 2×)

    Một số cột giữ hard-coded max có ý nghĩa (RSI: 100, pct_from_high: 0...)
    User vẫn nhập tay vào ô số để vượt ra ngoài range slider bất cứ lúc nào.
    """
    global _filter_ranges_cache

    with _filter_ranges_lock:
        if _filter_ranges_cache is not None:
            return _filter_ranges_cache

    ranges = dict(_FALLBACK_RANGES)

    # ── Phân loại percentile clipping theo kết quả debug_ranges ─────────────
    # CLIP_P2_P98: outlier cực mạnh — giữ 96% giữa
    _CLIP_P2_P98 = {
        "filter-eps",
        "filter-perf-1w",
        "filter-perf-1m",
        "filter-pb",
        "filter-net-margin",
        "filter-rev-growth-yoy",
        "filter-eps-growth-yoy",
        "filter-macd-hist",
        "filter-alpha",
        "filter-vol-vs-sma50",
        "filter-price-vs-sma5",
        "filter-price-vs-sma10",
        "filter-price-vs-sma20",
        "filter-price-vs-sma50",
        "filter-price-vs-sma100",
        "filter-price-vs-sma200",
    }

    # CLIP_P1_P99: outlier vừa — giữ 98% giữa
    _CLIP_P1_P99 = {
        "filter-pe",
        "filter-div-yield",
        "filter-roe",
        "filter-roa",
        "filter-ebit-margin",
        "filter-rev-cagr-5y",
        "filter-de",
        "filter-bb-width",
        "filter-consec-up",
        "filter-rs-1m",
        "filter-rs-3m",
        "filter-rs-1y",
        "filter-rs-avg",
        "filter-vol-vs-sma10",
        "filter-vol-vs-sma20",
        "filter-net-cash-assets",
        "filter-beta",    # ← thêm
        "filter-rs-3d",   # ← thêm
    }

    # CLIP_P5_P95: cột phân phối lệch rất nặng, cần clip mạnh hơn
    # (P/S, current ratio, pct_from_low có outlier >10× ngay cả ở p98)
    _CLIP_P5_P95 = {
        "filter-ps",
        "filter-current-ratio",
        "filter-pct-from-low-1y",
        "filter-pct-from-low-all",
        "filter-gross-margin",    # min side: -54000, p5 = 0.02
    }

    # Cột có hard-coded bounds có nghĩa kinh doanh rõ ràng
    _HARD_BOUNDS = {
        "filter-rsi14":            (0.0,  100.0),
        "filter-pct-from-high-1y": (-100.0, 0.0),
        "filter-pct-from-high-all":(-100.0, 0.0),
        "filter-canslim":          (0.0,   5.0),
        "filter-consec-down":      (0.0,  10.0),
        "filter-vol-vs-sma5":      (0.0,   5.0),
        # XÓA 2 DÒNG NÀY nếu muốn đọc từ data thực tế:
        #"filter-beta":             (-1.0,  3.0),
        #"filter-rs-3d":            (-25.0, 25.0),
    }

    try:
        # Gọi hàm get_snapshot_df() để tự động quản lý: RAM -> Disk -> Rebuild
        df = get_snapshot_df()

        # Chỉ dùng fallback khi hệ thống thực sự lỗi, không thể build nổi dữ liệu
        if df is None or df.empty:
            logger.error("CRITICAL: Không thể tạo hoặc đọc snapshot_cache → Hệ thống rỗng, dùng fallback ranges")
            with _filter_ranges_lock:
                _filter_ranges_cache = ranges
            return ranges

        logger.info(f"Tính filter ranges từ snapshot: {len(df)} mã, {len(df.columns)} cột")

        for filter_id, col_name in FILTER_COL_MAP.items():
            if col_name not in df.columns:
                continue
            series = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if len(series) < 2:
                continue

            # Hard-coded bounds — ưu tiên cao nhất
            if filter_id in _HARD_BOUNDS:
                min_v, max_v = _HARD_BOUNDS[filter_id]
                ranges[filter_id] = [min_v, max_v]
                logger.info(f"  Range {filter_id}: [{min_v}, {max_v}]  (hard-coded)")
                continue

            # Chọn percentile clipping
            if filter_id in _CLIP_P2_P98:
                lo_pct, hi_pct = 2, 98
            elif filter_id in _CLIP_P1_P99:
                lo_pct, hi_pct = 1, 99
            elif filter_id in _CLIP_P5_P95:
                lo_pct, hi_pct = 5, 95
            else:
                # OK_AS_IS: dùng p0.5/p99.5 — vẫn cắt 0.5% hai đầu
                # để loại mã shell/lỗi dữ liệu cực hiếm
                lo_pct, hi_pct = 1, 99

            raw_min = float(np.percentile(series, lo_pct))
            raw_max = float(np.percentile(series, hi_pct))

            # Đảm bảo min < max
            if raw_min >= raw_max:
                raw_min = float(series.min())
                raw_max = float(series.max())

            min_v = _round_nice(raw_min, is_min=True)
            max_v = _round_nice(raw_max, is_min=False)

            if min_v >= max_v:
                min_v, max_v = raw_min, raw_max

            if min_v < max_v:
                ranges[filter_id] = [min_v, max_v]
                logger.info(
                    f"  Range {filter_id}: [{min_v}, {max_v}]  "
                    f"(p{lo_pct}/p{hi_pct}, cột '{col_name}')"
                )

    except Exception as e:
        logger.error(f"Lỗi tính filter ranges: {e}")
        import traceback; traceback.print_exc()

    with _filter_ranges_lock:
        _filter_ranges_cache = ranges

    return ranges

def get_ticker_list() -> list:
    records = get_latest_snapshot()
    if not records:
        return []

    # 1. Load bản đồ tên tiếng Việt từ file CSV
    vn_names = {}
    try:
        import os, pandas as pd
        # Đường dẫn tới file COMP INFO.csv
        csv_path = os.path.join(BASE_DIR, "data", "raw", "COMP INFO.csv")
        if os.path.exists(csv_path):
            df_info = pd.read_csv(csv_path)
            # Ánh xạ Ticker -> Tên tiếng Việt
            for _, r in df_info.iterrows():
                symbol = str(r.get('symbol', r.get('Ticker', ''))).replace('.HM','').replace('.HN','').strip()
                name_vi = str(r.get('company_name_vi', r.get('organ_name', ''))).strip()
                if symbol and name_vi:
                    vn_names[symbol] = name_vi
    except Exception as e:
        logger.warning(f"Không thể load tên tiếng Việt: {e}")

    # 2. Xây dựng options với label đầy đủ 3 thành phần
    seen = set()
    options = []
    for row in records:
        ticker = str(row.get('Ticker', '')).strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        
        name_en = str(row.get('Company Common Name', '')).strip()
        if name_en.lower() in ('nan', 'none', '', '0'): name_en = ""
        
        name_vi = vn_names.get(ticker, "")
        
        # Tạo chuỗi hiển thị: TICKER - Tên Việt - Tên Anh
        label_parts = [ticker]
        if name_vi: label_parts.append(name_vi)
        if name_en and name_en != name_vi: label_parts.append(name_en)
        
        full_label = " - ".join(label_parts)

        options.append({
            'label': full_label, 
            'value': ticker,
            'title': full_label # Hiển thị tooltip khi di chuột vào
        })

    options.sort(key=lambda x: x['value'])
    return options

import requests
import logging

logger = logging.getLogger(__name__)

def fetch_index_constituents(index_code):
    """
    Gọi API của SSI để lấy danh sách mã cổ phiếu theo chỉ số.
    Trả về tuple: (list_tickers, error_message)
    """
    if not index_code or index_code == "all":
        return None, None
        
    url = f"https://iboard-query.ssi.com.vn/stock/group/{index_code.upper()}"
    try:
        # Giả mạo User-Agent để tránh bị SSI block request (403 Forbidden)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=2) # Timeout 2s tránh treo web
        response.raise_for_status()
        
        data = response.json()
        if data.get("code") == "SUCCESS":
            # Duyệt qua mảng data, lấy ra trường 'stockSymbol'
            tickers = [item["stockSymbol"] for item in data.get("data", [])]
            return tickers, None
        else:
            return [], f"Lỗi từ SSI: {data.get('message')}"
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi fetch API SSI cho {index_code}: {e}")
        return [], "Không thể kết nối máy chủ dữ liệu. Vui lòng thử lại sau."
import pandas as pd
import os

# Nếu thư mục processed của bạn nằm ở data/processed/
def get_full_prices_df():
    """Lấy toàn bộ lịch sử giá để tính toán chỉ báo (Momentum, MA)"""
    try:
        return pd.read_parquet("data/processed/HISTORICAL_PRICES.parquet")
    except:
        print("❌ Không tìm thấy file HISTORICAL_PRICES.parquet")
        return pd.DataFrame()

def get_index_df():
    """Lấy dữ liệu VNINDEX"""
    try:
        return pd.read_parquet("data/processed/INDEX.parquet")
    except:
        print("❌ Không tìm thấy file INDEX.parquet")
        return pd.DataFrame()

def get_company_info():
    """Lấy thông tin ngành nghề của doanh nghiệp"""
    try:
        # Tên file có thể là COMP.parquet hoặc COMP_INFO.parquet tuỳ vào file convert của bạn
        if os.path.exists("data/processed/COMP_INFO.parquet"):
            return pd.read_parquet("data/processed/COMP_INFO.parquet")
        return pd.read_parquet("data/processed/COMP.parquet")
    except:
        print("❌ Không tìm thấy file COMP.parquet")
        return pd.DataFrame()