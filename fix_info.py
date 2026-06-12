import pandas as pd
import requests
from pathlib import Path


def fetch_company_info():
    """Tự động lấy thông tin cơ bản (Tên, Ngành) của toàn bộ cổ phiếu trên HOSE, HNX, UPCOM"""
    print("🔎 Đang tải thông tin doanh nghiệp từ VNDirect...")
    url = "https://finfo-api.vndirect.com.vn/v4/stocks"
    params = {
        "q": "type:STOCK",  # Lấy tất cả cổ phiếu
        "size": 2000  # Size lớn để bao trọn 3 sàn
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            info_data = []

            for item in data:
                symbol = item.get("symbol", "")
                if len(symbol) != 3:  # Lọc bỏ chứng quyền
                    continue

                # Xử lý format data
                exchange = item.get("floor", "HOSE")
                sector = item.get("industryName", "Khác")

                info_data.append({
                    "ticker": f"{symbol}.{exchange}",
                    "company_name": item.get("companyName", f"Công ty Cổ phần {symbol}"),
                    "country": "VN",
                    "industry": sector,
                    "sector": sector,
                    "ipo_date": item.get("listedDate", "2015-01-01"),
                    "sector_vn": sector,
                    "last_date": "2026-06-05"  # Default theo format cũ
                })

            return pd.DataFrame(info_data)

    except Exception as e:
        print(f"⚠ Lỗi tải thông tin API: {e}")
        return pd.DataFrame()


def main():
    print("=" * 50)
    print(" Cập nhật file HISTORICAL_PRICES (Thông tin Ngành)")
    print("=" * 50)

    df = fetch_company_info()

    if not df.empty:
        output_path = Path("data/HISTORICAL_PRICES.xlsx")
        output_path.parent.mkdir(exist_ok=True)

        # Xoá file Parquet cache cũ để engine_v2.py đọc lại file mới
        info_pq = Path("data/info.parquet")
        if info_pq.exists():
            info_pq.unlink()
            print("🗑️ Đã xoá cache info.parquet")

        df.to_excel(output_path, index=False, sheet_name="Sheet1")
        print(f"✅ Đã tạo xong file {output_path} với {len(df)} mã!")
    else:
        print("❌ Cập nhật thất bại.")


if __name__ == "__main__":
    main()