"""
app_dashboard.py  ── VN Stock Screener Dashboard
═══════════════════════════════════════════════════
Hiển thị toàn bộ ~400 mã HOSE với bộ lọc tín hiệu:
  🟢 Mua Mạnh   (ml_prob >= 0.65 và Total_Score >= 70)
  🔵 Mua        (ml_prob >= 0.50 và Total_Score >= 50)
  🔴 Tránh      (còn lại)

Cách chạy:
    python app_dashboard.py
Sau đó mở: http://127.0.0.1:8050
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Thêm thư mục hiện tại vào path để import ml_stock_screener ──────────────
sys.path.insert(0, str(Path(__file__).parent))

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG & DATA LOADER
# ════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
POSSIBLE_DIRS = [
    BASE_DIR / "data" / "processed",
    BASE_DIR / ".." / "data" / "processed",
    BASE_DIR / ".." / ".." / "data" / "processed",
    Path(r"C:\Users\Sang\Downloads\data\processed"),
]
PROCESSED_DIR = next((d for d in POSSIBLE_DIRS if d.exists()), None)


def get_parquet(filenames: list) -> pd.DataFrame:
    if not PROCESSED_DIR:
        return pd.DataFrame()
    for name in filenames:
        fp = PROCESSED_DIR / name
        if fp.exists():
            try:
                return pd.read_parquet(fp)
            except Exception:
                pass
    return pd.DataFrame()


def classify_signals(df: pd.DataFrame, df_source: pd.DataFrame = None) -> pd.Series:
    """
    Phân loại tín hiệu dùng ngưỡng đã calibrate từ walk-forward test set.

    Bước 1 — Hard filter volume: mã Vol_Flag <= -30 → "Tránh" ngay, không xét tiếp
    Bước 2 — Ngưỡng calibrated (ưu tiên) hoặc percentile fallback
    """
    if df.empty:
        return pd.Series(dtype=str)

    src = df_source if df_source is not None else df

    if src.attrs.get("calibrated"):
        t_strong = src.attrs["threshold_strong"]
        t_buy    = src.attrs["threshold_buy"]
        method   = "calibrated"
    else:
        t_strong = df["ml_prob"].quantile(0.90)
        t_buy    = df["ml_prob"].quantile(0.75)
        method   = "percentile fallback"

    score_p75 = df["Total_Score"].quantile(0.75)

    # Hard volume filter: loại thẳng mã volume rất xấu (Vol_Flag = -30)
    vol_flag = df.get("Vol_Flag", pd.Series(0, index=df.index))
    n_blocked = (vol_flag <= -30).sum()
    n_warned  = ((vol_flag == -15)).sum()
    print(
        f"  📊 Ngưỡng ({method}): Mua Mạnh >= {t_strong:.3f} | Mua >= {t_buy:.3f} | Score p75 = {score_p75:.1f}\n"
        f"  🚫 Volume hard filter: {n_blocked} mã bị loại thẳng | {n_warned} mã cảnh báo"
    )

    def _classify(row):
        # Bước 1: hard filter
        if row.get("Vol_Flag", 0) <= -30:
            return "Tránh"
        # Bước 2: phân loại theo ML + Score
        if row["ml_prob"] >= t_strong and row["Total_Score"] >= score_p75:
            return "Mua Mạnh"
        elif row["ml_prob"] >= t_buy:
            # Mã bị cảnh báo volume (-15) chỉ lên tối đa "Mua", không thể "Mua Mạnh"
            return "Mua"
        return "Tránh"

    return df.apply(_classify, axis=1)


def load_and_compute() -> pd.DataFrame:
    """Load data, chạy pipeline, trả về DataFrame đã có tín hiệu."""
    print("📊 Đang nạp dữ liệu...", flush=True)

    df_price = get_parquet(["prices.parquet", "HISTORICAL_PRICES.parquet"])
    df_index = get_parquet(["index.parquet", "INDEX.parquet"])
    df_info  = get_parquet(["info.parquet", "COMP.parquet", "COMP_INFO.parquet"])
    df_fund  = get_parquet(["financials.parquet"])

    if df_price.empty:
        print("❌ Không có dữ liệu giá!")
        return pd.DataFrame()

    from ml_stock_screener import FeatureEngineer, RuleBasedScorer, MLPredictor

    engineer    = FeatureEngineer(df_price, df_index, df_fund=df_fund, df_info=df_info)
    df_features = engineer.run_pipeline()

    scorer    = RuleBasedScorer(df_features, has_fundamentals=engineer.has_fundamentals)
    df_scored = scorer.calculate_total_score()

    ml_engine = MLPredictor(df_scored)
    df_final  = ml_engine.train_walk_forward()

    # Lấy snapshot ngày mới nhất mỗi mã
    df_final["Date"] = pd.to_datetime(df_final["Date"])
    latest = df_final.groupby("Ticker")["Date"].max().reset_index()
    latest.columns = ["Ticker", "LatestDate"]
    df_snap = df_final.merge(latest, on="Ticker")
    df_snap = df_snap[df_snap["Date"] == df_snap["LatestDate"]].copy()

    # Tín hiệu (percentile-based — tự thích nghi theo thị trường)
    df_snap["Tín Hiệu"] = classify_signals(df_snap, df_source=df_final)

    # Làm gọn để hiển thị
    DISPLAY_COLS = {
        "Ticker":          "Mã CP",
        "Sector":          "Ngành",
        "Close":           "Giá",
        "ret_20d":         "% 20 ngày",
        "ret_60d":         "% 60 ngày",
        "rsi_14":          "RSI",
        "Total_Score":     "Điểm",
        "ml_prob":         "Xác Suất ML",
        "market_regime":   "Thị Trường",
        "Tín Hiệu":        "Tín Hiệu",
    }
    available = {k: v for k, v in DISPLAY_COLS.items() if k in df_snap.columns}
    df_out = df_snap[list(available.keys())].copy()
    df_out = df_out.rename(columns=available)

    # Format số
    for col in ["% 20 ngày", "% 60 ngày"]:
        if col in df_out.columns:
            df_out[col] = (df_out[col] * 100).round(2)
    for col in ["RSI", "Điểm"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].round(1)
    if "Xác Suất ML" in df_out.columns:
        df_out["Xác Suất ML"] = (df_out["Xác Suất ML"] * 100).round(1)
    if "Giá" in df_out.columns:
        df_out["Giá"] = df_out["Giá"].round(0).astype(int)

    df_out = df_out.sort_values("Điểm", ascending=False).reset_index(drop=True)
    print(f"✅ Xong: {len(df_out)} mã | {datetime.now().strftime('%H:%M:%S')}")
    return df_out


# ════════════════════════════════════════════════════════════════════════════
#  LOAD DATA KHI KHỞI ĐỘNG
# ════════════════════════════════════════════════════════════════════════════
DF_GLOBAL = load_and_compute()

# Summary counts
def get_counts(df):
    if df.empty:
        return 0, 0, 0, 0
    total    = len(df)
    mua_manh = (df["Tín Hiệu"] == "Mua Mạnh").sum()
    mua      = (df["Tín Hiệu"] == "Mua").sum()
    tranh    = (df["Tín Hiệu"] == "Tránh").sum()
    return total, mua_manh, mua, tranh

total, mua_manh, mua, tranh = get_counts(DF_GLOBAL)

# ════════════════════════════════════════════════════════════════════════════
#  DASH APP
# ════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="VN Stock Screener",
)

# ── Màu sắc ──────────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#0d1117",
    "card":      "#161b22",
    "border":    "#30363d",
    "green":     "#3fb950",
    "blue":      "#58a6ff",
    "red":       "#f85149",
    "yellow":    "#d29922",
    "text":      "#e6edf3",
    "muted":     "#8b949e",
}

SIGNAL_COLOR = {
    "Mua Mạnh": COLORS["green"],
    "Mua":      COLORS["blue"],
    "Tránh":    COLORS["red"],
}

# ── Table columns ─────────────────────────────────────────────────────────────
TABLE_COLS = [{"name": c, "id": c} for c in DF_GLOBAL.columns] if not DF_GLOBAL.empty else []

TABLE_STYLE_CELL_CONDITIONAL = [
    {"if": {"column_id": "Mã CP"},       "fontWeight": "700", "color": COLORS["blue"]},
    {"if": {"column_id": "Tín Hiệu"},    "fontWeight": "700"},
    {"if": {"column_id": "Điểm"},        "fontWeight": "600"},
    {"if": {"column_id": "Xác Suất ML"}, "fontWeight": "600"},
]

TABLE_STYLE_DATA_CONDITIONAL = [
    {
        "if": {"filter_query": '{Tín Hiệu} = "Mua Mạnh"', "column_id": "Tín Hiệu"},
        "color": COLORS["green"], "backgroundColor": "#0d2818",
    },
    {
        "if": {"filter_query": '{Tín Hiệu} = "Mua"', "column_id": "Tín Hiệu"},
        "color": COLORS["blue"], "backgroundColor": "#0c1e35",
    },
    {
        "if": {"filter_query": '{Tín Hiệu} = "Tránh"', "column_id": "Tín Hiệu"},
        "color": COLORS["red"], "backgroundColor": "#2d1117",
    },
    # Row highlight khi hover
    {"if": {"state": "active"}, "backgroundColor": "#21262d", "border": f"1px solid {COLORS['border']}"},
    {"if": {"row_index": "odd"}, "backgroundColor": "#0d1117"},
]

# ── Helper functions ──────────────────────────────────────────────────────────

def _build_summary_cards(total, mua_manh, mua, tranh):
    def card(label, value, color, icon):
        return html.Div(
            style={
                "backgroundColor": COLORS["card"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "8px",
                "padding": "16px 24px",
                "minWidth": "150px",
                "flex": "1",
            },
            children=[
                html.Div(f"{icon} {label}", style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "6px"}),
                html.Div(str(value), style={"color": color, "fontSize": "28px", "fontWeight": "700"}),
            ]
        )

    return [
        card("Tổng mã",   total,    COLORS["text"],   "📊"),
        card("Mua Mạnh",  mua_manh, COLORS["green"],  "🟢"),
        card("Mua",       mua,      COLORS["blue"],   "🔵"),
        card("Tránh",     tranh,    COLORS["red"],    "🔴"),
    ]


def _sector_options():
    if DF_GLOBAL.empty or "Ngành" not in DF_GLOBAL.columns:
        return []
    sectors = sorted(DF_GLOBAL["Ngành"].dropna().unique())
    return [{"label": s, "value": s} for s in sectors]


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "fontFamily": "'Inter', 'Segoe UI', sans-serif"},
    children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div(
            style={
                "backgroundColor": COLORS["card"],
                "borderBottom": f"1px solid {COLORS['border']}",
                "padding": "16px 32px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
            },
            children=[
                html.Div([
                    html.Span("VN", style={"color": COLORS["green"], "fontWeight": "800", "fontSize": "22px"}),
                    html.Span(" Stock Screener", style={"color": COLORS["text"], "fontWeight": "600", "fontSize": "22px"}),
                    html.Span(
                        f" — {datetime.now().strftime('%d/%m/%Y')}",
                        style={"color": COLORS["muted"], "fontSize": "14px", "marginLeft": "12px"}
                    ),
                ]),
                html.Button(
                    "🔄 Cập nhật",
                    id="btn-refresh",
                    style={
                        "backgroundColor": COLORS["border"],
                        "color": COLORS["text"],
                        "border": "none",
                        "borderRadius": "6px",
                        "padding": "8px 18px",
                        "cursor": "pointer",
                        "fontSize": "14px",
                    }
                ),
            ]
        ),

        # ── Summary Cards ─────────────────────────────────────────────────────
        html.Div(
            id="summary-cards",
            style={"display": "flex", "gap": "16px", "padding": "24px 32px 8px"},
            children=_build_summary_cards(total, mua_manh, mua, tranh),
        ),

        # ── Filter Bar ────────────────────────────────────────────────────────
        html.Div(
            style={
                "padding": "16px 32px",
                "display": "flex",
                "gap": "16px",
                "alignItems": "center",
                "flexWrap": "wrap",
            },
            children=[
                # Tín hiệu filter
                html.Div([
                    html.Label("Tín Hiệu", style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(
                        id="filter-signal",
                        options=[
                            {"label": "🟢 Mua Mạnh", "value": "Mua Mạnh"},
                            {"label": "🔵 Mua",       "value": "Mua"},
                            {"label": "🔴 Tránh",     "value": "Tránh"},
                        ],
                        multi=True,
                        placeholder="Tất cả tín hiệu",
                        style={"width": "260px", "backgroundColor": COLORS["card"]},
                        className="dark-dropdown",
                    ),
                ]),

                # Ngành filter
                html.Div([
                    html.Label("Ngành", style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(
                        id="filter-sector",
                        options=_sector_options(),
                        multi=True,
                        placeholder="Tất cả ngành",
                        style={"width": "260px"},
                        className="dark-dropdown",
                    ),
                ]),

                # Tìm mã
                html.Div([
                    html.Label("Tìm mã CP", style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "4px", "display": "block"}),
                    dcc.Input(
                        id="filter-ticker",
                        type="text",
                        placeholder="VD: HPG, VNM...",
                        debounce=True,
                        style={
                            "backgroundColor": COLORS["card"],
                            "color": COLORS["text"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "6px",
                            "padding": "8px 12px",
                            "fontSize": "14px",
                            "width": "160px",
                        }
                    ),
                ]),

                # Điểm tối thiểu
                html.Div([
                    html.Label("Điểm tối thiểu", style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "4px", "display": "block"}),
                    dcc.Slider(
                        id="filter-score",
                        min=0, max=100, step=5, value=0,
                        marks={0: "0", 50: "50", 100: "100"},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="score-slider",
                    ),
                ], style={"width": "200px"}),

                # Thị trường
                html.Div([
                    html.Label("Thị Trường", style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(
                        id="filter-regime",
                        options=[
                            {"label": "🐂 Bull", "value": "bull"},
                            {"label": "😐 Neutral", "value": "neutral"},
                            {"label": "🐻 Bear", "value": "bear"},
                        ],
                        multi=True,
                        placeholder="Tất cả",
                        style={"width": "200px"},
                        className="dark-dropdown",
                    ),
                ]),
            ]
        ),

        # ── Result count ──────────────────────────────────────────────────────
        html.Div(
            id="result-count",
            style={"padding": "0 32px 8px", "color": COLORS["muted"], "fontSize": "13px"}
        ),

        # ── Data Table ────────────────────────────────────────────────────────
        html.Div(
            style={"padding": "0 32px 40px"},
            children=[
                dash_table.DataTable(
                    id="stock-table",
                    columns=TABLE_COLS,
                    data=DF_GLOBAL.to_dict("records") if not DF_GLOBAL.empty else [],
                    sort_action="native",
                    filter_action="none",
                    page_action="native",
                    page_size=30,
                    style_table={
                        "overflowX": "auto",
                        "borderRadius": "8px",
                        "border": f"1px solid {COLORS['border']}",
                    },
                    style_header={
                        "backgroundColor": "#21262d",
                        "color": COLORS["text"],
                        "fontWeight": "600",
                        "fontSize": "13px",
                        "border": f"1px solid {COLORS['border']}",
                        "padding": "10px 14px",
                    },
                    style_data={
                        "backgroundColor": COLORS["card"],
                        "color": COLORS["text"],
                        "fontSize": "13px",
                        "border": f"1px solid {COLORS['border']}",
                        "padding": "9px 14px",
                    },
                    style_cell_conditional=TABLE_STYLE_CELL_CONDITIONAL,
                    style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                    style_cell={"textAlign": "left", "whiteSpace": "normal"},
                )
            ]
        ),

        # ── Loading overlay ───────────────────────────────────────────────────
        dcc.Loading(
            id="loading-refresh",
            type="circle",
            color=COLORS["green"],
            children=html.Div(id="refresh-trigger", style={"display": "none"}),
        ),

        # Global CSS → xem file assets/custom.css
    ]
)


# ════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("stock-table",    "data"),
    Output("result-count",   "children"),
    Output("summary-cards",  "children"),
    Input("filter-signal",   "value"),
    Input("filter-sector",   "value"),
    Input("filter-ticker",   "value"),
    Input("filter-score",    "value"),
    Input("filter-regime",   "value"),
    Input("refresh-trigger", "children"),
)
def update_table(signals, sectors, ticker_search, min_score, regimes, _refresh):
    df = DF_GLOBAL.copy()
    if df.empty:
        return [], "Không có dữ liệu.", _build_summary_cards(0, 0, 0, 0)

    # Filter tín hiệu
    if signals:
        df = df[df["Tín Hiệu"].isin(signals)]

    # Filter ngành
    if sectors and "Ngành" in df.columns:
        df = df[df["Ngành"].isin(sectors)]

    # Filter mã
    if ticker_search and ticker_search.strip() and "Mã CP" in df.columns:
        q = ticker_search.strip().upper()
        df = df[df["Mã CP"].str.upper().str.contains(q, na=False)]

    # Filter điểm
    if min_score and "Điểm" in df.columns:
        df = df[df["Điểm"] >= min_score]

    # Filter thị trường
    if regimes and "Thị Trường" in df.columns:
        df = df[df["Thị Trường"].isin(regimes)]

    n = len(df)
    count_text = f"Hiển thị {n} mã cổ phiếu"

    t, mm, m, tr = get_counts(df)
    cards = _build_summary_cards(t, mm, m, tr)

    return df.to_dict("records"), count_text, cards


@app.callback(
    Output("refresh-trigger", "children"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_data(n_clicks):
    global DF_GLOBAL
    DF_GLOBAL = load_and_compute()
    return str(datetime.now())


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  VN Stock Screener Dashboard")
    print(f"  Mở trình duyệt: http://127.0.0.1:8050")
    print("═" * 55 + "\n")
    app.run(debug=False, port=8050)