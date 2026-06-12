"""
PATCH cho engine_v2.py — thay hàm load_scoring_weights() hiện tại
bằng phiên bản load theo regime động.

Cách dùng:
    Thay thế hàm load_scoring_weights() trong engine_v2.py bằng đoạn dưới đây,
    và cập nhật dòng gọi W = load_scoring_weights() trong compute_all_scores()
    thành W = load_scoring_weights(regime).
"""

import json
from pathlib import Path

BASE         = Path(__file__).parent
DATA_DIR     = BASE / "data"
WEIGHTS_FILE = DATA_DIR / "calibrated_weights.json"

DEFAULT_WEIGHTS_PER_REGIME = {
    "bull":     {"score_momentum": 0.35, "score_rs": 0.25,
                 "score_fundamental": 0.20, "score_seasonal": 0.20},
    "bear":     {"score_momentum": 0.15, "score_rs": 0.20,
                 "score_fundamental": 0.40, "score_seasonal": 0.25},
    "sideways": {"score_momentum": 0.25, "score_rs": 0.25,
                 "score_fundamental": 0.25, "score_seasonal": 0.25},
}


def load_scoring_weights(regime: str = "sideways") -> dict:
    """
    Load trọng số tối ưu cho regime hiện tại từ calibrated_weights.json.

    Parameters
    ----------
    regime : str
        "bull" | "bear" | "sideways" — lấy từ compute_market_regime()

    Returns
    -------
    dict với 4 keys: score_momentum, score_rs, score_fundamental, score_seasonal
    """
    regime = regime if regime in ("bull", "bear", "sideways") else "sideways"

    if WEIGHTS_FILE.exists():
        try:
            with open(WEIGHTS_FILE, encoding="utf-8") as f:
                all_weights = json.load(f)

            # Format mới: {"bull": {...}, "bear": {...}, "sideways": {...}}
            if regime in all_weights:
                w = all_weights[regime]
            # Format cũ (single dict) → dùng cho mọi regime
            elif "score_momentum" in all_weights:
                w = all_weights
            else:
                raise KeyError(f"Không tìm thấy regime '{regime}' trong file.")

            keys = ["score_momentum", "score_rs",
                    "score_fundamental", "score_seasonal"]
            if not all(k in w for k in keys):
                raise ValueError("Thiếu keys trong weights file.")

            # Re-normalize phòng trường hợp tổng != 1
            total  = sum(w[k] for k in keys)
            result = {k: w[k] / total for k in keys}

            precision = w.get("precision", "N/A")
            method    = w.get("calibration_method", "unknown")
            print(
                f"✓ Weights [{regime}] loaded (method={method}, "
                f"precision={precision}): "
                + " ".join(f"{k.replace('score_','')}={v:.3f}"
                           for k, v in result.items())
            )
            return result

        except Exception as e:
            print(f"⚠ Không load được calibrated_weights.json: {e} "
                  f"— dùng default [{regime}]")

    # Fallback: domain default theo regime
    w = DEFAULT_WEIGHTS_PER_REGIME[regime].copy()
    print(f"  Using domain default weights [{regime}]")
    return w


# ── Hướng dẫn sửa compute_all_scores() trong engine_v2.py ────────────────────
#
# Tìm đoạn:
#     W = load_scoring_weights()
#
# Thay bằng:
#     W = load_scoring_weights(regime)   # ← truyền regime vào
#
# Lúc này regime đã được tính ở trên trong compute_all_scores():
#     regime_info = compute_market_regime(index_df, ref_date)
#     regime      = regime_info["regime"]
#
# Không cần thay đổi gì khác — hàm trả về dict cùng format cũ.
# ─────────────────────────────────────────────────────────────────────────────