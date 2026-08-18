from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import latest_feature_rows
from src.model import ModelResult, predict_upside


CONFIDENCE_LABELS = ["관찰", "보통", "높음", "매우 높음"]
CONFIDENCE_LEVELS = {label: level for level, label in enumerate(CONFIDENCE_LABELS, start=1)}


def _reason(row: pd.Series) -> str:
    signals: list[str] = []
    if row["ma_20_gap"] > 0:
        signals.append("20일선 상회")
    if row["relative_strength_20"] > 0:
        signals.append("시장 대비 상대강도 우위")
    if 0.48 <= row["rsi_14"] <= 0.72:
        signals.append("RSI 상승 여력")
    if row["macd"] > row["macd_signal"]:
        signals.append("MACD 모멘텀 개선")
    if row["volume_ratio_20"] > 1.05:
        signals.append("거래량 증가")
    return " · ".join(signals[:3]) or "복합 기술지표 점수 상위"


def rank_stocks(featured: pd.DataFrame, result: ModelResult, top_n: int = 5) -> pd.DataFrame:
    latest = latest_feature_rows(featured)
    latest["probability"] = predict_upside(result.model, latest)
    # A transparent secondary score rewards probability, relative strength and liquidity confirmation.
    latest["score"] = (
        0.78 * latest["probability"]
        + 0.14 * (latest["relative_strength_20"].clip(-0.2, 0.2) + 0.2) / 0.4
        + 0.08 * latest["volume_ratio_20"].clip(0.5, 2.0) / 2.0
    )
    latest["entry"] = latest["close"]
    risk = (1.5 * latest["atr_pct"]).clip(0.025, 0.08)
    reward = (2.0 * risk).clip(0.05, 0.14)
    latest["target_price"] = latest["entry"] * (1 + reward)
    latest["stop_price"] = latest["entry"] * (1 - risk)
    latest["confidence"] = pd.cut(
        latest["probability"], bins=[-np.inf, 0.54, 0.62, 0.70, np.inf], labels=CONFIDENCE_LABELS
    ).astype(str)
    latest["confidence_level"] = latest["confidence"].map(CONFIDENCE_LEVELS).astype(int)
    latest["reason"] = latest.apply(_reason, axis=1)
    return latest.sort_values(["score", "probability"], ascending=False).head(top_n).reset_index(drop=True)
