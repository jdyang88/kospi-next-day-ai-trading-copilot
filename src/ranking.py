from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import latest_feature_rows
from src.model import ModelResult, predict_upside


CONFIDENCE_LABELS = ["관찰", "보통", "높음", "매우 높음"]
CONFIDENCE_LEVELS = {label: level for level, label in enumerate(CONFIDENCE_LABELS, start=1)}
CONFIDENCE_GUIDE = (
    "1/4 관찰(54% 이하) → 2/4 보통(54% 초과, 62% 이하) → "
    "3/4 높음(62% 초과, 70% 이하) → 4/4 매우 높음(70% 초과)"
)

# A stock is labelled "매수 추천" only when both the model and the individual
# signal clear transparent minimum gates. These are product safety gates, not a
# claim that the trade will be profitable.
MIN_BUY_AUC = 0.55
MIN_BUY_PROBABILITY = 0.62
MIN_CONFIRMATION_SIGNALS = 2
MODEL_GATE_GUIDE = (
    f"검증 AUC {MIN_BUY_AUC:.2f} 이상 · 방향 정확도가 단순 기준 이상"
)


def model_buy_gate(result: ModelResult) -> tuple[bool, str]:
    if result.auc < MIN_BUY_AUC:
        return False, f"검증 AUC {result.auc:.3f}로 기준 {MIN_BUY_AUC:.2f} 미만"
    baseline_accuracy = getattr(result, "baseline_accuracy", None)
    if baseline_accuracy is None:
        return False, "이전 형식의 캐시 모델이므로 재학습 필요"
    if result.accuracy < baseline_accuracy:
        return (
            False,
            f"방향 정확도 {result.accuracy:.1%}가 단순 기준 {baseline_accuracy:.1%} 미만",
        )
    return True, "모델 검증 기준 통과"


def _signals(row: pd.Series) -> list[str]:
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
    return signals


def _reason(row: pd.Series) -> str:
    return " · ".join(_signals(row)[:3]) or "복합 기술지표 점수 상위"


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
    latest["confirmation_count"] = latest.apply(lambda row: len(_signals(row)), axis=1)
    model_ready, gate_reason = model_buy_gate(result)
    buy_mask = (
        model_ready
        & (latest["probability"] >= MIN_BUY_PROBABILITY)
        & (latest["confirmation_count"] >= MIN_CONFIRMATION_SIGNALS)
    )
    observe_mask = (
        model_ready
        & ~buy_mask
        & (latest["probability"] > 0.54)
    )
    latest["decision"] = np.select(
        [buy_mask, observe_mask],
        ["매수 추천", "관찰"],
        default="매수 보류",
    )
    latest["decision_reason"] = np.where(
        model_ready,
        np.where(
            buy_mask,
            f"상승 점수 {MIN_BUY_PROBABILITY:.0%} 이상 · 기술 신호 {MIN_CONFIRMATION_SIGNALS}개 이상",
            "종목별 매수 기준 미충족",
        ),
        gate_reason,
    )
    latest["reason"] = latest.apply(_reason, axis=1)
    return latest.sort_values(["score", "probability"], ascending=False).head(top_n).reset_index(drop=True)
