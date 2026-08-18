from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ma_5_gap",
    "ma_20_gap",
    "ma_60_gap",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_position",
    "bb_width",
    "atr_pct",
    "volume_ratio_20",
    "relative_strength_20",
]

FEATURE_LABELS = {
    "ret_1d": "1일 수익률",
    "ret_5d": "5일 수익률",
    "ret_20d": "20일 수익률",
    "ma_5_gap": "5일 이동평균 괴리",
    "ma_20_gap": "20일 이동평균 괴리",
    "ma_60_gap": "60일 이동평균 괴리",
    "rsi_14": "RSI(14)",
    "macd": "MACD",
    "macd_signal": "MACD 신호선",
    "bb_position": "볼린저밴드 내 위치",
    "bb_width": "볼린저밴드 폭",
    "atr_pct": "ATR 변동성 비율",
    "volume_ratio_20": "20일 평균 대비 거래량",
    "relative_strength_20": "시장 대비 20일 상대강도",
}

FEATURE_DESCRIPTIONS = {
    "ret_1d": "전 거래일 대비 오늘 가격이 얼마나 변했는지 나타냅니다.",
    "ret_5d": "최근 5거래일 동안의 가격 변동률입니다.",
    "ret_20d": "최근 20거래일 동안의 가격 변동률입니다.",
    "ma_5_gap": "현재가가 최근 5일 평균가격보다 얼마나 위나 아래에 있는지 나타냅니다.",
    "ma_20_gap": "현재가가 최근 20일 평균가격보다 얼마나 위나 아래에 있는지 나타냅니다.",
    "ma_60_gap": "현재가가 최근 60일 평균가격보다 얼마나 위나 아래에 있는지 나타냅니다.",
    "rsi_14": "최근 14일의 상승과 하락 힘을 비교한 값입니다. 보통 70 이상은 과열, 30 이하는 침체로 참고합니다.",
    "macd": "단기와 장기 이동평균의 차이로 가격 추세와 상승·하락 힘의 변화를 살핍니다.",
    "macd_signal": "MACD의 최근 평균선으로, MACD와 비교해 추세 변화 가능성을 살핍니다.",
    "bb_position": "최근 가격 변동 범위의 하단과 상단 사이에서 현재가가 어디에 있는지 나타냅니다.",
    "bb_width": "볼린저밴드의 폭입니다. 넓을수록 최근 가격 변동성이 큰 편입니다.",
    "atr_pct": "최근 14일 평균 가격 변동폭을 현재가로 나눈 비율입니다. 높을수록 가격 움직임이 큰 편입니다.",
    "volume_ratio_20": "현재 거래량이 최근 20일 평균 거래량의 몇 배인지 나타냅니다.",
    "relative_strength_20": "해당 종목의 20일 수익률이 분석종목 평균보다 얼마나 강하거나 약한지 나타냅니다.",
}


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def add_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build features using only information available at each row's close."""
    required = {"date", "ticker", "name", "open", "high", "low", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

    frame = ohlcv.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    market_close = frame.pivot(index="date", columns="ticker", values="close")
    market_ret_20 = market_close.pct_change(20, fill_method=None).mean(axis=1)

    def transform(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        close = g["close"].astype(float)
        g["ret_1d"] = close.pct_change(fill_method=None)
        g["ret_5d"] = close.pct_change(5, fill_method=None)
        g["ret_20d"] = close.pct_change(20, fill_method=None)
        for window in (5, 20, 60):
            ma = close.rolling(window).mean()
            g[f"ma_{window}_gap"] = close / ma - 1
        g["rsi_14"] = _rsi(close) / 100.0
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_raw = ema12 - ema26
        g["macd"] = macd_raw / close
        g["macd_signal"] = macd_raw.ewm(span=9, adjust=False).mean() / close
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper, lower = mid + 2 * std, mid - 2 * std
        g["bb_position"] = (close - lower) / (upper - lower).replace(0, np.nan)
        g["bb_width"] = (upper - lower) / mid
        previous = close.shift(1)
        true_range = pd.concat(
            [(g["high"] - g["low"]), (g["high"] - previous).abs(), (g["low"] - previous).abs()], axis=1
        ).max(axis=1)
        g["atr_pct"] = true_range.rolling(14).mean() / close
        g["volume_ratio_20"] = g["volume"] / g["volume"].rolling(20).mean()
        g["relative_strength_20"] = g["ret_20d"] - g["date"].map(market_ret_20)
        g["next_return"] = close.shift(-1) / close - 1
        g["target"] = np.where(g["next_return"].notna(), (g["next_return"] > 0).astype(int), np.nan)
        return g

    # Explicit concatenation keeps identifiers and avoids pandas' changing
    # groupby.apply inclusion semantics.
    parts = [transform(group) for _, group in frame.groupby("ticker", sort=False)]
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def latest_feature_rows(featured: pd.DataFrame) -> pd.DataFrame:
    valid = featured.dropna(subset=FEATURE_COLUMNS)
    return valid.sort_values("date").groupby("ticker", as_index=False).tail(1).copy()


def equal_weight_market_index(featured: pd.DataFrame, window: int = 120) -> pd.Series:
    """Return a recent equal-weight index rebased to 100 at the first displayed date."""
    market_return = (
        featured.pivot(index="date", columns="ticker", values="close")
        .pct_change(fill_method=None)
        .mean(axis=1)
        .fillna(0)
        .tail(window)
    )
    index = (1 + market_return).cumprod()
    if index.empty or index.iloc[0] == 0:
        return index
    return index / index.iloc[0] * 100
