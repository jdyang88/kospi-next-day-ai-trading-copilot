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
