from __future__ import annotations

import time
from datetime import date, timedelta
import re
from typing import Mapping, Protocol

import pandas as pd


# Initial editable watchlist shown in a new browser session. This is not an
# investment recommendation and users can add or remove every item.
DEFAULT_WATCHLIST: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "051910": "LG화학",
    "096770": "SK이노베이션",
    "005380": "현대차",
    "069500": "KODEX 200",
    "035420": "NAVER",
    "034730": "SK",
    "000720": "현대건설",
    "068270": "셀트리온",
    "006260": "LS",
    "207940": "삼성바이오로직스",
    "012450": "한화에어로스페이스",
    "316140": "우리금융지주",
}

OHLCV_COLUMNS = ["date", "ticker", "name", "open", "high", "low", "close", "volume"]
MAX_UNIVERSE_SIZE = 30


def parse_watchlist(raw: str) -> tuple[dict[str, str], list[str]]:
    """Parse `종목코드:종목명` items separated by commas, semicolons or lines."""
    watchlist: dict[str, str] = {}
    errors: list[str] = []
    for item in re.split(r"[,;\n]+", raw or ""):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            errors.append(f"'{item}' → 종목코드:종목명 형식이 아닙니다.")
            continue
        ticker, name = (part.strip() for part in item.split(":", 1))
        if not re.fullmatch(r"\d{6}", ticker) or not name:
            errors.append(f"'{item}' → 6자리 종목코드와 종목명이 필요합니다.")
            continue
        watchlist[ticker] = name
    return watchlist, errors


def build_analysis_universe(
    *watchlists: Mapping[str, str],
    max_size: int = MAX_UNIVERSE_SIZE,
) -> tuple[dict[str, str], int]:
    """Merge watchlists by ticker and cap requests to a safe starter size."""
    combined: dict[str, str] = {}
    for watchlist in watchlists:
        combined.update(watchlist)
    truncated = max(0, len(combined) - max_size)
    return dict(list(combined.items())[:max_size]), truncated


class ReadOnlyMarketData(Protocol):
    def daily_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...

    def current_price(self, ticker: str) -> dict: ...


def fetch_kis_daily_history(
    adapter: ReadOnlyMarketData,
    universe: Mapping[str, str] = DEFAULT_WATCHLIST,
    *,
    lookback_days: int = 720,
    max_pages: int = 5,
    request_pause: float = 0.12,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch paged KIS daily history without allowing one symbol to abort a scan."""
    start_date = date.today() - timedelta(days=lookback_days)
    all_frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}

    for ticker, name in universe.items():
        ticker_frames: list[pd.DataFrame] = []
        page_end = date.today()
        try:
            for _ in range(max_pages):
                page = adapter.daily_prices(ticker, start_date.isoformat(), page_end.isoformat())
                if page.empty:
                    break
                ticker_frames.append(page)
                oldest = pd.Timestamp(page["date"].min()).date()
                if oldest <= start_date or len(page) < 90:
                    break
                next_end = oldest - timedelta(days=1)
                if next_end >= page_end:
                    break
                page_end = next_end
                if request_pause:
                    time.sleep(request_pause)

            if not ticker_frames:
                raise RuntimeError("일봉 데이터가 없습니다.")
            ticker_history = pd.concat(ticker_frames, ignore_index=True)
            ticker_history = ticker_history.drop_duplicates(["date", "ticker"], keep="last")
            ticker_history["name"] = name
            all_frames.append(ticker_history[OHLCV_COLUMNS])
        except Exception as exc:  # isolate upstream/API failures by symbol
            errors[ticker] = str(exc)
        if request_pause:
            time.sleep(request_pause)

    if not all_frames:
        return pd.DataFrame(columns=OHLCV_COLUMNS), errors
    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    return result, errors


def apply_kis_quote_snapshot(
    history: pd.DataFrame,
    adapter: ReadOnlyMarketData,
    universe: Mapping[str, str] = DEFAULT_WATCHLIST,
    *,
    request_pause: float = 0.12,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Replace each symbol's latest daily row with the current KIS OHLCV snapshot."""
    if history.empty:
        return history.copy(), {"market": "먼저 KIS 일봉 데이터를 불러와야 합니다."}

    snapshots: list[dict] = []
    errors: dict[str, str] = {}
    now_kst = now or pd.Timestamp.now(tz="Asia/Seoul")
    if now_kst.tzinfo is None:
        now_kst = now_kst.tz_localize("Asia/Seoul")
    else:
        now_kst = now_kst.tz_convert("Asia/Seoul")
    market_has_started = now_kst.weekday() < 5 and now_kst.hour >= 9
    for ticker, name in universe.items():
        ticker_history = history[history["ticker"] == ticker]
        if ticker_history.empty:
            continue
        try:
            quote = adapter.current_price(ticker)
            values = {
                "open": pd.to_numeric(quote.get("stck_oprc"), errors="coerce"),
                "high": pd.to_numeric(quote.get("stck_hgpr"), errors="coerce"),
                "low": pd.to_numeric(quote.get("stck_lwpr"), errors="coerce"),
                "close": pd.to_numeric(quote.get("stck_prpr"), errors="coerce"),
                "volume": pd.to_numeric(quote.get("acml_vol"), errors="coerce"),
            }
            latest_history = ticker_history.sort_values("date").iloc[-1]
            invalid_columns = [
                column for column, value in values.items() if pd.isna(value) or value <= 0
            ]
            if invalid_columns and not market_has_started:
                # Before the Korean market opens (and on weekends), KIS can
                # return the latest close while today's OHLCV fields are zero.
                # Use the latest confirmed KIS daily bar instead of rejecting
                # an otherwise valid symbol or inventing today's bar.
                for column in invalid_columns:
                    values[column] = pd.to_numeric(latest_history[column], errors="coerce")
            if any(pd.isna(value) or value <= 0 for value in values.values()):
                raise RuntimeError("현재가 응답의 OHLCV 값이 불완전합니다.")

            # On weekends/holidays the quote represents the latest business day.
            # Replacing that row avoids inventing a non-trading date.
            snapshot_date = pd.Timestamp(ticker_history["date"].max()).normalize()
            today = now_kst.tz_localize(None).normalize()
            if market_has_started:
                snapshot_date = max(snapshot_date, today)
            snapshots.append({"date": snapshot_date, "ticker": ticker, "name": name, **values})
        except Exception as exc:
            errors[ticker] = str(exc)
        if request_pause:
            time.sleep(request_pause)

    if not snapshots:
        return history.iloc[0:0].copy(), errors

    snapshot_frame = pd.DataFrame(snapshots, columns=OHLCV_COLUMNS)
    successful_tickers = set(snapshot_frame["ticker"])
    history = history[history["ticker"].isin(successful_tickers)].copy()
    snapshot_keys = pd.MultiIndex.from_frame(snapshot_frame[["date", "ticker"]])
    history_keys = pd.MultiIndex.from_frame(history[["date", "ticker"]])
    merged = pd.concat([history.loc[~history_keys.isin(snapshot_keys)], snapshot_frame], ignore_index=True)
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True), errors
