from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from src.config import Settings


class KISConfigurationError(RuntimeError):
    pass


@dataclass
class _Token:
    value: str
    expires_at: datetime


class KoreaInvestmentAdapter:
    """Read-only Korea Investment Securities Open API adapter.

    This MVP exposes authentication and market-data requests only. It contains
    no order endpoint by design, which keeps recommendations separate from
    execution even when live credentials are configured.
    """

    REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
    PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"

    def __init__(self, settings: Settings, timeout: int = 10) -> None:
        self.settings = settings
        self.timeout = timeout
        self.base_url = self.PAPER_BASE_URL if settings.paper_trading else self.REAL_BASE_URL
        self._token: _Token | None = None

    @property
    def mode_label(self) -> str:
        return "모의투자" if self.settings.paper_trading else "실전"

    def _ensure_configured(self) -> None:
        if not self.settings.credentials_ready:
            raise KISConfigurationError("APP_KEY, APP_SECRET, ACCOUNT_NO를 .env에 설정해 주세요.")

    def access_token(self) -> str:
        self._ensure_configured()
        now = datetime.now(timezone.utc)
        if self._token and self._token.expires_at > now + timedelta(minutes=3):
            return self._token.value
        response = requests.post(
            f"{self.base_url}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self.settings.app_key,
                "appsecret": self.settings.app_secret,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        self._token = _Token(payload["access_token"], now + timedelta(hours=23))
        return self._token.value

    def _headers(self, tr_id: str) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.access_token()}",
            "appkey": self.settings.app_key,
            "appsecret": self.settings.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
            "content-type": "application/json; charset=utf-8",
        }

    def current_price(self, ticker: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers=self._headers("FHKST01010100"),
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("output", {})

    def daily_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        response = requests.get(
            f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            headers=self._headers("FHKST03010100"),
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": start.replace("-", ""),
                "FID_INPUT_DATE_2": end.replace("-", ""),
                "FID_PERIOD_DIV_CODE": "D",
                "FID_ORG_ADJ_PRC": "0",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        items = response.json().get("output2", [])
        if not items:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(items).rename(
            columns={
                "stck_bsop_date": "date",
                "stck_oprc": "open",
                "stck_hgpr": "high",
                "stck_lwpr": "low",
                "stck_clpr": "close",
                "acml_vol": "volume",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["ticker"] = ticker
        return frame[["date", "ticker", "open", "high", "low", "close", "volume"]].sort_values("date")

    def realtime_snapshot(self, ticker: str) -> dict[str, Any]:
        """Return the read-only fields used to build an intraday OHLCV row."""
        quote = self.current_price(ticker)
        return {
            "ticker": ticker,
            "open": quote.get("stck_oprc"),
            "high": quote.get("stck_hgpr"),
            "low": quote.get("stck_lwpr"),
            "close": quote.get("stck_prpr"),
            "volume": quote.get("acml_vol"),
        }

    def submit_order(self, *_: Any, **__: Any) -> None:
        raise PermissionError("이 MVP는 자동매매를 지원하지 않습니다. 주문은 증권사 앱에서 직접 실행하세요.")
