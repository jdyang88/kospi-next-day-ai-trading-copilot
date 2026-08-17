from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
JOURNAL_PATH = DATA_DIR / "trade_journal.csv"


@dataclass(frozen=True)
class Settings:
    app_key: str = ""
    app_secret: str = ""
    account_no: str = ""
    account_product_code: str = "01"
    watchlist: str = ""
    paper_trading: bool = True
    auto_trading_enabled: bool = False

    @property
    def credentials_ready(self) -> bool:
        return bool(self.app_key and self.app_secret and self.account_no)


def load_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")

    # Streamlit Community Cloud exposes encrypted app secrets through
    # st.secrets, while local development continues to use .env variables.
    try:
        import streamlit as st

        cloud_secrets = dict(st.secrets)
    except (ImportError, FileNotFoundError, RuntimeError):
        cloud_secrets = {}

    def value(name: str, default: str = "") -> str:
        return str(os.getenv(name) or cloud_secrets.get(name, default)).strip()

    return Settings(
        app_key=value("APP_KEY"),
        app_secret=value("APP_SECRET"),
        account_no=value("ACCOUNT_NO"),
        account_product_code=value("ACCOUNT_PRODUCT_CODE", "01"),
        watchlist=value("WATCHLIST"),
        paper_trading=value("KIS_MODE", "paper").lower() != "live",
        # Safety invariant: this MVP never enables order submission.
        auto_trading_enabled=False,
    )
