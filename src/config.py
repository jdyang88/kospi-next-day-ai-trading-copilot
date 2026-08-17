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
    paper_trading: bool = True
    auto_trading_enabled: bool = False

    @property
    def credentials_ready(self) -> bool:
        return bool(self.app_key and self.app_secret and self.account_no)


def load_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")
    return Settings(
        app_key=os.getenv("APP_KEY", "").strip(),
        app_secret=os.getenv("APP_SECRET", "").strip(),
        account_no=os.getenv("ACCOUNT_NO", "").strip(),
        account_product_code=os.getenv("ACCOUNT_PRODUCT_CODE", "01").strip(),
        paper_trading=os.getenv("KIS_MODE", "paper").lower() != "live",
        # Safety invariant: this MVP never enables order submission.
        auto_trading_enabled=False,
    )

