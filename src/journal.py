from __future__ import annotations

from pathlib import Path

import pandas as pd


JOURNAL_COLUMNS = ["date", "ticker", "name", "side", "quantity", "price", "strategy", "memo"]


def load_journal(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    frame = pd.read_csv(path, dtype={"ticker": str})
    for column in JOURNAL_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[JOURNAL_COLUMNS]


def append_trade(path: Path, trade: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_journal(path)
    updated = pd.concat([current, pd.DataFrame([trade], columns=JOURNAL_COLUMNS)], ignore_index=True)
    updated.to_csv(path, index=False, encoding="utf-8-sig")

