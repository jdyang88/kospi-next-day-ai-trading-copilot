from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


DEMO_UNIVERSE = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "035420": "NAVER",
    "068270": "셀트리온",
    "051910": "LG화학",
    "105560": "KB금융",
    "055550": "신한지주",
    "012330": "현대모비스",
    "066570": "LG전자",
    "003550": "LG",
    "028260": "삼성물산",
}


@lru_cache(maxsize=4)
def make_demo_ohlcv(periods: int = 760, seed: int = 42) -> pd.DataFrame:
    """Create deterministic, correlated synthetic daily OHLCV data.

    The data is intentionally synthetic and should never be interpreted as a
    historical quote source. It allows the full MVP to run without credentials.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    market = rng.normal(0.00025, 0.0105, periods)
    rows: list[pd.DataFrame] = []

    for idx, (ticker, name) in enumerate(DEMO_UNIVERSE.items()):
        base = 35_000 + idx * 11_500
        beta = 0.75 + (idx % 5) * 0.13
        idio = rng.normal(0, 0.010 + (idx % 4) * 0.0015, periods)
        momentum = np.zeros(periods)
        for t in range(1, periods):
            momentum[t] = 0.10 * momentum[t - 1] + market[t]
        returns = np.clip(0.00012 + beta * market + idio + 0.035 * momentum, -0.12, 0.12)
        close = base * np.exp(np.cumsum(returns))
        overnight = rng.normal(0, 0.0038, periods)
        open_ = close * np.exp(overnight - returns * 0.35)
        spread = np.abs(rng.normal(0.012, 0.004, periods))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        volume_base = 350_000 + idx * 95_000
        volume = rng.lognormal(np.log(volume_base), 0.42, periods) * (1 + np.abs(returns) * 8)

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "name": name,
                    "open": open_.round(0),
                    "high": high.round(0),
                    "low": low.round(0),
                    "close": close.round(0),
                    "volume": volume.astype(int),
                }
            )
        )

    return pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def demo_universe_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"종목코드": code, "종목명": name, "시장": "KOSPI (데모)"} for code, name in DEMO_UNIVERSE.items()]
    )

