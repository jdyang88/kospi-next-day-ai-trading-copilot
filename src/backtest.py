from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.model import _build_model


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def walk_forward_backtest(
    featured: pd.DataFrame,
    test_days: int = 140,
    top_n: int = 5,
    retrain_every: int = 20,
    commission_bps: float = 8.0,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    """Expanding-window, close-to-next-close backtest with explicit costs.

    For each test date the model is trained only on labels whose outcome was
    already known before that date. Predictions never use future rows.
    """
    data = featured.dropna(subset=FEATURE_COLUMNS + ["target", "next_return"]).sort_values(["date", "ticker"])
    dates = np.sort(data["date"].unique())
    test_dates = dates[-min(test_days, max(1, len(dates) - 160)) :]
    cost = 2 * commission_bps / 10_000 + 2 * slippage_bps / 10_000
    daily_records: list[dict] = []
    trade_records: list[dict] = []
    model = None

    for i, raw_date in enumerate(test_dates):
        current_date = pd.Timestamp(raw_date)
        # One-day embargo: the label at t-1 depends on close(t), so only t-2 and earlier are known.
        eligible_dates = dates[dates < raw_date]
        if len(eligible_dates) < 2:
            continue
        train_cutoff = eligible_dates[-1]
        train = data[data["date"] < train_cutoff]
        today = data[data["date"] == current_date]
        if len(train) < 300 or today.empty:
            continue
        if model is None or i % retrain_every == 0:
            model, _ = _build_model(random_state=100 + i)
            model.fit(train[FEATURE_COLUMNS], train["target"].astype(int))
        today = today.copy()
        today["probability"] = model.predict_proba(today[FEATURE_COLUMNS])[:, 1]
        selected = today.nlargest(top_n, "probability")
        gross = float(selected["next_return"].mean())
        net = gross - cost
        benchmark = float(today["next_return"].mean())
        daily_records.append({"date": current_date, "strategy_return": net, "benchmark_return": benchmark})
        for _, row in selected.iterrows():
            trade_records.append(
                {
                    "date": current_date,
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "probability": row["probability"],
                    "gross_return": row["next_return"],
                    "net_return": row["next_return"] - cost,
                }
            )

    returns = pd.DataFrame(daily_records)
    if returns.empty:
        raise ValueError("백테스트 결과를 만들 수 없습니다.")
    returns["strategy_equity"] = (1 + returns["strategy_return"]).cumprod()
    returns["benchmark_equity"] = (1 + returns["benchmark_return"]).cumprod()
    drawdown = returns["strategy_equity"] / returns["strategy_equity"].cummax() - 1
    trades = pd.DataFrame(trade_records)
    annualized = returns["strategy_equity"].iloc[-1] ** (252 / len(returns)) - 1
    volatility = returns["strategy_return"].std() * np.sqrt(252)
    metrics = {
        "total_return": float(returns["strategy_equity"].iloc[-1] - 1),
        "annualized_return": float(annualized),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns["strategy_return"] > 0).mean()),
        "sharpe": float(annualized / volatility) if volatility > 0 else 0.0,
        "cost_per_trade": float(cost),
        "trade_count": float(len(trades)),
    }
    return BacktestResult(equity=returns, trades=trades, metrics=metrics)

