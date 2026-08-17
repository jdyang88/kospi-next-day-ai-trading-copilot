import numpy as np

from src.backtest import walk_forward_backtest
from src.api.kis_adapter import KoreaInvestmentAdapter
from src.config import Settings
from src.data.demo import make_demo_ohlcv
from src.features import FEATURE_COLUMNS, add_technical_features, latest_feature_rows
from src.model import train_direction_model
from src.ranking import rank_stocks


def test_feature_pipeline_and_latest_rows():
    raw = make_demo_ohlcv(periods=300)
    featured = add_technical_features(raw)
    latest = latest_feature_rows(featured)
    assert len(latest) == raw["ticker"].nunique()
    assert set(FEATURE_COLUMNS).issubset(featured.columns)
    assert np.isfinite(latest[FEATURE_COLUMNS].to_numpy()).all()


def test_model_ranking_outputs_risk_levels():
    featured = add_technical_features(make_demo_ohlcv(periods=420))
    result = train_direction_model(featured, validation_days=60)
    ranked = rank_stocks(featured, result)
    assert len(ranked) == 5
    assert ranked["probability"].between(0, 1).all()
    assert (ranked["target_price"] > ranked["entry"]).all()
    assert (ranked["stop_price"] < ranked["entry"]).all()


def test_walk_forward_backtest_has_costs_and_equity():
    featured = add_technical_features(make_demo_ohlcv(periods=360))
    result = walk_forward_backtest(featured, test_days=30, retrain_every=10)
    assert not result.equity.empty
    assert result.metrics["cost_per_trade"] > 0
    assert {"strategy_equity", "benchmark_equity"}.issubset(result.equity.columns)


def test_order_submission_is_always_disabled():
    adapter = KoreaInvestmentAdapter(Settings())
    try:
        adapter.submit_order(ticker="005930", quantity=1)
        raise AssertionError("주문 차단이 작동하지 않았습니다.")
    except PermissionError:
        pass
