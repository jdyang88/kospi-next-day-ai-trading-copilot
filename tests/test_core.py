import numpy as np
import pandas as pd

from src.backtest import walk_forward_backtest
from src.api.kis_adapter import KoreaInvestmentAdapter
from src.config import Settings
from src.data.demo import make_demo_ohlcv
from src.data.kis_live import (
    DEFAULT_WATCHLIST,
    apply_kis_quote_snapshot,
    build_analysis_universe,
    fetch_kis_daily_history,
    parse_watchlist,
)
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


class _FakeMarketData:
    def daily_prices(self, ticker, start, end):
        dates = pd.bdate_range(end=pd.Timestamp(end), periods=100)
        base = 50_000 if ticker == "005930" else 100_000
        return pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": base,
                "high": base + 2_000,
                "low": base - 2_000,
                "close": base + np.arange(100),
                "volume": 1_000_000 + np.arange(100),
            }
        )

    def current_price(self, ticker):
        return {
            "stck_oprc": "70000",
            "stck_hgpr": "72000",
            "stck_lwpr": "69000",
            "stck_prpr": "71500",
            "acml_vol": "2345678",
        }


def test_kis_history_and_live_snapshot_are_normalized():
    universe = {"005930": "삼성전자", "000660": "SK하이닉스"}
    history, history_errors = fetch_kis_daily_history(
        _FakeMarketData(), universe, max_pages=2, request_pause=0
    )
    live, quote_errors = apply_kis_quote_snapshot(
        history, _FakeMarketData(), universe, request_pause=0
    )
    assert not history_errors
    assert not quote_errors
    assert set(live.columns) == {"date", "ticker", "name", "open", "high", "low", "close", "volume"}
    latest = live.sort_values("date").groupby("ticker").tail(1)
    assert (latest["close"] == 71_500).all()
    assert (latest["volume"] == 2_345_678).all()


def test_watchlists_are_validated_deduplicated_and_capped():
    parsed, errors = parse_watchlist(
        "005930:삼성전자,086790:하나금융지주\n잘못된값;009150:삼성전기"
    )
    assert parsed["086790"] == "하나금융지주"
    assert len(errors) == 1
    universe, truncated = build_analysis_universe(parsed, {"005930": "삼성전자(관심)"})
    assert universe == {
        "005930": "삼성전자(관심)",
        "086790": "하나금융지주",
        "009150": "삼성전기",
    }
    assert truncated == 0
    oversized = {f"{index:06d}": f"종목{index}" for index in range(35)}
    universe, truncated = build_analysis_universe(oversized, max_size=11)
    assert len(universe) == 11
    assert truncated == 24


def test_default_watchlist_is_the_requested_editable_ten():
    assert DEFAULT_WATCHLIST == {
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
    }


class _PartiallyFailingMarketData(_FakeMarketData):
    def current_price(self, ticker):
        if ticker == "000660":
            raise RuntimeError("일시 오류")
        return super().current_price(ticker)


def test_failed_live_quote_is_excluded_from_analysis():
    universe = {"005930": "삼성전자", "000660": "SK하이닉스"}
    history, _ = fetch_kis_daily_history(
        _FakeMarketData(), universe, max_pages=1, request_pause=0
    )
    live, errors = apply_kis_quote_snapshot(
        history, _PartiallyFailingMarketData(), universe, request_pause=0
    )
    assert set(live["ticker"]) == {"005930"}
    assert "000660" in errors


class _PreMarketData(_FakeMarketData):
    def current_price(self, ticker):
        return {
            "stck_oprc": "0",
            "stck_hgpr": "0",
            "stck_lwpr": "0",
            "stck_prpr": "71500",
            "acml_vol": "0",
        }


def test_pre_market_zero_fields_use_latest_confirmed_daily_bar():
    universe = {"005930": "삼성전자"}
    history, _ = fetch_kis_daily_history(
        _FakeMarketData(), universe, max_pages=2, request_pause=0
    )
    live, errors = apply_kis_quote_snapshot(
        history,
        _PreMarketData(),
        universe,
        request_pause=0,
        now=pd.Timestamp("2026-08-18 08:00:00", tz="Asia/Seoul"),
    )
    latest = live.sort_values("date").iloc[-1]
    assert not errors
    assert latest["close"] == 71_500
    assert latest["open"] == 50_000
    assert latest["volume"] > 0
