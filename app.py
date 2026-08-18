from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.api.kis_adapter import KoreaInvestmentAdapter
from src.backtest import walk_forward_backtest
from src.config import JOURNAL_PATH, load_settings
from src.data.demo import DEMO_UNIVERSE, demo_universe_table, make_demo_ohlcv
from src.data.kis_live import (
    DEFAULT_WATCHLIST,
    apply_kis_quote_snapshot,
    build_analysis_universe,
    fetch_kis_daily_history,
    parse_watchlist,
)
from src.features import FEATURE_COLUMNS, add_technical_features, latest_feature_rows
from src.journal import append_trade, load_journal
from src.model import train_direction_model
from src.ranking import CONFIDENCE_GUIDE, rank_stocks
from src.ui.styles import APP_CSS


st.set_page_config(
    page_title="KOSPI Next-Day AI Trading Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = make_demo_ohlcv()
    return raw, add_technical_features(raw)


@st.cache_resource(show_spinner=False)
def load_kis_adapter() -> KoreaInvestmentAdapter:
    # Reuse one read-only adapter so KIS token issuance is not repeated between
    # historical and current-price requests.
    return KoreaInvestmentAdapter(load_settings())


@st.cache_data(ttl=21_600, show_spinner=False)
def load_kis_ticker_history(ticker: str, name: str) -> pd.DataFrame:
    """Cache only a successful symbol; raised failures are never cached."""
    frame, errors = fetch_kis_daily_history(
        load_kis_adapter(),
        {ticker: name},
        max_pages=3,
    )
    if ticker in errors:
        raise RuntimeError(errors[ticker])
    if frame["date"].nunique() < 180:
        raise RuntimeError(f"모델 학습용 일봉이 부족합니다 ({frame['date'].nunique()}일).")
    return frame


def load_kis_history_data(
    universe_items: tuple[tuple[str, str], ...],
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for ticker, name in universe_items:
        try:
            frames.append(load_kis_ticker_history(ticker, name))
        except Exception as exc:
            errors[ticker] = str(exc)
    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", "name", "open", "high", "low", "close", "volume"]
        ), errors
    return pd.concat(frames, ignore_index=True), errors


def load_kis_snapshot(
    history: pd.DataFrame, universe_items: tuple[tuple[str, str], ...]
) -> tuple[pd.DataFrame, dict[str, str]]:
    # The button is an explicit refresh action, so current quotes are never cached.
    return apply_kis_quote_snapshot(history, load_kis_adapter(), dict(universe_items))


@st.cache_resource(show_spinner=False)
def load_model(featured: pd.DataFrame):
    return train_direction_model(featured)


@st.cache_data(show_spinner=False)
def run_backtest(featured: pd.DataFrame, commission_bps: float, slippage_bps: float):
    return walk_forward_backtest(
        featured, commission_bps=commission_bps, slippage_bps=slippage_bps, test_days=140
    )


def money(value: float) -> str:
    return f"{value:,.0f}원"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def header(data_label: str) -> None:
    left, right = st.columns([4, 1.25], vertical_alignment="center")
    with left:
        st.title("KOSPI Next-Day AI Trading Copilot")
        st.markdown(
            '<div class="app-subtitle">KOSPI 다음 거래일 방향을 확률로 분석하는 의사결정 보조 도구</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="status-line"><span class="status-dot"></span>{data_label} · 자동매매 꺼짐</div>',
            unsafe_allow_html=True,
        )


def disclaimer(data_label: str) -> None:
    source_note = "실시간·과거 KIS 시세" if data_label.startswith("KIS") else "합성 일봉 데이터"
    st.markdown(
        f'<div class="disclaimer"><strong>중요:</strong> 모든 결과는 {source_note}에서 얻은 확률 기반 참고 정보이며, 투자 수익을 보장하지 않습니다. 실제 주문은 사용자가 별도로 판단하고 실행해야 합니다.</div>',
        unsafe_allow_html=True,
    )


def render_kis_errors(errors: dict[str, str], universe: dict[str, str]) -> None:
    if not errors:
        return
    st.warning(f"조회 실패 종목: {len(errors)}개")
    with st.expander("실패한 종목과 상세 사유", expanded=True):
        error_rows = [
            {
                "종목코드": ticker,
                "종목명": universe.get(ticker, "시장 데이터"),
                "사유": message,
            }
            for ticker, message in errors.items()
        ]
        st.dataframe(pd.DataFrame(error_rows), width="stretch", hide_index=True)


def recommendation_row(rank: int, row: pd.Series, price_source: str) -> None:
    with st.container(border=True):
        cols = st.columns([2.0, 1.0, 1.05, 1.05, 1.05, 1.2, 3.0], vertical_alignment="center")
        with cols[0]:
            st.markdown(
                f'<span class="rank">{rank}</span><span class="stock-name">{row["name"]}</span><span class="ticker">{row["ticker"]}</span>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(f'<span class="label">상승 확률</span><span class="prob">{pct(row["probability"], 0)}</span>', unsafe_allow_html=True)
            st.progress(float(row["probability"]))
        with cols[2]:
            st.markdown(
                f'<span class="label">진입 기준 · {price_source}</span><span class="value">{money(row["entry"])}</span>',
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.markdown(f'<span class="label">목표가</span><span class="value target">{money(row["target_price"])}</span>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<span class="label">손절가</span><span class="value stop">{money(row["stop_price"])}</span>', unsafe_allow_html=True)
        with cols[5]:
            st.markdown(
                f'<span class="label">상승확률 단계</span><span class="confidence">{row["confidence_level"]}/4 · {row["confidence"]}</span>',
                unsafe_allow_html=True,
            )
        with cols[6]:
            st.markdown(f'<span class="label">선정 이유</span><span class="reason">{row["reason"]}</span>', unsafe_allow_html=True)


def recommendations_page(featured: pd.DataFrame, model_result, data_label: str) -> None:
    title_col, action_col = st.columns([4.5, 1], vertical_alignment="center")
    with title_col:
        st.markdown(
            f'<div class="section-head"><strong>오늘의 Top 5</strong><span>{data_label} 기준</span></div>',
            unsafe_allow_html=True,
        )
    with action_col:
        if st.button("분석 새로고침", type="primary", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    with st.expander("상승 확률은 어떻게 계산하나요?"):
        st.markdown(
            """
- **예측 대상:** 오늘까지의 정보로 **다음 거래일 종가가 오늘 종가보다 높을 확률**을 추정합니다.
- **입력 정보:** 1·5·20일 수익률, 이동평균선과의 괴리, RSI, MACD, 볼린저밴드, ATR, 거래량 비율, 시장 대비 상대강도를 사용합니다.
- **학습·검증:** 과거 데이터는 시간순으로 나누며, 최근 검증 구간을 학습에서 제외해 미래 정보를 미리 보는 오류를 방지합니다.
- **표시 방법:** 모델의 상승 확률을 기준으로 종목을 정렬해 상위 5개를 보여줍니다.
- **확률 단계:** 상승확률을 읽기 쉽게 4개 구간으로 나눈 보조 표시이며, 별도로 계산한 모델 신뢰성이나 수익 보장이 아닙니다.

상승 확률은 **예상 수익률이나 수익 보장 수치가 아닙니다.** 예를 들어 60%는 과거 패턴을 학습한 모델이 상승 쪽에 0.60의 확률을 부여했다는 뜻이며, 상승 폭의 크기는 나타내지 않습니다.
            """
        )

    ranked = rank_stocks(featured, model_result)
    price_source = "KIS 시세" if data_label.startswith("KIS") else "합성 데모"
    for idx, row in ranked.iterrows():
        recommendation_row(idx + 1, row, price_source)

    st.caption(
        f"상승확률 단계: {CONFIDENCE_GUIDE}. "
        "단계가 높을수록 모델의 상승확률이 높지만 실제 수익을 보장하지 않습니다."
    )
    st.caption("진입 기준은 최신 종가이며 목표가·손절가는 ATR 기반 참고 범위입니다. 장중 갭과 유동성 위험은 별도로 확인하세요.")
    st.divider()
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("모델 검증")
        m1, m2, m3 = st.columns(3)
        model_display_name = {
            "RandomForest (fallback)": "RF",
        }.get(model_result.model_name, model_result.model_name)
        m1.metric(
            "검증 AUC",
            f"{model_result.auc:.3f}",
            help="상승 종목을 하락 종목보다 높게 평가하는 능력입니다. 0.5는 무작위 수준, 1.0은 이상적인 구분을 뜻합니다.",
        )
        m2.metric(
            "방향 정확도",
            pct(model_result.accuracy),
            help="검증 구간에서 상승 확률 50%를 기준으로 상승·하락 방향을 맞힌 비율입니다.",
        )
        m3.metric(
            "사용 모델",
            model_display_name,
            help="LightGBM을 우선 사용하며, 설치되지 않은 경우 XGBoost, RandomForest 순으로 대체합니다.",
        )
        st.caption(
            "AUC는 종목의 상대적 순위 구분 능력, 방향 정확도는 50% 기준의 정답 비율입니다. "
            f"현재 사용 모델은 {model_result.model_name}이며, 실행 환경에서 실제 학습에 사용된 알고리즘입니다."
        )
        st.caption(
            f"시간순 검증 · 학습 데이터 종료 {model_result.train_end:%Y-%m-%d} · "
            f"검증 시작 {model_result.validation_start:%Y-%m-%d}"
        )
        importance = model_result.feature_importance.head(8).sort_values("importance")
        fig = px.bar(importance, x="importance", y="feature", orientation="h", color_discrete_sequence=["#087443"])
        fig.update_layout(height=310, margin=dict(l=0, r=10, t=15, b=0), xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("최근 시장 흐름")
        market = featured.pivot(index="date", columns="ticker", values="close").pct_change(fill_method=None).mean(axis=1)
        index = (1 + market.fillna(0)).cumprod().tail(120) * 100
        fig = go.Figure(go.Scatter(x=index.index, y=index, mode="lines", line=dict(color="#087443", width=2), fill="tozeroy", fillcolor="rgba(8,116,67,.08)"))
        fig.update_layout(height=310, margin=dict(l=0, r=10, t=15, b=0), yaxis_title="합성 시장지수", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def scanner_page(
    featured: pd.DataFrame,
    model_result,
    data_label: str,
    analysis_universe: dict[str, str] | None = None,
) -> None:
    st.header("종목 스캐너")
    st.caption(f"{data_label} 유니버스의 최신 기술지표와 예측 점수를 확인합니다.")
    rows = latest_feature_rows(featured)
    rows["상승확률"] = model_result.model.predict_proba(rows[FEATURE_COLUMNS])[:, 1]
    view = rows[["ticker", "name", "close", "상승확률", "ret_20d", "rsi_14", "volume_ratio_20", "relative_strength_20"]].copy()
    view.columns = ["종목코드", "종목명", "종가", "상승확률", "20일수익률", "RSI(14)", "거래량비율", "상대강도"]
    threshold = st.slider("최소 상승 확률", 0.40, 0.80, 0.50, 0.01)
    view = view[view["상승확률"] >= threshold].sort_values("상승확률", ascending=False)
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "종가": st.column_config.NumberColumn(format="%,.0f원"),
            "상승확률": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.1%%"),
            "20일수익률": st.column_config.NumberColumn(format="%.2%%"),
            "RSI(14)": st.column_config.NumberColumn(format="%.2f"),
            "거래량비율": st.column_config.NumberColumn(format="%.2fx"),
            "상대강도": st.column_config.NumberColumn(format="%.2%%"),
        },
    )
    with st.expander("분석 유니버스 보기"):
        if data_label.startswith("KIS"):
            universe = pd.DataFrame(
                [
                    {"종목코드": ticker, "종목명": name}
                    for ticker, name in (analysis_universe or DEFAULT_WATCHLIST).items()
                ]
            )
            st.dataframe(universe, use_container_width=True, hide_index=True)
        else:
            st.dataframe(demo_universe_table(), use_container_width=True, hide_index=True)


def backtest_page(featured: pd.DataFrame) -> None:
    st.header("워크포워드 백테스트")
    st.caption("매 시점에 과거에 확정된 정보만 사용하며, 다음날 종가에 청산하는 MVP 일봉 전략입니다.")
    c1, c2 = st.columns(2)
    commission = c1.number_input("왕복 수수료 (bp, 편도)", 0.0, 50.0, 8.0, 1.0)
    slippage = c2.number_input("왕복 슬리피지 (bp, 편도)", 0.0, 50.0, 5.0, 1.0)
    with st.spinner("미래정보를 차단한 워크포워드 검증을 실행하고 있습니다..."):
        result = run_backtest(featured, commission, slippage)
    metrics = result.metrics
    cols = st.columns(5)
    cols[0].metric("누적수익률", pct(metrics["total_return"]))
    cols[1].metric("최대낙폭", pct(metrics["max_drawdown"]))
    cols[2].metric("승률", pct(metrics["win_rate"]))
    cols[3].metric("샤프지수", f"{metrics['sharpe']:.2f}")
    cols[4].metric("거래비용", pct(metrics["cost_per_trade"], 2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity["date"], y=result.equity["strategy_equity"], name="전략 (비용 반영)", line=dict(color="#087443", width=2.4)))
    fig.add_trace(go.Scatter(x=result.equity["date"], y=result.equity["benchmark_equity"], name="동일가중 벤치마크", line=dict(color="#8090a8", dash="dot")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="누적 자산", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.info("15:10 장중 스냅샷의 장기 무료 이력이 없어 MVP는 일봉 종가를 사용합니다. 향후 KIS WebSocket 수집기를 연결하면 같은 인터페이스에 15:10 특징 스냅샷을 저장해 검증할 수 있습니다.")
    with st.expander("백테스트 방법론과 거래 내역"):
        st.markdown("- 확장형 학습 구간과 1거래일 엠바고\n- 20거래일마다 재학습\n- 매일 확률 상위 5종목 동일가중\n- 편도 수수료와 편도 슬리피지를 진입·청산 양쪽에 반영")
        st.dataframe(result.trades.tail(100), use_container_width=True, hide_index=True)


def journal_page(analysis_universe: dict[str, str] | None = None) -> None:
    st.header("매매일지")
    st.caption("추천을 실제로 검토하거나 매매한 기록을 로컬 CSV에 저장합니다.")
    with st.form("journal_form", clear_on_submit=True):
        cols = st.columns(4)
        trade_date = cols[0].date_input("일자", value=date.today())
        journal_universe = {
            **DEMO_UNIVERSE,
            **DEFAULT_WATCHLIST,
            **(analysis_universe or {}),
        }
        name = cols[1].selectbox("종목", list(dict.fromkeys(journal_universe.values())))
        side = cols[2].selectbox("구분", ["관찰", "매수", "매도"])
        quantity = cols[3].number_input("수량", min_value=0, value=0)
        cols2 = st.columns([1, 1, 2])
        price = cols2[0].number_input("체결/기준 가격", min_value=0, value=0, step=100)
        strategy = cols2[1].selectbox("전략", ["Next-Day Top 5", "직접 판단"])
        memo = cols2[2].text_input("메모")
        submitted = st.form_submit_button("기록 저장", type="primary")
        if submitted:
            ticker = next(code for code, stock_name in journal_universe.items() if stock_name == name)
            append_trade(JOURNAL_PATH, {"date": trade_date.isoformat(), "ticker": ticker, "name": name, "side": side, "quantity": quantity, "price": price, "strategy": strategy, "memo": memo})
            st.success("매매일지를 저장했습니다.")
    journal = load_journal(JOURNAL_PATH).sort_values("date", ascending=False)
    st.dataframe(journal, use_container_width=True, hide_index=True)
    if JOURNAL_PATH.exists():
        st.download_button("CSV 내려받기", JOURNAL_PATH.read_bytes(), file_name="trade_journal.csv", mime="text/csv")


def settings_page(settings, data_label: str) -> None:
    st.header("데이터 및 API 설정")
    adapter = load_kis_adapter()
    c1, c2, c3 = st.columns(3)
    c1.metric("현재 데이터", data_label)
    c2.metric("KIS 자격증명", "설정됨" if settings.credentials_ready else "미설정")
    c3.metric("주문 기능", "영구 비활성화")
    st.subheader("환경변수")
    st.code(
        "APP_KEY=\nAPP_SECRET=\nACCOUNT_NO=\nACCOUNT_PRODUCT_CODE=01\nKIS_MODE=paper\nWATCHLIST=005930:삼성전자,000660:SK하이닉스,051910:LG화학,096770:SK이노베이션,005380:현대차,069500:KODEX 200,035420:NAVER,034730:SK,000720:현대건설,068270:셀트리온,006260:LS,207940:삼성바이오로직스",
        language="bash",
    )
    st.write(f"선택된 연결 모드: **{adapter.mode_label}**")
    st.warning("실전 모드를 선택해도 이 프로젝트는 주문 API를 호출하지 않습니다. 실제 주문은 한국투자증권 앱에서 직접 실행하세요.")

    st.subheader("API 연결 확인")
    st.caption("삼성전자(005930) 현재가를 한 번 조회해 인증과 시세 연결만 확인합니다. 주문은 발생하지 않습니다.")
    if st.button("한국투자증권 API 연결 테스트", type="primary", disabled=not settings.credentials_ready):
        try:
            with st.spinner("한국투자증권 시세 서버에 연결하고 있습니다..."):
                quote = adapter.current_price("005930")
            current_price = pd.to_numeric(quote.get("stck_prpr"), errors="coerce")
            if pd.isna(current_price):
                st.success(f"API 인증 성공 ({adapter.mode_label})")
                st.json({"종목코드": "005930", "응답": "시세 응답 수신"})
            else:
                st.success(f"API 연결 성공 · 삼성전자 현재가 {current_price:,.0f}원 ({adapter.mode_label})")
        except Exception as exc:
            st.error(f"연결 실패: {exc}")
            st.caption("모의투자 키에는 KIS_MODE=paper, 실전투자 키에는 KIS_MODE=live를 사용했는지 확인하세요.")

    st.subheader("15:10 연동 확장점")
    st.markdown("`src/api/kis_adapter.py`의 읽기 전용 REST 어댑터 옆에 WebSocket 수집기를 추가하고, 매 거래일 15:10 특징 스냅샷을 별도 저장소에 적재하도록 설계할 수 있습니다.")


def main() -> None:
    settings = load_settings()
    secret_watchlist, secret_watchlist_errors = parse_watchlist(settings.watchlist)
    if "session_watchlist" not in st.session_state:
        initial_watchlist = secret_watchlist or DEFAULT_WATCHLIST
        st.session_state["session_watchlist"] = dict(initial_watchlist)
        st.session_state["watchlist_editor"] = "\n".join(
            f"{ticker}:{name}" for ticker, name in initial_watchlist.items()
        )
    with st.sidebar:
        st.markdown("### 메뉴")
        page = st.radio("페이지", ["오늘의 추천", "종목 스캐너", "백테스트", "매매일지", "설정"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### 데이터 모드")
        source = st.radio(
            "데이터 모드",
            ["데모", "KIS 실시간"],
            horizontal=True,
            label_visibility="collapsed",
            disabled=not settings.credentials_ready,
        )
        if not settings.credentials_ready:
            st.caption("KIS Secrets를 설정하면 실시간 모드를 사용할 수 있습니다.")

        with st.expander("관심종목 설정"):
            with st.form("watchlist_form"):
                st.text_area(
                    "현재 관심종목",
                    placeholder="086790:하나금융지주\n009150:삼성전기",
                    help="종목코드:종목명 형식입니다. 줄을 추가하거나 삭제한 뒤 적용하세요.",
                    key="watchlist_editor",
                )
                apply_watchlist = st.form_submit_button("관심종목 적용", use_container_width=True)
            if apply_watchlist:
                session_watchlist, input_errors = parse_watchlist(st.session_state["watchlist_editor"])
                st.session_state["session_watchlist"] = session_watchlist
                st.session_state["watchlist_input_errors"] = input_errors
                if input_errors:
                    st.warning("일부 입력을 확인해 주세요.")
                else:
                    st.success(f"관심종목 {len(session_watchlist)}개를 적용했습니다.")

        session_watchlist = st.session_state.get("session_watchlist", {})
        active_universe, truncated = build_analysis_universe(session_watchlist)
        universe_items = tuple(active_universe.items())
        universe_key = universe_items
        st.caption(f"현재 분석 대상: {len(active_universe)}종목")
        input_errors = st.session_state.get("watchlist_input_errors", [])
        if secret_watchlist_errors or input_errors:
            st.warning("형식이 잘못된 관심종목은 제외했습니다.")
        if truncated:
            st.warning(f"호출 안정성을 위해 최대 30종목만 사용합니다. {truncated}종목은 제외됐습니다.")

        if source == "KIS 실시간":
            if st.button("실시간 추천 데이터 불러오기", type="primary", use_container_width=True):
                for key in (
                    "kis_live_raw",
                    "kis_live_errors",
                    "kis_live_updated_at",
                    "kis_live_universe_key",
                    "kis_load_error",
                    "kis_last_attempted_count",
                    "kis_last_success_count",
                ):
                    st.session_state.pop(key, None)
                try:
                    with st.spinner("KIS 일봉과 현재 시세를 읽고 있습니다..."):
                        history, history_errors = load_kis_history_data(universe_items)
                        live_raw, quote_errors = load_kis_snapshot(history, universe_items)
                        combined_errors = {**history_errors, **quote_errors}
                        successful_count = int(live_raw["ticker"].nunique()) if not live_raw.empty else 0
                        st.session_state["kis_live_errors"] = combined_errors
                        st.session_state["kis_last_attempted_count"] = len(active_universe)
                        st.session_state["kis_last_success_count"] = successful_count
                        if live_raw.empty or live_raw["ticker"].nunique() < 3:
                            raise RuntimeError("분석 가능한 종목이 3개 미만입니다.")
                        st.session_state["kis_live_raw"] = live_raw
                        st.session_state["kis_live_updated_at"] = datetime.now(ZoneInfo("Asia/Seoul"))
                        st.session_state["kis_live_universe_key"] = universe_key
                    st.success("실시간 추천 데이터를 불러왔습니다.")
                except Exception as exc:
                    st.session_state["kis_load_error"] = str(exc)
                    st.error(f"KIS 데이터 조회 실패: {exc}")
            updated_at = st.session_state.get("kis_live_updated_at")
            if updated_at:
                st.caption(f"최근 조회: {updated_at:%Y-%m-%d %H:%M:%S} KST")
                if st.session_state.get("kis_live_universe_key") != universe_key:
                    st.info("관심종목이 변경되었습니다. 실시간 데이터를 다시 불러오세요.")
            else:
                st.info("위 버튼을 눌러야 KIS 시세가 추천에 반영됩니다.")
        st.markdown("---")
        st.caption("v1.4.4 · 읽기 전용")
        st.markdown('<div class="mode-note">일봉 모델 + 장중 스냅샷<br>자동주문 영구 비활성화</div>', unsafe_allow_html=True)

    use_live = (
        source == "KIS 실시간"
        and "kis_live_raw" in st.session_state
        and st.session_state.get("kis_live_universe_key") == universe_key
    )
    if use_live:
        raw = st.session_state["kis_live_raw"]
        data_label = "KIS 최신 시세"
        with st.spinner("KIS 시장 데이터를 분석하고 있습니다..."):
            featured = add_technical_features(raw)
    elif source == "데모":
        data_label = "합성 데모"
        with st.spinner("데모 시장을 분석하고 있습니다..."):
            _, featured = load_demo_data()
    else:
        data_label = "KIS 데이터 없음"
        featured = None

    header(data_label)
    if featured is None:
        st.error("실시간 KIS 데이터가 준비되지 않아 추천 종목과 가격을 표시하지 않습니다.")
        load_error = st.session_state.get("kis_load_error")
        if load_error:
            st.warning(f"최근 조회 실패 사유: {load_error}")
        attempted_count = st.session_state.get("kis_last_attempted_count")
        if attempted_count is not None:
            successful_count = int(st.session_state.get("kis_last_success_count", 0))
            count_cols = st.columns(3)
            count_cols[0].metric("조회 시도", f"{attempted_count}종목")
            count_cols[1].metric("조회 성공", f"{successful_count}종목")
            count_cols[2].metric("조회 실패", f"{attempted_count - successful_count}종목")
        render_kis_errors(st.session_state.get("kis_live_errors", {}), active_universe)
        st.info(
            "사이드바에서 분석 대상을 확인한 뒤 '실시간 추천 데이터 불러오기'를 누르세요. "
            "성공한 종목의 일봉은 6시간 재사용하고 실패한 종목만 다음 클릭에서 다시 조회합니다."
        )
        if page == "매매일지":
            journal_page(active_universe)
        elif page == "설정":
            settings_page(settings, data_label)
        return

    disclaimer(data_label)
    live_errors = st.session_state.get("kis_live_errors", {}) if use_live else {}
    if use_live:
        successful_count = int(raw["ticker"].nunique())
        count_cols = st.columns(3)
        count_cols[0].metric("분석 대상", f"{len(active_universe)}종목")
        count_cols[1].metric("조회 성공", f"{successful_count}종목")
        count_cols[2].metric("조회 제외", f"{len(active_universe) - successful_count}종목")
    if live_errors:
        render_kis_errors(live_errors, active_universe)

    model_result = load_model(featured)

    if page == "오늘의 추천":
        recommendations_page(featured, model_result, data_label)
    elif page == "종목 스캐너":
        scanner_page(featured, model_result, data_label, active_universe)
    elif page == "백테스트":
        backtest_page(featured)
    elif page == "매매일지":
        journal_page(active_universe)
    else:
        settings_page(settings, data_label)


if __name__ == "__main__":
    main()
