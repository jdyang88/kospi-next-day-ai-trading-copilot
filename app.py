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
from src.data.kis_live import KIS_STARTER_UNIVERSE, apply_kis_quote_snapshot, fetch_kis_daily_history
from src.features import FEATURE_COLUMNS, add_technical_features, latest_feature_rows
from src.journal import append_trade, load_journal
from src.model import train_direction_model
from src.ranking import rank_stocks
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
def load_kis_history_data() -> tuple[pd.DataFrame, dict[str, str]]:
    return fetch_kis_daily_history(load_kis_adapter())


@st.cache_data(ttl=30, show_spinner=False)
def load_kis_snapshot(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    return apply_kis_quote_snapshot(history, load_kis_adapter())


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


def recommendation_row(rank: int, row: pd.Series) -> None:
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
            st.markdown(f'<span class="label">진입 기준</span><span class="value">{money(row["entry"])}</span>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'<span class="label">목표가</span><span class="value target">{money(row["target_price"])}</span>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<span class="label">손절가</span><span class="value stop">{money(row["stop_price"])}</span>', unsafe_allow_html=True)
        with cols[5]:
            st.markdown(f'<span class="label">신뢰도</span><span class="confidence">{row["confidence"]}</span>', unsafe_allow_html=True)
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

    ranked = rank_stocks(featured, model_result)
    for idx, row in ranked.iterrows():
        recommendation_row(idx + 1, row)

    st.caption("진입 기준은 최신 종가이며 목표가·손절가는 ATR 기반 참고 범위입니다. 장중 갭과 유동성 위험은 별도로 확인하세요.")
    st.divider()
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("모델 검증")
        m1, m2, m3 = st.columns(3)
        m1.metric("검증 AUC", f"{model_result.auc:.3f}")
        m2.metric("방향 정확도", pct(model_result.accuracy))
        m3.metric("사용 모델", model_result.model_name)
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


def scanner_page(featured: pd.DataFrame, model_result, data_label: str) -> None:
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
                [{"종목코드": ticker, "종목명": name} for ticker, name in KIS_STARTER_UNIVERSE.items()]
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


def journal_page() -> None:
    st.header("매매일지")
    st.caption("추천을 실제로 검토하거나 매매한 기록을 로컬 CSV에 저장합니다.")
    with st.form("journal_form", clear_on_submit=True):
        cols = st.columns(4)
        trade_date = cols[0].date_input("일자", value=date.today())
        journal_universe = {**DEMO_UNIVERSE, **KIS_STARTER_UNIVERSE}
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
    st.code("APP_KEY=\nAPP_SECRET=\nACCOUNT_NO=\nACCOUNT_PRODUCT_CODE=01\nKIS_MODE=paper", language="bash")
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
        if source == "KIS 실시간":
            if st.button("실시간 추천 데이터 불러오기", type="primary", use_container_width=True):
                try:
                    with st.spinner("KIS 일봉과 현재 시세를 읽고 있습니다..."):
                        history, history_errors = load_kis_history_data()
                        live_raw, quote_errors = load_kis_snapshot(history)
                        if live_raw.empty or live_raw["ticker"].nunique() < 3:
                            raise RuntimeError("분석 가능한 종목이 3개 미만입니다.")
                        st.session_state["kis_live_raw"] = live_raw
                        st.session_state["kis_live_errors"] = {**history_errors, **quote_errors}
                        st.session_state["kis_live_updated_at"] = datetime.now(ZoneInfo("Asia/Seoul"))
                    st.success("실시간 추천 데이터를 불러왔습니다.")
                except Exception as exc:
                    st.error(f"KIS 데이터 조회 실패: {exc}")
            updated_at = st.session_state.get("kis_live_updated_at")
            if updated_at:
                st.caption(f"최근 조회: {updated_at:%Y-%m-%d %H:%M:%S} KST")
            else:
                st.info("위 버튼을 눌러야 KIS 시세가 추천에 반영됩니다.")
        st.markdown("---")
        st.caption("v1.1.0 · 읽기 전용")
        st.markdown('<div class="mode-note">일봉 모델 + 장중 스냅샷<br>자동주문 영구 비활성화</div>', unsafe_allow_html=True)

    use_live = source == "KIS 실시간" and "kis_live_raw" in st.session_state
    if use_live:
        raw = st.session_state["kis_live_raw"]
        data_label = "KIS 실시간 시세"
        with st.spinner("KIS 시장 데이터를 분석하고 있습니다..."):
            featured = add_technical_features(raw)
    else:
        data_label = "합성 데모" if source == "데모" else "합성 데모 미리보기"
        with st.spinner("데모 시장을 분석하고 있습니다..."):
            _, featured = load_demo_data()

    header(data_label)
    disclaimer(data_label)
    live_errors = st.session_state.get("kis_live_errors", {}) if use_live else {}
    if live_errors:
        st.warning(f"일부 종목 조회 실패: {len(live_errors)}개. 성공한 종목만 분석했습니다.")

    model_result = load_model(featured)

    if page == "오늘의 추천":
        recommendations_page(featured, model_result, data_label)
    elif page == "종목 스캐너":
        scanner_page(featured, model_result, data_label)
    elif page == "백테스트":
        backtest_page(featured)
    elif page == "매매일지":
        journal_page()
    else:
        settings_page(settings, data_label)


if __name__ == "__main__":
    main()
