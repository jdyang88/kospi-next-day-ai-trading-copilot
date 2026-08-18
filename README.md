# KOSPI Next-Day AI Trading Copilot

한국 개인 투자자를 위한 **다음 거래일 방향 확률 기반 의사결정 보조 MVP**입니다. Python + Streamlit으로 동작하며, 한국투자증권 자격증명 없이도 합성 데모 일봉 데이터로 즉시 실행됩니다.

> 이 프로젝트의 출력은 확률적 참고 정보이며 투자 수익을 보장하지 않습니다. 자동매매는 코드 수준에서 비활성화되어 있습니다.

## 공개 데모

[Streamlit Community Cloud에서 앱 실행하기](https://kospi-next-day-ai-trading-copilot-lfyci4mtch4f6djvy6mzyk.streamlit.app/)

## 주요 기능

- KOSPI 데모 또는 사용자가 편집하는 KIS 관심종목 실시간 스냅샷 스캔 및 Top 5 랭킹
- 수익률, 이동평균, RSI, MACD, 볼린저 밴드, ATR, 거래량 비율, 상대강도 특징
- LightGBM → XGBoost → RandomForest 순서의 견고한 모델 폴백
- 상승 확률, 진입 기준, ATR 기반 목표가·손절가, 신뢰도, 선정 이유
- 확장형 학습 구간, 1거래일 엠바고, 수수료·슬리피지를 반영한 워크포워드 백테스트
- 로컬 CSV 매매일지
- 한국투자증권 Open API 읽기 전용 어댑터와 paper/live 환경 분리
- 관심종목을 자유롭게 추가·삭제하고 최대 30종목 분석

## Windows 설치 및 실행

PowerShell에서 프로젝트 폴더로 이동한 뒤 실행합니다.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

실행 후 브라우저에서 `http://localhost:8501`을 엽니다. PowerShell 실행 정책 때문에 가상환경 활성화가 막히면 활성화 없이 다음처럼 실행할 수 있습니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## 한국투자증권 API 설정(선택)

`.env.example`을 `.env`로 복사하고 값을 입력합니다.

```powershell
Copy-Item .env.example .env
```

```dotenv
APP_KEY=발급받은_앱키
APP_SECRET=발급받은_앱시크릿
ACCOUNT_NO=계좌번호_앞자리
ACCOUNT_PRODUCT_CODE=01
KIS_MODE=paper
WATCHLIST=005930:삼성전자,000660:SK하이닉스,051910:LG화학,096770:SK이노베이션,005380:현대차,069500:KODEX 200,035420:NAVER,034730:SK,000720:현대건설,068270:셀트리온,006260:LS,207940:삼성바이오로직스
```

추천 UI는 기본적으로 데모 데이터로 안전하게 실행됩니다. Secrets가 설정된 경우 사이드바에서 **KIS 실시간 → 실시간 추천 데이터 불러오기**를 누르면 현재 관심종목의 KIS 과거 일봉과 현재 OHLCV 스냅샷으로 특징·모델·Top 5를 다시 계산합니다. 최초 일봉 수집은 수 초 이상 걸릴 수 있으며 6시간 캐시되고, 현재 시세는 버튼을 누를 때 갱신됩니다. 일부 종목 조회가 실패해도 성공한 종목으로 분석을 계속합니다.

화면의 상승 확률은 오늘까지의 기술지표를 바탕으로 **다음 거래일 종가가 오늘 종가보다 높을 모델 추정 확률**입니다. 예상 수익률이나 상승 폭을 뜻하지 않습니다. 모델 검증 영역에서는 최근 기간을 시간순으로 분리해 측정한 AUC와 50% 기준 방향 정확도, 실제 사용 모델 및 학습·검증 기준일을 확인할 수 있습니다.

추천 카드의 `상승확률 단계`는 확률을 읽기 쉽게 나눈 4단계 보조 표시입니다. `1/4 관찰(54% 이하)`, `2/4 보통(54% 초과, 62% 이하)`, `3/4 높음(62% 초과, 70% 이하)`, `4/4 매우 높음(70% 초과)` 순이며, 별도의 모델 신뢰성이나 수익 보장을 뜻하지 않습니다.

모델 검증 아래의 지표 그래프는 각 기술지표가 모델의 판단에서 차지한 상대적 중요도 비중을 보여줍니다. 막대가 길다고 상승 방향으로 작용했다는 뜻은 아닙니다. `분석종목 평균 흐름`은 현재 분석 대상 종목을 동일 비중으로 평균하고 최근 120거래일의 첫날을 100으로 환산한 합성 흐름이며, 실제 KOSPI 지수가 아닙니다.

KIS 모드를 선택했지만 조회가 실패했거나 아직 실행하지 않은 경우에는 합성 데모 추천으로 대체하지 않습니다. 실제 KIS 데이터가 준비될 때까지 추천 종목과 가격을 숨기며, 추천 카드의 진입 기준에는 `KIS 시세` 또는 `합성 데모` 출처를 표시합니다.

KIS 일봉은 종목별로 성공한 결과만 6시간 캐시합니다. 실패한 종목은 캐시하지 않아 다음 버튼 클릭에서 다시 조회하며, 현재가는 매번 새로 조회합니다. 분석 최소 종목 수를 충족하지 못하더라도 실패 종목과 API 오류 사유를 화면에서 확인할 수 있습니다.

한국 장 시작 전이나 주말에는 KIS 현재가 응답의 당일 시가·고가·저가·누적거래량이 0일 수 있습니다. 이때는 최신 확정 KIS 일봉을 사용하며 당일 거래 행을 새로 만들지 않습니다. 장 시작 이후에는 실제 당일 OHLCV 스냅샷을 사용합니다.

새 세션의 초기 관심종목은 삼성전자, SK하이닉스, LG화학, SK이노베이션, 현대차, KODEX 200, NAVER, SK, 현대건설, 셀트리온, LS, 삼성바이오로직스의 12종목입니다. 앱 사이드바의 **관심종목 설정**에서 `종목코드:종목명` 줄을 추가하거나 삭제한 뒤 적용할 수 있습니다. 호출 안정성을 위해 한 번에 최대 30종목을 분석합니다. 앱에서 편집한 목록은 현재 브라우저 세션에만 유지되므로 영구 목록 전체는 Streamlit Secrets의 `WATCHLIST`에 저장하세요.

Streamlit Community Cloud에서는 같은 이름의 앱 Secrets를 읽습니다. `src/api/kis_adapter.py`는 인증·현재가·일봉 조회만 제공하고 `submit_order()`는 항상 예외를 발생시켜 자동주문을 차단합니다.

## 데이터와 15:10 제약

무료 소스에서 장기간의 KOSPI 전 종목 **15:10 장중 스냅샷 이력**을 안정적으로 구하기 어렵기 때문에 데모 백테스트는 합성 일봉 OHLCV를 사용합니다. KIS 모드의 추천은 현재 스냅샷을 반영하지만, 장기간의 15:10 전용 학습 이력이 쌓이기 전까지 모델은 일봉 기반입니다. 따라서 성과 수치를 실제 투자 성과로 해석하면 안 됩니다.

향후 구조는 다음 확장을 염두에 둡니다.

1. KIS WebSocket으로 실시간 체결·호가 수신
2. 매 거래일 15:10에 전 종목 특징 스냅샷 저장
3. 다음 거래일 결과가 확정된 뒤 라벨 생성
4. 동일한 `features → model → ranking → backtest` 인터페이스로 재학습

한국투자증권의 공식 예제 저장소에는 국내주식 REST 및 WebSocket 샘플이 제공됩니다: [KIS Open API 공식 GitHub](https://github.com/koreainvestment/open-trading-api)

## 프로젝트 구조

```text
.
├── app.py
├── data/
│   ├── demo/README.md
│   └── trade_journal.csv
├── src/
│   ├── api/kis_adapter.py
│   ├── data/demo.py
│   ├── data/kis_live.py
│   ├── ui/styles.py
│   ├── backtest.py
│   ├── config.py
│   ├── features.py
│   ├── journal.py
│   ├── model.py
│   └── ranking.py
├── tests/test_core.py
├── .env.example
└── requirements.txt
```

## 검증

```powershell
pip install pytest
pytest -q
```

백테스트는 교육용 구현입니다. 생존편향, 상장폐지, 거래정지, 가격제한폭, 체결 가능 수량, 세금, 배당·분할 등 실제 시장의 모든 요소를 완전히 반영하지 않습니다.
