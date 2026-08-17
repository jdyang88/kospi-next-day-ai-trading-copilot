# KOSPI Next-Day AI Trading Copilot

한국 개인 투자자를 위한 **다음 거래일 방향 확률 기반 의사결정 보조 MVP**입니다. Python + Streamlit으로 동작하며, 한국투자증권 자격증명 없이도 합성 데모 일봉 데이터로 즉시 실행됩니다.

> 이 프로젝트의 출력은 확률적 참고 정보이며 투자 수익을 보장하지 않습니다. 자동매매는 코드 수준에서 비활성화되어 있습니다.

## 공개 데모

[Streamlit Community Cloud에서 앱 실행하기](https://kospi-next-day-ai-trading-copilot-lfyci4mtch4f6djvy6mzyk.streamlit.app/)

## 주요 기능

- KOSPI 데모 유니버스 스캔 및 Top 5 랭킹
- 수익률, 이동평균, RSI, MACD, 볼린저 밴드, ATR, 거래량 비율, 상대강도 특징
- LightGBM → XGBoost → RandomForest 순서의 견고한 모델 폴백
- 상승 확률, 진입 기준, ATR 기반 목표가·손절가, 신뢰도, 선정 이유
- 확장형 학습 구간, 1거래일 엠바고, 수수료·슬리피지를 반영한 워크포워드 백테스트
- 로컬 CSV 매매일지
- 한국투자증권 Open API 읽기 전용 어댑터와 paper/live 환경 분리

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
```

현재 UI는 항상 데모 데이터로 안전하게 실행되며, `src/api/kis_adapter.py`는 인증·현재가·일봉 조회만 제공합니다. `submit_order()`는 항상 예외를 발생시켜 자동주문을 차단합니다. 먼저 모의투자 환경에서 데이터 조회만 검증하세요.

## 데이터와 15:10 제약

무료 소스에서 장기간의 KOSPI 전 종목 **15:10 장중 스냅샷 이력**을 안정적으로 구하기 어렵기 때문에 MVP 백테스트는 합성 일봉 OHLCV를 사용합니다. 따라서 현재 성과 수치는 실제 투자 성과로 해석하면 안 됩니다.

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
