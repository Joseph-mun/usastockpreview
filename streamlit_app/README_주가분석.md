# 주가 예측 확률 분석 시스템

주가 분석 코드를 리팩토링하고 스트림릿 UI로 최종 확률을 확인할 수 있는 시스템입니다.

## 주요 개선 사항

### 1. 코드 리팩토링 (`stock_analysis_refactored.py`)
- ✅ 중복 코드 제거 및 함수화
- ✅ 채권 데이터 수집 로직 통합
- ✅ 기술적 지표 계산 함수화 (RSI, MACD, 이동평균선)
- ✅ 클래스 기반 구조로 재구성
- ✅ 모델 저장/로드 기능 추가

### 2. 스트림릿 UI (`stock_prediction_app.py`)
- ✅ 현재 상승 확률 실시간 확인
- ✅ 확률 추이 그래프 (Plotly)
- ✅ 주요 기술적 지표 표시
- ✅ 모델 정보 및 성능 지표
- ✅ 모델 학습/로드 기능

## 설치 방법

```bash
pip install -r requirements_stock.txt
```

## 사용 방법

### 1. 스트림릿 앱 실행

```bash
cd c:\Users\youyj\.cursor\projects\SAMPLE
streamlit run stock_prediction_app.py
```

### 2. 모델 학습

1. 사이드바에서 **"새 모델 학습"** 선택
2. **"실시간 데이터 수집"** 체크 (선택사항)
3. 분석할 주식 수 조정 (기본값: 100)
4. **"모델 학습 시작"** 버튼 클릭
5. 학습 완료 후 모델이 자동 저장됨

### 3. 확률 확인

- 학습된 모델이 있으면 자동으로 로드되어 현재 확률을 표시합니다
- 확률 추이 그래프와 주요 지표를 확인할 수 있습니다

## 파일 구조

```
SAMPLE/
├── stock_analysis_refactored.py  # 리팩토링된 분석 코드
├── stock_prediction_app.py        # 스트림릿 UI 앱
├── requirements_stock.txt         # 필요한 패키지 목록
├── stock_model.pkl               # 저장된 모델 (학습 후 생성)
└── README_주가분석.md            # 사용 설명서
```

## 주요 기능

### 데이터 수집
- S&P500 주식 리스트 자동 수집
- 이동평균선 데이터 계산 (15일, 30일, 50일)
- 채권 데이터 수집 (DGS2, DGS10, DGS30)
- VIX 데이터 수집
- 기술적 지표 계산 (RSI, MACD, 이동평균선)

### 모델 학습
- RandomForest Classifier 사용
- 하이퍼파라미터:
  - n_estimators: 2000
  - max_depth: 16
  - min_samples_leaf: 3
  - min_samples_split: 4

### 예측 기능
- 현재 상승 확률 계산
- 과거 확률 추이 분석
- 확률 기반 매수/매도 신호 제공

## 주의사항

⚠️ **이 시스템은 참고용입니다. 실제 투자 결정에 사용하기 전에 전문가의 조언을 구하세요.**

- 모델 학습은 시간이 오래 걸릴 수 있습니다 (수십 분 소요 가능)
- 인터넷 연결이 필요합니다
- FinanceDataReader API 사용량 제한이 있을 수 있습니다

## 문제 해결

### 모델이 로드되지 않는 경우
- `stock_model.pkl` 파일이 같은 디렉토리에 있는지 확인
- 모델을 먼저 학습하세요

### 데이터 수집 오류
- 인터넷 연결 확인
- FinanceDataReader API 상태 확인
- 주식 수를 줄여서 다시 시도

### Import 오류
- 모든 패키지가 설치되었는지 확인: `pip install -r requirements_stock.txt`
- Python 버전 확인 (3.8 이상 권장)
