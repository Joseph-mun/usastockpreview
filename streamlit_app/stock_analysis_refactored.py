# -*- coding: utf-8 -*-
"""
주가 분석 리팩토링 버전
중복 제거 및 효율성 개선
"""

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import requests
import io
import urllib.request as urllib_request
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

# 일부 환경에서 잘못된 프록시(예: 127.0.0.1:9)가 설정되어
# Yahoo/FinanceDataReader/requests 호출이 실패하는 경우가 있어, 해당 케이스만 자동 해제합니다.
def _disable_bad_local_proxy_env():
    bad_markers = ("127.0.0.1:9", "localhost:9")
    keys = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    )
    removed = False
    for k in keys:
        v = os.environ.get(k)
        if v and any(m in str(v) for m in bad_markers):
            os.environ.pop(k, None)
            removed = True
    if removed:
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

_disable_bad_local_proxy_env()

# requests가 Windows 시스템 프록시를 잡아 타는 경우가 있어(환경변수 제거만으로 해결 안 됨),
# '127.0.0.1:9' 같은 죽은 프록시가 감지되면 requests 호출을 프록시 없이 강제합니다.
def _force_requests_no_proxy_if_bad_local_proxy():
    bad_markers = ("127.0.0.1:9", "localhost:9")

    def _has_bad_proxy() -> bool:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            v = os.environ.get(k)
            if v and any(m in str(v) for m in bad_markers):
                return True
        try:
            px = urllib_request.getproxies() or {}
            for v in px.values():
                if v and any(m in str(v) for m in bad_markers):
                    return True
        except Exception:
            pass
        return False

    if not _has_bad_proxy():
        return

    _orig_request = requests.request

    def _request_no_proxy(method, url, **kwargs):
        if "proxies" not in kwargs:
            kwargs["proxies"] = {}
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 20
        with requests.Session() as s:
            s.trust_env = False
            return s.request(method=method, url=url, **kwargs)

    requests.request = _request_no_proxy
    requests.api.request = _request_no_proxy
    requests.get = lambda url, **kwargs: _request_no_proxy("GET", url, **kwargs)
    requests.post = lambda url, **kwargs: _request_no_proxy("POST", url, **kwargs)

_force_requests_no_proxy_if_bad_local_proxy()

# ==================== 유틸리티 함수 ====================

def build_feature_matrix(spy: pd.DataFrame) -> pd.DataFrame:
    """
    누수/미래정보 컬럼이 섞이지 않도록, 컬럼 '위치'가 아니라 '이름' 기준으로 피처 X를 생성합니다.

    기존 코드의 iloc[:, 9:]는 컬럼 순서가 바뀌면 after/after2/Target 같은 미래/타겟 컬럼이
    X에 포함될 수 있어 train/test가 동시에 1.0이 되는 누수를 유발할 수 있습니다.
    """
    if spy is None or getattr(spy, "empty", False):
        return pd.DataFrame()

    X = spy.copy()

    # 타겟/미래/성과(누수) 컬럼
    leak_cols = {
        "Target", "TargetDown",
        "after", "after2", "after2_low",
        "suik_rate",
    }

    # 원본 가격/거래량 컬럼(모델 입력에서 제외하던 영역을 명시적으로 제거)
    base_cols = {
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "Change",  # 일부 계산 중간값이 남는 경우
    }

    drop_cols = [c for c in X.columns if c in leak_cols or c in base_cols]
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    # 비수치 컬럼 방어(혹시 문자열이 섞인 경우)
    for c in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # 결측은 모델 입력 안정성을 위해 0으로(학습/예측 파이프라인에서 ffill/bfill도 병행)
    X = X.fillna(0)

    # 최종 방어: 누수 컬럼이 남아있으면 즉시 제거
    X = X.drop(columns=[c for c in leak_cols if c in X.columns], errors="ignore")
    return X

def get_sp500_tickers():
    """S&P500 티커 리스트 가져오기"""
    # 1) Wikipedia(기존 방식) - 단, 잘못된 환경 프록시(예: 127.0.0.1:9)를 타지 않도록 trust_env=False 사용
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        sess = requests.Session()
        sess.trust_env = False  # HTTP(S)_PROXY 등 환경변수 무시
        response = sess.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            if 'Symbol' in table.columns and 'Security' in table.columns:
                df_nas = table.rename(columns={'Security': 'Name'})
                df_nas['Symbol'] = df_nas['Symbol'].astype(str).str.replace('.', '-', regex=False)
                return list(df_nas.Symbol[:500])
    except Exception:
        pass

    # 2) Fallback: FinanceDataReader listing 사용 (환경에 따라 더 안정적일 수 있음)
    try:
        listing = fdr.StockListing('S&P500')
        if listing is not None and not listing.empty and 'Symbol' in listing.columns:
            syms = listing['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
            return syms[:500]
    except Exception:
        pass

    raise ValueError("Could not load S&P 500 tickers (Wikipedia/FDR both failed)")

def get_bond_data(bond_code, start_date='2015-01-01'):
    """채권 데이터 가져오기 및 변동률 계산"""
    df = fdr.DataReader(f'FRED:{bond_code}', start=start_date, data_source='fred')
    df = df.reset_index().rename(columns={'DATE': 'Date'})
    df = df.set_index('Date')
    
    # 변동률 계산
    periods = [5, 20, 60]
    for period in periods:
        df[f'{bond_code}_{period}up'] = (df[bond_code] - df[bond_code].shift(period)) / df[bond_code].shift(period)
    
    # 결측치 제거
    for period in periods:
        df = df[~df[f'{bond_code}_{period}up'].isnull()]
    
    return df[[f'{bond_code}_{period}up' for period in periods]]

def calculate_rsi(df, period=14):
    """RSI 계산"""
    df = df.copy()
    df['Close'] = df['Adj Close']
    delta = df['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    gains = up.ewm(com=period-1, min_periods=period).mean()
    losses = down.abs().ewm(com=period-1, min_periods=period).mean()
    RS = gains / losses
    rsi = pd.Series(100 - (100 / (1 + RS)), name="RSI")
    return rsi

def calculate_macd(df, short=12, long=26, signal=9):
    """MACD 계산"""
    df = df.copy()
    df['Close'] = df['Adj Close']
    df['ema_short'] = df['Close'].ewm(span=short, adjust=False).mean()
    df['ema_long'] = df['Close'].ewm(span=long, adjust=False).mean()
    df['macd'] = df['ema_short'] - df['ema_long']
    df['signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macdhist'] = df['macd'] - df['signal']
    return df[['macd', 'signal', 'macdhist']]

def calculate_moving_averages(df, windows=[5, 20, 60, 120, 200]):
    """이동평균선 비율 계산"""
    df = df.copy()
    df['Close'] = df['Adj Close']
    for window in windows:
        df[f'MA{window}'] = df['Close'] / df['Close'].rolling(window=window).mean()
    return df[[f'MA{window}' for window in windows]]

def get_vix_data(start_date='2015-01-01', end_date=None):
    """VIX 데이터 가져오기 및 변동률 계산"""
    if end_date is None:
        end_date = datetime.now().date() + timedelta(days=1)
    
    df = fdr.DataReader('VIX', start_date, end_date)
    df = df.reset_index().rename(columns={"index": "Date"})
    df['Close'] = df['Adj Close']
    
    # 변동률 계산
    periods = [1, 5, 10]
    for period in periods:
        df[f'vix_flunc{period}'] = 100 * (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
    
    df = df[['Date', 'Close'] + [f'vix_flunc{period}' for period in periods]]
    df = df.rename(columns={'Close': 'vix'})
    df = df.set_index("Date")
    df = df.ffill().bfill()
    
    return df[['vix'] + [f'vix_flunc{period}' for period in periods]]

# ==================== 데이터 수집 및 전처리 ====================

class StockDataCollector:
    """
    주식 데이터 수집 및 전처리 클래스
    
    S&P500 주식들의 이동평균선 데이터를 수집하고, 기술적 지표 및 외부 데이터를 추가하여
    머신러닝 모델 학습에 필요한 데이터셋을 생성합니다.
    """
    
    def __init__(self, start_date='2015-01-01', end_date=None):
        """
        초기화
        
        Parameters:
        -----------
        start_date : str
            데이터 수집 시작 날짜 (기본값: '2015-01-01')
        end_date : datetime.date, optional
            데이터 수집 종료 날짜 (기본값: 오늘 날짜 + 1일)
        """
        self.start_date = start_date
        self.end_date = end_date or (datetime.now().date() + timedelta(days=1))
        self.list_window = [15, 30, 50]  # 이동평균선 기간 (15일, 30일, 50일)
        self.dataframes = {}  # 각 이동평균선별 데이터프레임 저장
        
    def collect_sma_data(self, ticker_list, max_tickers=500, progress_callback=None, status_callback=None):
        """
        이동평균선 데이터 수집
        
        S&P500 주식들의 이동평균선(SMA) 데이터를 FinanceDataReader API를 통해 수집합니다.
        각 주식에 대해 15일, 30일, 50일 이동평균선을 계산하고, 현재 가격이 이동평균선 위에 있는지
        아래에 있는지를 판단합니다. 이 정보는 전체 시장의 강세/약세를 파악하는 데 사용됩니다.
        
        Parameters:
        -----------
        ticker_list : list
            분석할 티커 리스트 (예: ['AAPL', 'MSFT', 'GOOGL', ...])
        max_tickers : int
            최대 분석할 티커 수 (기본값: 500, 전체 S&P500 주식 수)
        progress_callback : function, optional
            진행도 업데이트 콜백 함수 (progress_value: float, 0.0~1.0)
            스트림릿 UI에서 진행도 바를 업데이트하는 데 사용됩니다.
        status_callback : function, optional
            상태 업데이트 콜백 함수 (status_text: str)
            스트림릿 UI에서 현재 상태 메시지를 표시하는 데 사용됩니다.
        
        Returns:
        --------
        tuple : (tickers_above_sma50, tickers_below_sma50)
            tickers_above_sma50: 50일 이동평균선 위에 있는 티커 리스트
            tickers_below_sma50: 50일 이동평균선 아래에 있는 티커 리스트
        
        Note:
        -----
        이 함수는 FinanceDataReader API를 호출하므로 시간이 오래 걸릴 수 있습니다.
        주식당 약 1-2초가 소요되며, 100개 주식을 수집하면 약 2-3분이 걸립니다.
        """
        if status_callback:
            status_callback("S&P500 주식 데이터 수집 중...")
        
        tickers_above_sma50 = []
        tickers_below_sma50 = []

        # concat 비용을 줄이기 위해 window별로 리스트에 모았다가 마지막에 한 번만 concat
        window_rows = {k: [] for k in self.list_window}
        
        total_tickers = min(len(ticker_list), max_tickers)
        
        for idx, code in enumerate(ticker_list[:max_tickers]):
            # 진행도 업데이트
            if progress_callback:
                progress = (idx + 1) / total_tickers
                progress_callback(progress)
            
            if status_callback:
                status_callback(f"수집 중: {code} ({idx+1}/{total_tickers})")

            # 핵심 최적화: 티커당 DataReader는 1번만 호출하고, 15/30/50 SMA를 한 번에 계산
            try:
                stock_df = fdr.DataReader(code, self.start_date, self.end_date)
                if stock_df is None or stock_df.empty:
                    continue

                # 안정적으로 날짜 인덱스 정렬
                stock_df = stock_df.sort_index()
                stock_df['Code'] = code

                # 각 window별 데이터 생성
                for b in self.list_window:
                    temp = stock_df[['Adj Close', 'Code']].copy()
                    temp[f'SMA{b}'] = temp['Adj Close'].rolling(window=b).mean()
                    # 1 = "가격이 이동평균선 위(강세)", 0 = "가격이 이동평균선 아래(약세)"
                    # (기존에는 비교 방향이 반대라 위/아래가 뒤집혀 보이는 문제가 있었음)
                    temp[f'SMA{b}_YN'] = np.where(temp['Adj Close'] > temp[f'SMA{b}'], 1, 0)
                    temp = temp[~temp[f'SMA{b}'].isnull()]

                    if temp.empty:
                        continue

                    # Date 컬럼을 갖도록 인덱스를 컬럼으로 변환 (후처리 reset/rename 반복 제거)
                    temp = temp.reset_index().rename(columns={"index": "Date"})
                    window_rows[b].append(temp)

                    # SMA50 기준 위/아래 판단은 마지막 유효 row 기준
                    if b == 50:
                        if int(temp[f'SMA{b}_YN'].iloc[-1]) == 1:
                            tickers_above_sma50.append(code)
                        else:
                            tickers_below_sma50.append(code)

            except Exception as e:
                if status_callback:
                    status_callback(f"⚠️ {code} 데이터 수집 실패: {str(e)[:50]}")
                continue
        
        # window별 데이터프레임 생성 (한 번만 concat)
        for k in self.list_window:
            if window_rows[k]:
                self.dataframes[f'sma{k}stock_df'] = pd.concat(window_rows[k], ignore_index=True)
            else:
                self.dataframes[f'sma{k}stock_df'] = pd.DataFrame()
        
        if status_callback:
            status_callback(f"✅ 데이터 수집 완료 (위: {len(tickers_above_sma50)}, 아래: {len(tickers_below_sma50)})")
        
        return tickers_above_sma50, tickers_below_sma50
    
    def prepare_target_data(self, ticker='IXIC', for_prediction=False):
        """
        타겟 데이터 준비
        
        Parameters:
        -----------
        ticker : str
            티커 심볼 (기본값: 'IXIC')
        for_prediction : bool
            예측용 데이터인지 여부 (True면 after 필터링 제거)
        """
        spy = fdr.DataReader(ticker, self.start_date, self.end_date)
        spy['Close'] = spy['Adj Close']
        spy['after'] = spy['Close'].shift(-15)
        
        spy0 = spy.sort_index(ascending=False)
        spy0['after2'] = spy0['Close'].rolling(window=20).max()
        # 하락 타겟용: 향후 20거래일 최저가 기준
        spy0['after2_low'] = spy0['Close'].rolling(window=20).min()
        spy = spy0.sort_index(ascending=True)
        
        spy['Target'] = np.where(spy['after2'] >= 1.03 * spy['Close'], 1, 0)
        spy['TargetDown'] = np.where(spy['after2_low'] <= 0.97 * spy['Close'], 1, 0)
        
        # 예측용이 아닐 때만 after가 null인 행 제거 (학습용)
        if not for_prediction:
            spy = spy[~spy['after'].isnull()]
        
        return spy
    
    def add_features(self, spy, skip_sma=False, for_prediction=False):
        """
        기술적 지표 및 외부 데이터 추가
        
        Parameters:
        -----------
        spy : pandas.DataFrame
            타겟 데이터프레임
        skip_sma : bool
            True면 SMA 비율 계산을 건너뜀 (예측 시 빠른 계산을 위해)
        for_prediction : bool
            True면 날짜 누락 방지를 위해 spy(IXIC) 인덱스를 기준으로 정렬/정합(reindex) 후 결측치를
            ffill/bfill로 채웁니다. (예측용) False면 기존처럼 교집합(inner join) 기반으로 학습용 데이터를 만듭니다.
        """
        spy = spy.copy()
        spy.index = pd.to_datetime(spy.index)
        spy = spy.sort_index()
        base_index = spy.index

        join_how = 'left' if for_prediction else 'inner'

        # Change20day 계산 (참조 코드에 따라 window=20 사용)
        k1 = fdr.DataReader('IXIC', self.start_date, self.end_date)
        k1.index = pd.to_datetime(k1.index)
        k1 = k1.sort_index()
        k1['Close'] = k1['Adj Close']
        k1['Change'] = (k1['Close'] - k1['Close'].shift(1)) / k1['Close'].shift(1)
        k1 = k1[~k1['Change'].isnull()]
        k2 = np.log10(1 + k1['Change']).rolling(window=20).sum()
        k1['Change20day'] = (pow(10, k2) - 1) * 100
        spy = spy.join(k1[['Change20day']], how=join_how)
        if for_prediction:
            spy['Change20day'] = spy['Change20day'].reindex(base_index).ffill().bfill()
        else:
            spy = spy[~spy['Change20day'].isnull()]
        
        # SMA 비율 추가 (skip_sma가 False일 때만)
        if not skip_sma:
            for b in self.list_window:
                temp_df = self.dataframes[f'sma{b}stock_df'].copy()
                if 'Date' in temp_df.columns:
                    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
                    temp_df = temp_df.set_index('Date')
                else:
                    temp_df.index = pd.to_datetime(temp_df.index)
                temp_df = temp_df.sort_index()

                ratio = temp_df.groupby(level=0)[f'SMA{b}_YN'].sum() / temp_df.groupby(level=0)[f'SMA{b}_YN'].count()
                ratio = ratio.sort_index()

                # spy의 날짜 인덱스에 정합
                ratio_aligned = ratio.reindex(base_index)
                if for_prediction:
                    ratio_aligned = ratio_aligned.ffill().bfill()

                spy.loc[:, f'ratio_sma{b}'] = ratio_aligned
                if not for_prediction:
                    spy = spy[~spy[f'ratio_sma{b}'].isnull()]
        else:
            # SMA 비율을 건너뛰는 경우, 최근 평균값으로 채움 (모델 호환성을 위해)
            # 또는 마지막 유효값을 forward fill
            for b in self.list_window:
                spy[f'ratio_sma{b}'] = 0.5  # 기본값 설정 (중립)
        
        spy['suik_rate'] = 100 * (spy['after'] - spy['Close']) / spy['Close']
        
        # 채권 데이터 추가
        bond_codes = ['DGS2', 'DGS10', 'DGS30']
        for bond_code in bond_codes:
            bond_df = get_bond_data(bond_code, '2015')
            bond_df = bond_df.copy()
            bond_df.index = pd.to_datetime(bond_df.index)
            bond_df = bond_df.sort_index()
            spy = spy.join(bond_df, how=join_how)
            if for_prediction:
                for c in bond_df.columns:
                    spy[c] = spy[c].reindex(base_index).ffill().bfill()
        
        # VIX 데이터 추가
        vix_df = get_vix_data(self.start_date, self.end_date)
        vix_df = vix_df.copy()
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_df = vix_df.sort_index()
        spy = spy.join(vix_df, how=join_how)
        if for_prediction:
            for c in vix_df.columns:
                spy[c] = spy[c].reindex(base_index).ffill().bfill()
        
        # MACD 추가
        ixic_df = fdr.DataReader('IXIC', self.start_date, self.end_date)
        ixic_df.index = pd.to_datetime(ixic_df.index)
        ixic_df = ixic_df.sort_index()
        macd_df = calculate_macd(ixic_df)
        macd_df.index = pd.to_datetime(macd_df.index)
        macd_df = macd_df.sort_index()
        spy = spy.join(macd_df, how=join_how)
        if for_prediction:
            for c in macd_df.columns:
                spy[c] = spy[c].reindex(base_index).ffill().bfill()
        
        # RSI 추가
        rsi_series = calculate_rsi(ixic_df)
        rsi_series.index = pd.to_datetime(rsi_series.index)
        rsi_series = rsi_series.sort_index()
        spy['rsi'] = rsi_series.reindex(base_index)
        if for_prediction:
            spy['rsi'] = spy['rsi'].ffill().bfill()
        else:
            spy = spy[~spy['rsi'].isnull()]
        
        # 이동평균선 추가
        ma_df = calculate_moving_averages(ixic_df)
        ma_df.index = pd.to_datetime(ma_df.index)
        ma_df = ma_df.sort_index()
        spy = spy.join(ma_df, how=join_how)
        if for_prediction:
            for c in ma_df.columns:
                spy[c] = spy[c].reindex(base_index).ffill().bfill()
        
        # 장단기 금리차 추가 (FinanceDataReader 사용)
        try:
            # T10Y2Y는 10년-2년 금리차
            k9 = fdr.DataReader('FRED:T10Y2Y', start='2015-01-01', data_source='fred')
            df8 = k9.reset_index()
            if 'DATE' in df8.columns:
                df8 = df8.rename(columns={'DATE': 'Date'})
            elif df8.index.name == 'Date':
                df8 = df8.reset_index()
            df8 = df8.set_index("Date")
            df8 = df8.ffill()
            df8.index = pd.to_datetime(df8.index)
            df8 = df8.sort_index()
            spy = spy.join(df8, how=join_how)
            if for_prediction:
                for c in df8.columns:
                    spy[c] = spy[c].reindex(base_index).ffill().bfill()
        except Exception as e:
            # T10Y2Y 데이터를 가져올 수 없는 경우 스킵
            print(f"Warning: T10Y2Y 데이터를 가져올 수 없습니다: {e}")

        # 예측용은 날짜 누락 없이 유지 + 결측치 보정
        if for_prediction:
            spy = spy.reindex(base_index).sort_index()
            # 날짜는 유지하되, 학습/타겟 관련 컬럼은 임의로 채우지 않음
            fill_exclude = {'after', 'after2', 'Target', 'suik_rate'}
            fill_cols = [c for c in spy.columns if c not in fill_exclude]
            if fill_cols:
                spy.loc[:, fill_cols] = spy.loc[:, fill_cols].ffill().bfill()
        
        return spy

# 예측용 최신 데이터 준비 함수 추가
def prepare_prediction_data(end_date=None, progress_callback=None, status_callback=None, sma_dataframes=None):
    """
    예측을 위한 최신 데이터 준비
    Train 데이터는 1월 2일까지지만, 예측 시에는 현재 날짜까지의 최신 데이터를 가져옴
    학습 시 수집한 이평선 데이터를 재사용하여 빠르게 예측 가능
    
    Parameters:
    -----------
    end_date : datetime.date, optional
        데이터 수집 종료 날짜 (기본값: 오늘 날짜 + 1일)
    progress_callback : function, optional
        진행도 업데이트 콜백 함수 (progress_value: float, 0.0~1.0)
    status_callback : function, optional
        상태 업데이트 콜백 함수 (status_text: str)
    sma_dataframes : dict, optional
        학습 시 수집한 이평선 데이터프레임 딕셔너리 (재사용용)
    
    Returns:
    --------
    tuple : (X, spy)
        X: 최신 feature 데이터 (마지막 행이 최신 데이터)
        spy: 전체 데이터프레임
    """
    if end_date is None:
        end_date = datetime.now().date() + timedelta(days=1)
    
    if status_callback:
        status_callback("1/3: 데이터 수집기 초기화 중...")
    if progress_callback:
        progress_callback(0.1)
    
    # 최신 데이터로 collector 생성
    collector = StockDataCollector(start_date='2015-01-01', end_date=end_date)
    
    # 학습 시 수집한 이평선 데이터가 있으면 재사용
    if sma_dataframes is not None:
        collector.dataframes = sma_dataframes.copy()
        if status_callback:
            status_callback("1/3: 학습 시 수집한 이평선 데이터 재사용 중...")
    
    # 타겟 데이터 준비 (최신 데이터까지 포함, after 필터링 제거)
    if status_callback:
        status_callback("2/3: IXIC 데이터 준비 중...")
    if progress_callback:
        progress_callback(0.4)
    
    spy = collector.prepare_target_data('IXIC', for_prediction=True)
    
    # 특성 데이터 추가 (학습 시 수집한 이평선 데이터 재사용)
    if status_callback:
        status_callback("3/3: 특성 데이터 준비 중...")
    if progress_callback:
        progress_callback(0.7)
    
    # 이평선 데이터가 있으면 사용, 없으면 건너뛰기
    #skip_sma = (sma_dataframes is None)
    #spy = collector.add_features(spy, skip_sma=skip_sma)


    # 1. sma_dataframes가 None이라도 기능을 건너뛰지 않도록 skip_sma를 False로 설정
    # (단, collector.add_features 내부에서 sma_dataframes가 None일 때의 예외처리가 되어 있어야 합니다)
    skip_sma = False 

    # 2. 피처 추가 실행
    # 예측용: 날짜(거래일) 누락 방지를 위해 IXIC 인덱스를 유지하며 피처를 정합/보정
    spy = collector.add_features(spy, skip_sma=skip_sma, for_prediction=True)

    # 3. 결측치 보정 (날짜는 유지, 피처만 채움)
    # add_features(for_prediction=True)에서 대부분 채우지만, 혹시 남아있는 피처 결측치만 전일값으로 보정
    fill_exclude = {'after', 'after2', 'Target', 'suik_rate'}
    fill_cols = [c for c in spy.columns if c not in fill_exclude]
    if fill_cols:
        spy.loc[:, fill_cols] = spy.loc[:, fill_cols].ffill()


    # 최신 특성 데이터 추출 (누수 방지)
    X = build_feature_matrix(spy)
    
    if progress_callback:
        progress_callback(1.0)
    if status_callback:
        status_callback("✅ 데이터 준비 완료")
    
    return X, spy

# ==================== 모델 학습 및 예측 ====================

class StockPredictor:
    """주가 예측 모델 클래스"""
    
    def __init__(self, model_path='stock_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_columns = None
        
    def train_model(self, X, y, test_size=0.2, random_state=500, n_estimators=2000, progress_callback=None, status_callback=None):
        """
        RandomForest 모델 학습
        
        주가 상승/하락을 예측하기 위한 RandomForest 분류 모델을 학습합니다.
        모델은 2000개의 의사결정 트리를 사용하며, 각 트리는 최대 깊이 16까지 확장됩니다.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            학습 데이터 (특성) - 기술적 지표, 채권 데이터, VIX 등
        y : pandas.Series
            학습 데이터 (타겟) - 주가가 15일 후 3% 이상 상승할지 여부 (0 또는 1)
        test_size : float
            테스트 데이터 비율 (기본값: 0.2, 즉 20%)
        random_state : int
            랜덤 시드 (재현 가능한 결과를 위해 사용)
        progress_callback : function, optional
            진행도 업데이트 콜백 함수 (progress_value: float, 0.0~1.0)
            스트림릿 UI에서 진행도 바를 업데이트하는 데 사용됩니다.
        status_callback : function, optional
            상태 업데이트 콜백 함수 (status_text: str)
            스트림릿 UI에서 현재 학습 상태를 표시하는 데 사용됩니다.
        
        Returns:
        --------
        tuple : (train_score, test_score, oob_score)
            train_score: 훈련 세트 정확도
            test_score: 테스트 세트 정확도
            oob_score: Out-of-Bag 샘플 정확도 (교차 검증 없이 모델 성능 평가)
        
        Note:
        -----
        모델 학습은 시간이 오래 걸릴 수 있습니다 (2000개 트리 생성, 수분 소요).
        n_jobs=-1로 설정하여 모든 CPU 코어를 사용하여 학습 속도를 향상시킵니다.
        """
        if status_callback:
            status_callback("데이터 전처리 중...")
        
        X2 = np.array(X.values)
        y2 = np.array(y.values).reshape(-1, 1)
        
        if status_callback:
            status_callback("데이터 분할 중...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X2, y2, test_size=test_size, random_state=random_state
        )
        
        if status_callback:
            status_callback(f"학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
        
        # Windows/보안 환경에서 joblib(스레드/프로세스) 백엔드가 파이프(IPC)를 만들 때
        # WinError 5가 발생할 수 있어 기본은 단일 작업(n_jobs=1)으로 설정합니다.
        n_jobs = 1 if os.name == "nt" else -1

        self.model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=16,
            min_samples_leaf=3,
            min_samples_split=4,
            random_state=52,
            oob_score=True,
            n_jobs=n_jobs  # Windows: 안정성 우선(1), 그 외: 멀티코어(-1)
        )
        
        if status_callback:
            status_callback("RandomForest 모델 학습 중... (2000개 트리 생성)")
        
        # 모델 학습
        # sklearn 내부가 Parallel(prefer="threads")를 타는 경우가 있어,
        # WinError 5 회피를 위해 joblib 백엔드를 "sequential"로 강제합니다.
        with joblib.parallel_backend("sequential"):
            self.model.fit(X_train, y_train)
        
        if progress_callback:
            progress_callback(0.8)  # 학습 완료
        
        self.feature_columns = X.columns.tolist()
        
        if status_callback:
            status_callback("모델 평가 중...")
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        oob_score = self.model.oob_score_

        # threshold 기반 성능은 별도 기능으로 분리(이전 버전으로 롤백)
        
        if progress_callback:
            progress_callback(1.0)  # 완료
        
        if status_callback:
            status_callback(f"✅ 학습 완료! (정확도: {test_score:.3f})")
        
        print(f"훈련 세트 정확도: {train_score:.3f}")
        print(f"테스트 세트 정확도: {test_score:.3f}")
        print(f"OOB 샘플의 정확도: {oob_score:.3f}")
        
        return train_score, test_score, oob_score
    
    def save_model(self):
        """모델 저장"""
        if self.model is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_columns': self.feature_columns
                }, f)
            print(f"모델이 {self.model_path}에 저장되었습니다.")
    
    def load_model(self):
        """모델 로드"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_columns = data['feature_columns']
            print(f"모델이 {self.model_path}에서 로드되었습니다.")
            return True
        return False
    
    def predict_proba(self, X):
        """확률 예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        # DataFrame이면 학습 당시 feature_columns에 맞춰 정렬/필터링
        if isinstance(X, pd.DataFrame) and self.feature_columns:
            X = X.reindex(columns=self.feature_columns).fillna(0)
        return self.model.predict_proba(X)
    
    def get_current_probability(self, feature_data):
        """현재 확률 계산"""
        if self.model is None:
            return None
        
        # Series면 학습 당시 feature_columns에 맞춰 정렬/필터링
        if isinstance(feature_data, pd.Series) and self.feature_columns:
            feature_data = feature_data.reindex(self.feature_columns).fillna(0)
            row = [feature_data.values]
        else:
            row = [feature_data]

        # 최신 데이터의 확률 계산
        proba = self.model.predict_proba(row)[0]
        return proba[1]  # 클래스 1의 확률
    
    def get_probability_history(self, X, days=1300):
        """과거 확률 추이 계산"""
        if self.model is None:
            return None

        # 기존 구현은 1행씩 predict_proba를 호출해서 매우 느립니다.
        # 벡터화로 한 번에 계산하여 속도를 크게 개선합니다.
        n = min(days, len(X))
        if n <= 0:
            return None

        X_slice = X.iloc[-n:].copy()
        if isinstance(X_slice, pd.DataFrame) and self.feature_columns:
            X_slice = X_slice.reindex(columns=self.feature_columns).fillna(0)
        proba = self.model.predict_proba(X_slice.values)[:, 1]

        out = pd.DataFrame({'Probability': proba}, index=X_slice.index)
        out.index.name = 'Date'
        return out.sort_index()
