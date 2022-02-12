from pykrx import stock
from pykrx import bond

# KOSPI/KOSDAQ/KONEX 종목코드 조회
stock_code = stock.get_market_ticker_list(date="20210520", market="ALL")
print(stock_code[:6], len(stock_code))

# KOSPI 종목코드 조회
stock_code = stock.get_market_ticker_list(date="20210520", market="KOSPI")
print(stock_code[:6], len(stock_code))

# KOSDAQ 종목코드 조회
stock_code = stock.get_market_ticker_list(date="20210520", market="KOSDAQ")
print(stock_code[:6], len(stock_code))

# KONEX 종목코드 조회
stock_code = stock.get_market_ticker_list(date="20210520", market="KONEX")
print(stock_code[:6], len(stock_code))

# 종목명 반환
stock_name = stock.get_market_ticker_name("005930")
print(stock_name)

# 삼성전자의 20210501~20210520의 주가데이터
df = stock.get_market_ohlcv_by_date(fromdate="20210501", todate="20210520", ticker="005930")
print(df)

# 한번에 많은 종목을 조회하면 ip가 차단될 수 있기 때문에 반복분을 돌면서 사용할때는 time 모듈로 sleep을 걸어주는 것이 좋음
import time
import pandas as pd
stock_code = stock.get_market_ticker_list() # 현재일자 기준 가장 가까운 영업일의 코스피 상장종목 리스트
res = pd.DataFrame()
for ticker in stock_code[:4]:
    df = stock.get_market_ohlcv_by_date(fromdate="20220209", todate="20220212", ticker=ticker)
    df = df.assign(종목코드=ticker, 종목명=stock.get_market_ticker_name(ticker))
    res = pd.concat([res, df], axis=0)
    time.sleep(1)
res = res.reset_index()
print(res)


df = stock.get_market_ohlcv_by_ticker(date="20210520")
print(df.head())
# get_market_ohlcv_by_ticker(date="YYYYMMDD", market="거래소명")
# market의 기본값은 "ALL"(전체 시장; 코스피(KOSPI)/코스닥(KOSDAQ)/코넥스(KONEX))


df = stock.get_market_price_change_by_ticker(fromdate="20210517", todate="20210520")
print(df.head())
# 모든 종목의 가격 변동 조회
# get_market_price_change_by_ticker(fromdate="조회시작일", todate="조회종료일", market="거래소명")
# market의 기본값은 "ALL"
# 조회시작일 대비 조회종료일의 변동을 계산


df = stock.get_market_fundamental_by_ticker(date="20210520")
df.head()
#  특정일자의 종목별 DIV/BPS/PER/EPS 조회
# get_market_fundamental_by_ticker(date="YYYYMMDD", market="거래소명")

# market의 기본값은 "ALL"

# DIV(배당수익률): (주가배당금/주가) * 100

# BPS(주당순자산가치=청산가치): (순자산)/(총발행주식수)

# PER(주가수익비율): (주가)/(주당순이익)

# EPS(주당순이익): (당기순이익)/(총발행주식수)

# PBR(주가순자산비율) = (주가)/(BPS) = PER*EPS / BPS


df = stock.get_market_fundamental_by_date(fromdate="20210517", todate="20210520", ticker="005930")
df.head()
# 일자별 DIV/BPS/PER/EPS 조회
# get_market_fundamental_by_date(fromdate, todate, ticker, freq='d', name_display=False)

# freq.6 : d(일), m(월), y(연도)





# https://psystat.tistory.com/114
# https://github.com/sharebook-kr/pykrx
