from ntpath import join
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import ndiffs
import pmdarima as pm
import os
from matplotlib import font_manager, rc
import matplotlib as mpl
import csv
import matplotlib.font_manager as fm
import matplotlib_inline
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import matplotlib.ticker as ticker

# 1. 데이터
woori = fdr.DataReader('041190', '2021', '2022')
woori.reset_index(inplace=True)
woori.set_index('Date', inplace=True)
print(woori.head(100))

woori = woori[['Close']]
woori = woori.rename(columns = {'Close':'Price'})

y_train = woori['Price'][:int(0.7*len(woori))]
y_test = woori['Price'][int(0.7*len(woori)):]

# 시간의 흐름에 따른 데이터 분포를 보자
woori['Price'].plot()
plt.ylabel('price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(y_train)
plt.show()

plt.ylabel('price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(y_test)
plt.show()

# ACF와 PACF를 보자
import statsmodels.graphics.tsaplots as sgt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
sgt.plot_acf(woori, zero = False)
plt.show() 

sgt.plot_pacf(woori, zero = False)
plt.show()
# 그래프로 보아서는 acf의 stationary를 추정하기 애매함.

# data가 nonstationary하기 때문에 differencing을 해주어야하는데
# differencing을 몇번해주는게 좋을까~?
diff_1 = woori['Price'].diff().dropna()
plt.plot(diff_1)
plt.show()

diff_2 = diff_1.diff().dropna()
plt.plot(diff_2)
plt.show()
# 1번 해주었을때와 2번해주었을때 시각화를 통해 비교 -> 비교불가

# ndiffs함수를 이용하여 pmdarima 테스트를 통해 추정해보자
kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)   
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"추정된 차수 d = {n_diffs}")  # 추정된 차수 d = 1

# 1번 differencing 진행 후 ACF와 PACF를 보자
woori = woori['Price'].diff().dropna()

sgt.plot_acf(woori)
plt.show() 

sgt.plot_pacf(woori)
plt.show()
# 그래프로 보아서는 acf와 pacf 둘다 절단점이 0을 가지게 된다.



#2. 모델링
# pm.auto_arima(y_train, d = 1, seasonal = False, trace = True)
# '''
# Performing stepwise search to minimize aic
#  ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.29 sec
#  ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=2577.444, Time=0.01 sec
#  ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=2578.486, Time=0.02 sec
#  ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=2578.479, Time=0.02 sec
#  ARIMA(0,1,0)(0,0,0)[0]             : AIC=2575.704, Time=0.00 sec
#  ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=2580.473, Time=0.07 sec

# Best model:  ARIMA(0,1,0)(0,0,0)[0]
# Total fit time: 0.407 seconds

# # => Best model: ARIMA(0,1,0)을 사용하는 것이 가장 좋음
# # '''
model = ARIMA(woori, order=(1,1,0))
model_fit = model.fit()
print(model_fit.summary())


# model.fit(y_train)
# print(model.summary())
# '''
#                                SARIMAX Results
# ==============================================================================
# Dep. Variable:                      y   No. Observations:                  173
# Model:               SARIMAX(0, 1, 0)   Log Likelihood               -1286.852
# Date:                Mon, 04 Apr 2022   AIC                           2575.704
# Time:                        20:59:13   BIC                           2578.851
# Sample:                             0   HQIC                          2576.981
#                                 - 173
# Covariance Type:                  opg
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# sigma2      1.843e+05   9929.030     18.563      0.000    1.65e+05    2.04e+05
# ===================================================================================
# Ljung-Box (L1) (Q):                   0.83   Jarque-Bera (JB):               280.25
# Prob(Q):                              0.36   Prob(JB):                         0.00
# Heteroskedasticity (H):               0.68   Skew:                             1.12
# Prob(H) (two-sided):                  0.16   Kurtosis:                         8.84
# ===================================================================================
# '''
# 3. 잔자검정 
# 위 result를 시각화하여 잔차검정을 해보자~!
# model.plot_diagnostics(figsize=(16, 8))
# plt.show()

# # 4. 데이터 예측
# # 테스트 데이터 개수만큼 예측
# y_predict = model.predict(n_periods=len(y_test))
# y_predict = pd.DataFrame(y_predict,index = y_test.index,columns=['Prediction'])

# # 그래프
# fig, axes = plt.subplots(1, 1, figsize=(12, 4))
# plt.plot(y_train, label='Train')        # 훈련 데이터
# plt.plot(y_test, label='Test')          # 테스트 데이터
# plt.plot(y_predict, label='Prediction')  # 예측 데이터
# plt.legend()
# plt.show()


# def forecast_one_step():
#     fc, conf_int = model.predict(n_periods=1 # 한 스텝씩!
#         , return_conf_int=True)              # 신뢰구간 출력
#     return (
#         fc.tolist()[0],
#         np.asarray(conf_int).tolist()[0]
#     )
    
# forecasts = []
# y_pred = []
# pred_upper = []
# pred_lower = []

# for new_ob in y_test:
#     fc, conf = forecast_one_step()
#     y_pred.append(fc)
#     pred_upper.append(conf[1])
#     pred_lower.append(conf[0])

#     ## 모형 업데이트 !!
#     model.update(new_ob)
    
# pd.DataFrame({"test": y_test, "pred": y_pred})

# print(model.summary())
# '''
#                                SARIMAX Results
# ==============================================================================
# Dep. Variable:                      y   No. Observations:                  248  -> 데이터 늘어남
# Model:               SARIMAX(0, 1, 0)   Log Likelihood               -1847.399  ->
# Date:                Mon, 04 Apr 2022   AIC                           3696.798  ->     업데이트 됨:
# Time:                        23:11:03   BIC                           3700.307  ->
# Sample:                             0   HQIC                          3698.211  ->
#                                 - 248
# Covariance Type:                  opg
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# sigma2       1.84e+05   8505.102     21.639      0.000    1.67e+05    2.01e+05
# ===================================================================================
# Ljung-Box (L1) (Q):                   0.71   Jarque-Bera (JB):               351.21
# Prob(Q):                              0.40   Prob(JB):                         0.00
# Heteroskedasticity (H):               0.70   Skew:                             1.02
# Prob(H) (two-sided):                  0.11   Kurtosis:                         8.48
# ===================================================================================
# '''

# # 모델 업데이트 시각화해보자
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# fig = go.Figure([
#     # 훈련 데이터-------------------------------------------------------
#     go.Scatter(x = y_train.index, y = y_train, name = "Train", mode = 'lines'
#               ,line=dict(color = 'royalblue'))
#     # 테스트 데이터------------------------------------------------------
#     , go.Scatter(x = y_test.index, y = y_test, name = "Test", mode = 'lines'
#                 ,line = dict(color = 'rgba(0,0,30,0.5)'))
#     # 예측값-----------------------------------------------------------
#     , go.Scatter(x = y_test.index, y = y_pred, name = "Prediction", mode = 'lines'
#                      ,line = dict(color = 'red', dash = 'dot', width=3))
    
#     # 신뢰 구간---------------------------------------------------------
#     , go.Scatter(x = y_test.index.tolist() + y_test.index[::-1].tolist() 
#                 ,y = pred_upper + pred_lower[::-1] ## 상위 신뢰 구간 -> 하위 신뢰 구간 역순으로
#                 ,fill='toself'
#                 ,fillcolor='rgba(0,0,30,0.1)'
#                 ,line=dict(color='rgba(0,0,0,0)')
#                 ,hoverinfo="skip"
#                 ,showlegend=False)
# ])

# fig.update_layout(height=400, width=1000, title_text="ARIMA(0,1,0)모형")
# # fig.show()

# # 5. 평가 - MAPE (Mean Absolute Percentage Error) 지표 이용 : 작을수록 좋음
# def MAPE(y_test, y_pred):
#     	return np.mean(np.abs((y_test - y_pred) / y_test)) * 100 
    
# print(f"MAPE: {MAPE(y_test, y_pred):.3f}")  # MAPE: 2.991 -> 잔차가 실제값의 
