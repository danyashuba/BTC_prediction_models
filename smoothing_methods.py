import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, \
    mean_absolute_percentage_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose


def forecast_errors(info, test_data, real_data):
    mse = mean_squared_error(test_data, real_data)
    mae = mean_absolute_error(test_data, real_data)
    r2 = r2_score(test_data, real_data)
    mslge = mean_squared_log_error(test_data, real_data)
    mape = mean_absolute_percentage_error(test_data, real_data)
    print(info, '\nMSE = ', mse, '\nMAE =', mae,
          '\nR2 = ', r2, '\nMSLGE = ', mslge, '\nMAPE = ', mape, '\n')
    return None


def set_plot(title='', axis='', y=''):
    plt.title(title), plt.xlabel(axis), plt.ylabel(y)
    plt.legend()
    plt.show()
    return None


stock_data = pd.read_csv('/Users/daniilsuba/Desktop/BTC-USD.csv', sep=',')
# print(stock_data)
split = 0.3
stock_data.index = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d')
stock_data = stock_data.iloc[::-1]
# print(stock_data)
# print(stock_data.columns)
stock_data = stock_data.drop(columns='Adj Close', axis=1)
# print(stock_data.columns)

stock_data["average"] = (stock_data["High"] + stock_data["Low"]) / 2
stock_data.head()
# print(stock_data)
x = stock_data.average.values
# print(x)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
stock_data.average = x_scaled

n_test = int(split * len(stock_data))
test = stock_data[:n_test]
train = stock_data[n_test:]
plt.plot(train.average, label='Train')
plt.plot(test.average, label='Test')
set_plot(title='Share prices', axis="Time", y='Price')

# ******************************moving_average*********************************
window = [30, 10, 5]
plt.figure(figsize=(16, 8))
plt.plot(train['average'], label='Train')
plt.plot(test['average'], label='Test')

for w in window:
    y_hat = ((stock_data['average'].iloc[::-1].rolling(w).mean()).iloc[::-1])
    plt.plot(y_hat[:n_test], label='Moving average forecast window=' + str(w))
    text = f'Moving Average: window = {w}'
    forecast_errors(text, test.average, y_hat[:n_test])

plt.legend(loc='best')
plt.show()

# **************************************Simple_Exp_smoothing*****************
plt.figure(figsize=(16, 8))


plt.plot(train['average'], label='Train')
plt.plot(test['average'], label='Test')
parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for p in parameters:
    fit1 = SimpleExpSmoothing(stock_data.average).fit(smoothing_level=p, optimized=False)
    y_hat = fit1.fittedvalues
    plt.plot(y_hat[:n_test], label='Simple exp smoothing level=' + str(p))
    text = f'Simple_Exp_Smoothing: level = {p}'
    forecast_errors(text, test.average, y_hat[:n_test])

plt.legend(loc='best')
plt.show()

# **********************DOUBLE_EXP_SMOOTHING**************************
plt.figure(figsize=(16, 8))

result = seasonal_decompose(stock_data.average, model='additive')
result.plot()
plt.show()

parameters = [[0.2, 0.1], [0.3, 0.5], [0.9, 0.9]]
plt.figure(figsize=(16, 8))
plt.plot(train['average'], label='Train')
plt.plot(test['average'], label='Test')

for p, s in parameters:
    fit1 = Holt(stock_data.average).fit(smoothing_level=p, smoothing_trend=s)
    y_hat = fit1.fittedvalues
    plt.plot(y_hat[0:n_test], label='Double exp smoothing level=' + str(p) + ' slope=' + str(s))
    text = f'Double exp smoothing: level = {p}, season = {s}'
    forecast_errors(text, test.average, y_hat[:n_test])

plt.legend(loc='best')
plt.show()

# ****************************************HOLT_WINTERES**************************************
# look for seasonal periods, trend and seas
plt.figure(figsize=(16, 8))
plt.show()
parameters = [[10, 'add', 'add']]
plt.figure(figsize=(16, 8))
plt.plot(train['average'], label='Train')
plt.plot(test['average'], label='Test')

for p, tr, seas in parameters:
    fit1 = ExponentialSmoothing(stock_data.average, seasonal_periods=p, trend=tr, seasonal=seas).fit()
    y_hat = fit1.fittedvalues
    plt.plot(y_hat[:n_test], label=f'HOLT-WINTERES periods={p} trend={tr} seasonal={seas}')
    text = f'Holt-Wi: level = {p}, trend = {tr}, season = {seas}'
    forecast_errors(text, test.average, y_hat[:n_test])
    residual = test.average - y_hat[0:n_test]

plt.legend(loc='best')
plt.show()
