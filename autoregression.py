# AR Methods
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, \
    mean_absolute_percentage_error
import statsmodels.api as sm
from sklearn import preprocessing
from pmdarima.arima import auto_arima


def forecast_errors(info, test_data, real_data):
    mse = mean_squared_error(test_data, real_data)
    mae = mean_absolute_error(test_data, real_data)
    r2 = r2_score(test_data, real_data)
    mslge = mean_squared_log_error(test_data, real_data)
    mape = mean_absolute_percentage_error(test_data, real_data)
    print(info, '\nMSE = ', mse, '\nMAE =', mae,
          '\nR2 = ', r2, '\nMSLGE = ', mslge, '\nMAPE = ', mape, '\n')
    return None


stock_data = pd.read_csv('/Users/daniilsuba/Desktop/BTC-USD.csv')
stock_data = stock_data.iloc[::-1]

stock_data = stock_data.drop(columns='Adj Close', axis=1)
stock_data.index = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d')
stock_data["average"] = (stock_data["High"] + stock_data["Low"]) / 2
stock_data.head()

x = stock_data.average.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
stock_data.average = x_scaled

n_test = int(0.3*len(stock_data))
test = stock_data[0:n_test]
train = stock_data[n_test:]
plt.plot(train.average, label='Train')
plt.plot(test.average, label='Test')
plt.title("Share prices")
plt.legend()
plt.xlabel("Time")
plt.show()
y = stock_data.average[::-1]
for el in range(1, 3):
    stepwise_model = auto_arima(y, start_p=3, start_q=1,
                                max_p=4, max_q=5, m=12,
                                start_P=0, seasonal=False,
                                d=el, D=0, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
    # print(stepwise_model.aic())
#     (1, 1, 2)


def set_arima(dataset, order, train_data):
    model = sm.tsa.statespace.SARIMAX(dataset,
                                      order=tuple(order),
                                      seasonal_order=(0, 0, 0, 12),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    results_ar = model.fit()
    pred_ar = results_ar.get_prediction(start=len(train_data), dynamic=False)
    pred_ci = pred_ar.conf_int()
    plt.figure(figsize=(16, 8))
    ax = train.average.plot(label='train', color='green')
    ax = test.average.plot(label='test', color='blue')
    pred_ar.predicted_mean.plot(ax=ax, label=f'AR {order}', alpha=0.7, color='red')

    ax.set_xlabel('Time check')
    ax.set_ylabel('Share price')
    plt.legend()
    plt.show()
    y_forecasted = pred_ar.predicted_mean
    plt.figure(figsize=(16, 8))
    forecast_errors("", test.average, y_forecasted)
    return None


set_arima(dataset=y, order=(1, 0, 0), train_data=train)
set_arima(dataset=y, order=(1, 0, 2), train_data=train)
set_arima(dataset=y, order=(4, 2, 2), train_data=train)
