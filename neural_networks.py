# RNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, \
    mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler


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


stock_data = pd.read_csv('/Users/daniilsuba/Desktop/BTC-USD.csv')
stock_data = stock_data.iloc[::-1]
plt.plot(stock_data.Open[::-1], color='green', label='Open Price')
set_plot(title='Stock Open Price', axis='Time', y='USD')

stock_data = stock_data.drop(columns='Adj Close', axis=1)
stock_data.index = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d')
stock_data["average"] = (stock_data["High"] + stock_data["Low"]) / 2
stock_data.head()

stock_data.describe()
input_feature = stock_data.iloc[:, [2, 6]].values
input_data = input_feature

sc = MinMaxScaler(feature_range=(0, 1))
input_data[:, 0:2] = sc.fit_transform(input_feature[:, :])
lookback = 10

test_size = int(0.3 * len(stock_data))
X = []
y = []
for i in range(len(stock_data) - lookback - 1):
    t = []
    for j in range(lookback):
        t.append(input_data[[i + j], :])
    X.append(t)
    y.append(input_data[i + lookback, 1])
X, y = np.array(X), np.array(y)
# print(X, y)
X_test = X[:test_size]
y_test = y[:test_size]
X_train = X[test_size:]
y_train = y[test_size:]

X = X.reshape(X.shape[0], lookback, 2)
X_test = X_test.reshape(X_test.shape[0], lookback, 2)
X_train = X_train.reshape(X_train.shape[0], lookback, 2)
# print(X.shape)
# print(X_test.shape)
# print(X_train.shape)

plt.plot(stock_data.average[:test_size], color='orange', label='Train')
plt.plot(stock_data.average[test_size:], color='blue', label='Test')
set_plot(title='Stock Average Prices', axis='Time', y='Stock Opening Price')

model = Sequential()
model.add(LSTM(units=30, return_sequences=True, input_shape=(X.shape[1], 2)))
model.add(LSTM(units=30, return_sequences=True))
model.add(LSTM(units=30))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='sgd', loss=['mean_squared_error', 'mean_absolute_error'])
history = model.fit(X_train, y_train, epochs=1000, batch_size=32)
predicted_value_test = model.predict(X_test)
predicted_value_train = model.predict(X_train)

score = model.evaluate(X_test, y_test, verbose=0)
forecast_errors("", y_test, predicted_value_test)

plt.plot(input_data[:, 1][::-1], color='green', label='dataset')
plt.plot(np.concatenate((predicted_value_test, predicted_value_train))[::-1], color='yellow', label='predicted')
plt.plot(np.concatenate((predicted_value_test, predicted_value_train))[:test_size:-1], color='red', label='train')
set_plot(title='Price of stocks sold', axis='Time check', y='Stock Price')
