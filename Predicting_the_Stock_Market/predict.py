import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sp500 = pd.read_csv('sphist.csv')
sp500.Date = pd.to_datetime(sp500.Date)
sp500.sort_values('Date', inplace=True)
print(sp500.head())
sp500.info()
sp500.isnull().sum()

def moving_av_pr(period):
    return sp500.Close.rolling(period).mean().shift(1)

sp500["MAP5"] = moving_av_pr(5)
sp500["MAP30"] = moving_av_pr(30)
sp500["MAP365"] = moving_av_pr(365)
sp500.head(50)

sp500.dropna(inplace=True)
date_filter=sp500['Date'] >= datetime(year=2013, month=1, day=1)
train= sp500[~date_filter]
test=sp500[date_filter]

print(train.head())
print(train.tail())
print(test.head())
print(test.tail())

features=['MAP5', 'MAP30', 'MAP365']
target=['Close']
model=LinearRegression()
model.fit(train[features], train[target])
closePricePred=model.predict(test[features])
mae=mean_absolute_error(closePricePred, test[target])
rmse=mean_squared_error(closePricePred, test[target])**0.5
print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)
sp500.Close.describe()
plt.figure(figsize=(15, 10))
plt.plot(test.Date, test.Close, label="Actual price", lw = 3)
plt.plot(test.Date, test.Close, '--', c='red', label="Predicted price")
plt.title("S&P500 Index - Closing price prediction")
plt.xlabel("Date")
plt.legend()
plt.grid()
plt.show()