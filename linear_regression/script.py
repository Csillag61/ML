import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

#  A system of linear equations consists of multiple, related functions with a common set of variables. The word linear equation is often used interchangeably with linear function. Many real world processes can be modeled using multiple, related linear equations.
# Transform x using the equation 
#  y=30x+1000  and assign the result to y1.
# Transform x using the equation 
# y=50x+100  and assign the result to y2

x = np.linspace(0, 50, 1000)
y1 = 30 * x + 1000
y2 = 50 * x + 100

plt.plot(x, y1, c='blue')
plt.plot(x, y2, c='red')
# both functions intersect at somewhere near the point 
# (45,2200), this is a solution.

#  Gaussian elimination: using linear algebra

data = pd.read_csv('AmesHousing.txt', delimiter='\t')
train = data[0:1460]
test = data[1460:]
print(train.info())
target = "SalePrice"
 

fig = plt.figure(figsize=(7, 15))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)                    
train.plot(x='Garage Area', y='SalePrice', ax=ax1, kind='scatter')
train.plot(x='Gr Liv Area', y='SalePrice', ax=ax2, kind='scatter')
train.plot(x='Overall Cond', y='SalePrice', ax=ax3, kind='scatter')
plt.show()
# correlation between pairs of these columns using the pandas.DataFrame.corr() method:

print(train[['Garage Area', 'Gr Liv Area', 'Overall Cond', 'SalePrice']].corr())


lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
print(lr.coef_)
print(lr.intercept_)

a0 = lr.intercept_
a1 = lr.coef_



lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])


train_predictions = lr.predict(train[['Gr Liv Area']])
test_predictions = lr.predict(test[['Gr Liv Area']])

train_mse = mean_squared_error(train_predictions, train['SalePrice'])
test_mse = mean_squared_error(test_predictions, test['SalePrice'])

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print(train_rmse)
print(test_rmse)

cols = ['Overall Cond', 'Gr Liv Area']
lr.fit(train[cols], train['SalePrice'])
train_predictions = lr.predict(train[cols])
test_predictions = lr.predict(test[cols])

train_rmse_2 = np.sqrt(mean_squared_error(train_predictions, train['SalePrice']))
test_rmse_2 = np.sqrt(mean_squared_error(test_predictions, test['SalePrice']))

print(train_rmse_2)
print(test_rmse_2)

numerical_train = train.select_dtypes(include=['int', 'float'])
cols = ['PID', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Mo Sold', 'Yr Sold']
numerical_train = numerical_train.drop(numerical_train[cols], axis=1)
null_series = numerical_train.isnull().sum()
full_cols_series = null_series[null_series == 0]

len(full_cols_series) # 25

# pairwise correlation coefficients between all of the columns in train_subset

train_subset = train[full_cols_series.index]
# SalePrice column ,  absolute value of each term, sort the resulting Series by the correlation values

corrmat = train_subset.corr()
sorted_corrs = corrmat['SalePrice'].abs().sort_values()

# check on potential collinearity between feature columns with 
# correlation matrix heatmap

strong_corr = sorted_corrs[sorted_corrs > 0.3]
corrmat = train_subset[ strong_corr.index].corr()
sns.heatmap(corrmat)

final_corr_cols = strong_corr.drop(['Garage Cars', 'TotRms AbvGrd'])
print(test[final_corr_cols.index].info())
features = final_corr_cols.drop(['SalePrice']).index
target = 'SalePrice'
clean_test = test[final_corr_cols.index].dropna() #get rid of the missing values

# linearRegression model
lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])
train_predictions = lr.predict(train[features])
test_predictions = lr.predict(clean_test[features])

train_mse = mean_squared_error(train_predictions, train[target])
test_mse = mean_squared_error(test_predictions, clean_test[target])

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print(train_rmse)
print(test_rmse)



