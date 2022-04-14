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

# gradient descent algorithm works by iteratively trying different parameter values until the model with the lowest mean squared error is found

def derivative(a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(0, len_data):
        error += xi_list[i]*(a1*xi_list[i] - yi_list[i])
    deriv = 2*error/len_data
    return deriv


def gradient_descent(xi_list, yi_list, max_iterations, alpha, a1_initial):
    a1_list = [a1_initial]

    for i in range(0, max_iterations):
        a1 = a1_list[i]
        deriv = derivative(a1, xi_list, yi_list)
        a1_new = a1 - alpha*deriv
        a1_list.append(a1_new)
    return(a1_list)

param_iterations = gradient_descent(train['Gr Liv Area'], train['SalePrice'], 20, .0000003, 150)
final_param = param_iterations[-1]

def a1_derivative(a0, a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(0, len_data):
        error += xi_list[i]*(a0 + a1*xi_list[i] - yi_list[i])
    deriv = 2*error/len_data
    return deriv

def a0_derivative(a0, a1, xi_list, yi_list):
    len_data=len(xi_list)
    error=0
    for i in range(0, len_data):
        error += a0 + a1*xi_list[i] - yi_list[i]
    deriv = 2*error/len_data
    return deriv
    return 1

def gradient_descent(xi_list, yi_list, max_iterations, alpha, a1_initial, a0_initial):
    a1_list = [a1_initial]
    a0_list = [a0_initial]

    for i in range(0, max_iterations):
        a1 = a1_list[i]
        a0 = a0_list[i]
        
        a1_deriv = a1_derivative(a0, a1, xi_list, yi_list)
        a0_deriv = a0_derivative(a0, a1, xi_list, yi_list)
        
        a1_new = a1 - alpha*a1_deriv
        a0_new = a0 - alpha*a0_deriv
        
        a1_list.append(a1_new)
        a0_list.append(a0_new)
    return(a0_list, a1_list)

# Uncomment when ready.
a0_params, a1_params = gradient_descent(train['Gr Liv Area'], train['SalePrice'], 20, .0000003, 150, 1000)

#  OLS estimation/ Ordinary Least Squares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

features = ['Wood Deck SF', 'Fireplaces', 'Full Bath', '1st Flr SF', 'Garage Area',
       'Gr Liv Area', 'Overall Qual']

X = train[features]
X['bias']= 1
X=X[['bias'] + features]
y= train['SalePrice']
first_term = np.linalg.inv(
        np.dot(
            np.transpose(X), 
            X
        )
    )
second_term = np.dot(
        np.transpose(X),
        y
    )
ols_estimation = np.dot(first_term, second_term)
print(ols_estimation)

# feature engineering /  processing and trandforming features
# select data with no missing values
data = pd.read_csv('Ameshousing.txt', delimiter = '\t')
train = data[0:1460]
test = data[1460:0]
train_null_counts= train.isnull().sum()
df_no_mv = train[train_null_counts[train_null_counts == 0].index]

#  categorical features groups a specific training example into a specific category

# We can convert any column that contains no missing values (or an error will be thrown) to the categorical data type using the pandas.Series.astype() method

# need to use the .cat accessor followed by the .codes property to actually access the underlying numerical representation of a column:
## all text columns to the categorical datatype:

text_cols = df_no_mv.select_dtypes(include=['object']).columns
for col in text_cols:
    print(col+":", len(train[col].unique()))
for col in text_cols:
    train[col] = train[col].astype('category')

train['Utilities'].cat.codes.value_counts()

#   The drawback with this approach is that one of the assumptions of linear regression is violated here. Linear regression operates under the assumption that the features are linearly correlated with the target column. For a categorical feature, however, there's no actual numerical meaning to the categorical codes that pandas assigned for that column

# dummy coding : pandas.get_dummies

dummy_cols = pd.DataFrame()

for col in text_cols:
    col_dummies = pd.get_dummies(train[col])
    train = pd.concat([train, col_dummies], axis = 1)
    del train[col]


train['years_until_remod'] = train['Year Remod/Add'] - train['Year Built']   
# missing values/imputation techniques
## drop columns if 50% of the values is missing

import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

train_null_counts = train.isnull().sum()
df_missing_values = train[train_null_counts[(train_null_counts>0) & (train_null_counts<584)].index]

print(df_missing_values.isnull().sum())
print(df_missing_values.dtypes)

# For numerical columns with missing values, a common strategy is to compute the mean, median, or mode of each column and replace all missing values in that column with that value.

float_cols = df_missing_values.select_dtypes(include=['float'])
float_cols = float_cols.fillna(float_cols.mean())
print(float_cols.isnull().sum())