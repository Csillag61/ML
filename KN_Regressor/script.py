import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

test_df = pd.read_csv('dc_airbnb_test.csv')
train_df = pd.read_csv('dc_airbnb_train.csv')
two_features = ['accommodates', 'bathrooms']
three_features = ['accommodates', 'bathrooms', 'bedrooms']
hyper_params = [x for x in range(1, 21)]
# Append the first model's MSE values to this list.
two_mse_values = list()
# Append the second model's MSE values to this list.
three_mse_values = list()
two_hyp_mse = dict()
three_hyp_mse = dict()
for p in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=p, algorithm='brute')
    knn.fit(train_df[two_features], train_df['price'])
    predictions = knn.predict(test_df[two_features])
    mse = mean_squared_error(predictions, test_df['price'])
    two_mse_values.append(mse)

two_lowest_mse = two_mse_values[0]
two_lowest_k = 1
for k, mse in enumerate(two_mse_values):
    if mse < two_lowest_mse:
        two_lowest_mse = mse
        two_lowest_k = k + 1

plt.scatter(x=hyper_params, y=two_mse_values)
plt.show()

for p in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=p, algorithm='brute')
    knn.fit(train_df[three_features], train_df['price'])
    predictions = knn.predict(test_df[three_features])
    mse_v = mean_squared_error(predictions, test_df['price'])
    three_mse_values.append(mse_v)

three_lowest_mse = three_mse_values[0]
three_lowest_k = 1
for k, mse in enumerate(three_mse_values):
    if mse < three_lowest_mse:
        three_lowest_mse = mse
        three_lowest_k = k + 1

plt.scatter(x=hyper_params, y=three_mse_values)
plt.show()

two_hyp_mse[two_lowest_k] = two_lowest_mse
three_hyp_mse[three_lowest_k] = three_lowest_mse
print(two_hyp_mse)
print(three_hyp_mse)  # itt a vege fuss el vele#