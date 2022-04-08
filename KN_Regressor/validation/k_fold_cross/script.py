import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
#  cleaning the data
dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

dc_listings.loc[dc_listings.index[0:745], "fold"] = 1
dc_listings.loc[dc_listings.index[745:1490], "fold"] = 2
dc_listings.loc[dc_listings.index[1490:2234], "fold"] = 3
dc_listings.loc[dc_listings.index[2234:2978], "fold"] = 4
dc_listings.loc[dc_listings.index[2978:3723], "fold"] = 5

print(dc_listings['fold'].value_counts())
print("\n Num of missing values: ", dc_listings['fold'].isnull().sum())

# Training
model = KNeighborsRegressor()
train_iteration_one = dc_listings[dc_listings["fold"] != 1]
test_iteration_one = dc_listings[dc_listings["fold"] == 1].copy()
model.fit(train_iteration_one[["accommodates"]], train_iteration_one["price"])

# Predicting
labels = model.predict(test_iteration_one[["accommodates"]])
test_iteration_one["predicted_price"] = labels
iteration_one_mse = mean_squared_error(test_iteration_one["price"], test_iteration_one["predicted_price"])
iteration_one_rmse = iteration_one_mse ** (1/2)


fold_ids = [1, 2, 3, 4, 5]
def train_and_validate(df, folds):
    fold_rmses = []
    for fold in folds:
        # Train
        model = KNeighborsRegressor()
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold].copy()
        model.fit(train[["accommodates"]], train["price"])
        # Predict
        labels = model.predict(test[["accommodates"]])
        test["predicted_price"] = labels
        mse = mean_squared_error(test["price"], test["predicted_price"])
        rmse = mse**(1/2)
        fold_rmses.append(rmse)
    return(fold_rmses)

rmses = train_and_validate(dc_listings, fold_ids)
print(rmses)
avg_rmse = np.mean(rmses)
print(avg_rmse)

##### scikit-learn  cross-validation ##### 
#kf = KFold(n_splits, shuffle=False, random_state=None)
# n_splits is the number of folds you want to use,
# shuffle is used to toggle shuffling of the ordering of the observations in the dataset,
# random_state is used to specify the random seed value if shuffle is set to True

#cross_val_score(estimator, X, Y, scoring=None, cv=None)
# estimator is a sklearn model that implements the fit method (e.g. instance of KNeighborsRegressor),
# X is the list or 2D array containing the features you want to train on,
# y is a list containing the values you want to predict (target column),
# scoring is a string describing the scoring criteria (list of accepted values here).
# cv describes the number of folds. Here are some examples of accepted values:
# an instance of the KFold class,
# an integer representing the number of folds.

kf = KFold(n_splits=5, shuffle=True, random_state=1)
knn = KNeighborsRegressor()
mses=cross_val_score(knn, dc_listings[['accommodates']], dc_listings['price'], scoring='neg_mean_squared_error', cv=kf)
rmses= np.sqrt(np.absolute(mses))
avg_rmse = np.mean(rmses) 

# a k value of 2 is really just holdout validation. On the other end, setting k equal to n (the number of observations in the data set) is known as leave-one-out cross validation, or LOOCV for short. Through lots of trial and error, data scientists have converged on 10 as the standard k value.
num_folds = [3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 21, 23]

for fold in num_folds:
    kf = KFold(fold, shuffle=True, random_state=1)
    model = KNeighborsRegressor()
    mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="neg_mean_squared_error", cv=kf)
    rmses = np.sqrt(np.absolute(mses))
    avg_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    print(str(fold), "folds: ", "avg RMSE: ", str(avg_rmse), "std RMSE: ", str(std_rmse))