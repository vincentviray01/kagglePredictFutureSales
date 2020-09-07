
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import math
import seaborn as sns
import lightgbm as lgbm
from xgboost import XGBRegressor
from xgboost import plot_importance

new_sales_data = pd.read_pickle("new_sales_data_train.pickle")
model = XGBRegressor(max_depth=8, n_estimators=10, min_child_weight=300, colsample_by_tree=0.2, subsample=0.2, eta=0.3, seed=42, verbosity=2)
X = new_sales_data.drop('itemCountPerShopPerMonth', axis = 1)
y = new_sales_data['itemCountPerShopPerMonth']
del new_sales_data



X = X[1330200 * 12:]
y = y[1330200 * 12:]
X = X[:-1330200]
y = y[:-1330200]

X_train = X[:-1330200]
y_train = y[:-1330200]
X_val = X[-1330200:]
y_val = y[-1330200:]

del X, y

print("GOOD")
lr = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train, eval_metric="rsme", eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True, early_stopping_rounds=10)
lr.partial_fit()
# lr.fit(X_train, y_train)

with open("afs.pickle", 'wb') as pick:
    pickle.dump(model, pick)


