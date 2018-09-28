import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#setting up notebook, importing packages and reading data set
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs

import sklearn.model_selection
import sklearn.metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import sklearn.model_selection
import sklearn.metrics

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures




df = pd.read_csv('assessment_dsml1_train.csv')

pd.plotting.scatter_matrix(df, figsize=(12,12))
plt.show()


X = df[[c for c in df if c != 'y']].values
y = df['y'].values

def make_predictor(X, y):
    ss = sklearn.preprocessing.StandardScaler()
    X_scaled = ss.fit_transform(X)
    
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_ = poly.fit_transform(X_scaled)
    model = LinearRegression(fit_intercept=False, normalize=False)
    model.fit(X_, y)
    
    def predict_one(x):
        X_scaled = ss.transform([x])
        X_ = poly.fit_transform(X_scaled)
        return model.predict(X_)[0]

    predict = lambda X: np.array([predict_one(x) for x in X])

    return predict_one, predict


X = df[['x1','x2','x3','x4','x5']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

_,predict = make_predictor(X_train, y_train)
print('RMSE (train):', np.sqrt(sklearn.metrics.mean_squared_error(y_train, predict(X_train))))
print('RMSE (test):', np.sqrt(sklearn.metrics.mean_squared_error(y_test, predict(X_test))))
