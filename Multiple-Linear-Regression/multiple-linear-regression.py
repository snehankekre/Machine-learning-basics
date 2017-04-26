# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 00:14:29 2017

@author: Snehan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding Independent Vraiables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Vraiable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit the mutiple linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Evaluating our model on the test set
y_pred = regressor.predict(X_test)

# Building an optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regresor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresor_OLS.summary()
X_opt = X[:, [0, 1, 2, 3, 5]]
regresor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresor_OLS.summary()
X_opt = X[:, [0, 1, 3, 5]]
regresor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresor_OLS.summary()
X_opt = X[:, [0, 5]]
regresor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresor_OLS.summary()
