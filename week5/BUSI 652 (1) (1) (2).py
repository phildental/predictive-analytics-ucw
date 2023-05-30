#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 13:34:49 2023

@author: rushdialsaleh
"""


import pandas


df = pandas.read_excel("/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/week5/Regression (24) (3) (5).xlsx")


y = df[ "Earnings"]
x = df[[ "Cost", "Grad", "Debt", "City" ]]


y_train = y[0:93]
y_test = y[93: ]

x_train = x[0:93]
x_test = x[93: ]


import sklearn.ensemble

B_model = sklearn.ensemble.RandomForestRegressor(n_estimators=2000, max_features= 3)

B_model.fit(x_train, y_train)

# Create a DataFrame with the correct feature names for the prediction
prediction_data = pandas.DataFrame([[22000, 70, 80, 1]], columns=["Cost", "Grad", "Debt", "City"])
B_model.predict(prediction_data)


y_pred = B_model.predict(x_test)


ape = abs(y_pred - y_test)/y_test *100

mape = ape.mean()

mape



mean_cost = df["Cost"].mean()

mean_debt = df["Debt"].mean()

mean_grad = df["Grad"].mean()

mean_city = df["City"].mean()


min(df["Cost"])
max(df["Cost"])


import pandas as pd
B_model.predict(pd.DataFrame([[22000, 70, 80, 1]], columns=["Cost", "Grad", "Debt", "City"]))

B_model.predict(pd.DataFrame([[10000, mean_grad, mean_debt, mean_city]], columns=["Cost", "Grad", "Debt", "City"]))

B_model.predict(pd.DataFrame([[15000, mean_grad, mean_debt, mean_city]], columns=["Cost", "Grad", "Debt", "City"]))

B_model.predict(pd.DataFrame([[20000, mean_grad, mean_debt, mean_city]], columns=["Cost", "Grad", "Debt", "City"]))

B_model.predict(pd.DataFrame([[30000, mean_grad, mean_debt, mean_city]], columns=["Cost", "Grad", "Debt", "City"]))



B_model.feature_importances_



















y = df[ "Earnings"]
x = df[[ "Grad", "Debt" ]]


y_train = y[0:93]
y_test = y[93: ]

x_train = x[0:93]
x_test = x[93: ]


import sklearn.ensemble

B_model = sklearn.ensemble.RandomForestRegressor(n_estimators=2000, max_features= 1)

B_model.fit(x_train, y_train )



y_pred = B_model.predict(x_test)


ape = abs(y_pred - y_test)/y_test *100

mape = ape.mean()

mape

## Checking results

print("Actual Values:\n", y_test)
print("Predicted Values:\n", y_pred)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared Score:", r2)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()
