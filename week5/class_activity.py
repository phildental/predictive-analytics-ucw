# Question A
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV


# Read the data
df = pd.read_excel("/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/week5/House Prices Week 5.xls")


# Split the data into X and y
y = df[ "SalePrice"]
x = df[[ "GarageCars", "GarageArea", "OverallQual", "GrLivArea", "city" ]]

# One-hot encode the data to make it usable by the model (strings to numbers)
x_encoded = pd.get_dummies(x, columns=["city"])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

# Print df to check format

# Create the model and fit it to the training data
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(x_train, y_train)

# Predict the test data
y_pred = tree_model.predict(x_test)

# Calculate the metrics
model = RandomForestRegressor(n_estimators=100, max_features=4, random_state=42, max_depth=7)
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Grid search for hyperparameter tuning (regularization)
param_grid = {
    'n_estimators': [50, 75, 100],
    'max_features': [3, 4, 5],
    'max_depth': [5, 7, 10],
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_


# Print the metrics
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print("Best hyperparameters:", grid_search.best_params_)


# Plot the actual vs. predicted values
import matplotlib.pyplot as plt

plt.scatter(y_test, y_test, color='blue', marker='o', label="Actual Values")
plt.scatter(y_test, y_pred, color='red', marker='x', label="Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.show()


#The results of the Random Forest model in predicting house prices indicate that it performs relatively well but leaves room for improvement. The Cross-Validation score, which gives an understanding of how well the model generalizes to unseen data, has a mean of approximately 0.757. This suggests that on average, the model explains about 75.7% of the variability in the target variable that is accounted for by the predictors.

#The Mean Absolute Error (MAE) is 30,856.05, which implies that the model's predictions are on average about $30,856.05 off the actual values. This is a significant amount, particularly when dealing with real estate pricing.

#The Mean Squared Error (MSE) is 2tri, a rather large value. MSE gives more weight to larger errors due to the squaring, indicating the presence of some predictions that significantly deviate from the actual values.

#The R-squared value is 0.72, which means that the model explains 72% of the variance in the house prices. An R-squared of 100% indicates that all changes in the dependent variable are completely explained by changes in the independent variable(s). Therefore, an R-squared of 72% suggests a good level of prediction, but with notable room for improvement.

#Finally, the scatter plot of actual vs. predicted values should visually represent how well the model performs. Ideally, the predicted values (red crosses) would perfectly align with the actual values (blue circles). Any deviations from this ideal line represent prediction errors.

#In summary, the model demonstrates a decent predictive ability but could likely be improved by refining feature selection, engineering new features, or tuning model hyperparameters.
