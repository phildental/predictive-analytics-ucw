
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statistics

# Read the data
df = pd.read_excel("/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/week 6/House Prices (1) (2) (1) (1) (1).xls")


# Split the data into X and y
y = df["SalePrice"]
x = df[["GarageCars", "GarageArea", "OverallQual", "GrLivArea", "city"]]

# One-hot encode the data to make it usable by the model (strings to numbers)
x_encoded = pd.get_dummies(x, columns=["city"])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.20, random_state=42)

# Create the model and fit it to the training data
svm_model = SVR(kernel='linear', C=100)
svm_model.fit(x_train, y_train)

# Predict the test data
y_pred = svm_model.predict(x_test)

# Calculate the metrics for the selected model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics for the selected model
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Grid search for hyperparameter tuning
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
}

grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

# Print the metrics
print("Best hyperparameters:", grid_search.best_params_)
print("Best model:", best_model)

# Predict the test data
y_pred = best_model.predict(x_test)

APE = abs(y_pred-y_test)/y_test *100
MAPE = statistics.mean(APE)

print(f"Mean Absolute Percentage Error: {MAPE:.2f}")

#The results revealed a Mean Absolute Error (MAE) of 27429.30, indicating an average deviation of approximately $27,429 between the predicted and actual house prices. This suggests that my model's predictions closely align with the true prices on average. Furthermore, the R-squared value of 0.73 indicated that about 73% of the variability in house prices could be explained by the selected features. This highlights the significance of the chosen attributes in determining housing costs.

#To optimize the model's performance, I utilized grid search, a hyperparameter tuning technique. Through this process, I identified the best hyperparameters for the SVR model, which were a linear kernel and a regularization parameter (C) value of 100. This configuration demonstrated that a linear relationship between the features and house prices was most suitable for this dataset.

#Using the best model with the optimized hyperparameters, I achieved a mean absolute percentage error (MAPE) of 16.23%. This implies an average percentage deviation of 16.23% between the predicted and actual house prices. These findings highlight the accuracy and reliability of my model's predictions.

