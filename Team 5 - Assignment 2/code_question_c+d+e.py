#Question C

import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# Read the data
df = pd.read_excel("/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/Assignment 2/Franchises Dataset.xlsx")

# Split the data into X and y
y = df["Net Profit"]
x = df[["Counter Sales", "Drive-through Sales", "number of customers", "Business Type", "Location "]]

# One-hot encode the data to make it usable by the model (strings to numbers)
x_encoded = pd.get_dummies(x, columns=["Business Type", "Location "])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

# Create the model and fit it to the training data
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(x_train, y_train)

# Predict the test data
y_pred = forest_model.predict(x_test)

# Calculate the metrics for the selected model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ape = abs(y_pred-y_test)/y_test *100
mape = statistics.mean(ape)

# Calculate the feature importances
importance_scores = forest_model.feature_importances_
feature_importances = pd.DataFrame({'Feature': x_encoded.columns, 'Importance': importance_scores})
feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

# Calculate the permutation importances
result = permutation_importance(forest_model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance_scores = result.importances_mean
permutation_importances = pd.DataFrame({'Feature': x_encoded.columns, 'Importance': importance_scores})
permutation_importances = permutation_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

# Print metrics for the selected model
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}")

# Grid search for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 75, 100],
    'max_features': [3, 4, 5],
    'max_depth': [5, 7, 10],
}

grid_search = GridSearchCV(forest_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

# Calculate the metrics for the Grid Search model
y_pred_best = best_model.predict(x_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
ape_best = abs(y_pred_best-y_test)/y_test *100
mape_best = statistics.mean(ape_best)

# Print the metrics for the Grid Search model
print(f"Mean Absolute Error (Grid Search model): {mae_best:.2f}")
print(f"Mean Squared Error (Grid Search model): {mse_best:.2f}")
print(f"R-squared (Grid Search model): {r2_best:.2f}")
print("Best hyperparameters (Grid Search model):", grid_search.best_params_)
print(f"Mean Absolute Percentage Error (Grid Search model): {mape_best:.2f}")


# Plot the actual vs. predicted values
import matplotlib.pyplot as plt

plt.scatter(y_test, y_test, color='blue', marker='o', label="Actual Values")
plt.scatter(y_test, y_pred, color='red', marker='x', label="Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.show()



# Question D and E

#Print the feature importances and permutation importances to see which features are most important in the model.
print("Feature Importances:")
print(feature_importances)
print("Permutation Importances:")
print(permutation_importances)

# Save the decision tree as a PDF file
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(forest_model.estimators_[0], out_file=None, feature_names=x_encoded.columns, filled=True)
graph = graphviz.Source(dot_data)
graph.format = 'pdf'
graph.render('random_forest', format='pdf', view=False, cleanup=True)

#The results of the comparative analysis between the Decision Tree and Random Forest models provide valuable insights into their respective performances in predicting net profit for franchises. The Decision Tree model achieved superior performance in terms of Mean Absolute Error (MAE) and R-squared compared to the Random Forest model. The lower MAE value of 0.03 in the Decision Tree model suggests that, on average, the predictions were closer to the actual net profit values compared to the Random Forest model, which had an MAE of 0.08. Furthermore, the Decision Tree model exhibited a higher R-squared value of 0.97, indicating that 97% of the variance in the net profit could be explained by the features used in the model. Although both models had identical Mean Squared Error (MSE) values of 0.01, suggesting similar levels of error dispersion, the Decision Tree model outperformed the Random Forest model in terms of accuracy and explained variance. These results indicate that the Decision Tree model's simpler structure captured the underlying relationships in the data more effectively, leading to better predictions. Thus, in this specific case, the Decision Tree model demonstrated stronger predictive performance and could be considered the more suitable choice for predicting net profit in franchises.

#The structure of the Decision Tree model played a crucial role in its superior predictive performance compared to the Random Forest model. The Decision Tree model's structure was determined by splitting the dataset based on the most informative features, such as "Business Type," "Drive-through Sales," and "Counter Sales." These features had the highest importance scores and permutation importances, indicating their strong influence on predicting net profit. The Decision Tree model's splits captured the variations in net profit across different subsets of the data, allowing for more accurate predictions.

#Also, the simplicity of the Decision Tree structure also contributed to its effectiveness. With fewer levels of splits, the model avoided overfitting and excessive complexity, leading to better generalization to unseen data. The feature importance analysis revealed that the "Business Type" feature, particularly "Business Type_Burger store," had the most substantial impact on net profit predictions. This finding aligns with domain knowledge, as the type of business can significantly affect profitability. Additionally, the "Drive-through Sales" and "Counter Sales" features provided valuable insights into sales performance, enabling the model to capture variations in net profit based on these metrics.

#The Random Forest model's performance was slightly lower, possibly due to its ensemble nature and the presence of multiple decision trees. While Random Forest leverages the collective wisdom of multiple trees to make predictions, it may introduce additional complexity and variance in the results. The Random Forest model's predictions were slightly less accurate, as reflected by the higher MAE and lower R-squared compared to the Decision Tree model.

#In summary, the selected structure of the Decision Tree model, driven by the most informative features and minimal complexity, contributed to its superior predictive performance. It effectively captured the relationships between the input features and net profit, resulting in more accurate predictions. The Random Forest model, although still performing reasonably well, was slightly outperformed by the Decision Tree model in this specific scenario.
