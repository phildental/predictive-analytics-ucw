# Question A

import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Read the data
df = pd.read_excel("/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/Assignment 2/Franchises Dataset.xlsx")


# Split the data into X and y
y = df["Net Profit"]
x = df[["Counter Sales", "Drive-through Sales", "number of customers", "Business Type", "Location "]]

# One-hot encode the data to make it usable by the model (strings to numbers)
x_encoded = pd.get_dummies(x, columns=["Business Type", "Location "])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.20, random_state=42)

# Create the model and fit it to the training data
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(x_train, y_train)

tree_model_test = DecisionTreeRegressor(random_state=42)
tree_model_test.fit(x_test, y_test)

# Predict the test data
y_pred = tree_model.predict(x_test)

# Calculate the metrics for the selected model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ape = abs(y_pred-y_test)/y_test *100
mape = statistics.mean(ape)

# Calculate the feature importances
importance_scores = tree_model.feature_importances_
feature_importances = pd.DataFrame({'Feature': x_encoded.columns, 'Importance': importance_scores})
feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

# Calculate the permutation importances
result = permutation_importance(tree_model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance_scores = result.importances_mean
permutation_importances = pd.DataFrame({'Feature': x_encoded.columns, 'Importance': importance_scores})
permutation_importances = permutation_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

# Print metrics for the selected model
print(f"Mean Absolute Error (train set): {mae:.2f}")
print(f"Mean Squared Error (train set): {mse:.2f}")
print(f"R-squared (train set): {r2:.2f}")
print(f"Mean Absolute Percentage Error (train set): {mape:.2f}")

# Calculate the metrics for the test model
y_pred_test = tree_model_test.predict(x_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
ape_test = abs(y_pred_test-y_test)/y_test *100
mape_test = statistics.mean(ape_test)

# Print metrics for the test model
print(f"Mean Absolute Error (test set): {mae_test:.2f}")
print(f"Mean Squared Error (test set): {mse_test:.2f}")
print(f"R-squared (test set): {r2_test:.2f}")
print(f"Mean Absolute Percentage Error (test set): {mape_test:.2f}")

# Grid search for hyperparameter tuning
param_grid = {
    'max_features': [3, 5, 7],
    'max_depth': [8, 10, 14],
}

grid_search = GridSearchCV(tree_model, param_grid, cv=5, n_jobs=-1)
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


# Mean Absolute Error (MAE): This metric measures the average absolute difference between the predicted values and the actual values. 
# A smaller MAE indicates better performance. An MAE of 0.03 suggests that, on average, the predictions are quite close to the actual values.

# Mean Squared Error (MSE): This metric measures the average squared difference between the predicted values and the actual values. 
# A smaller MSE indicates better performance. Since MSE is more sensitive to large errors than MAE, a low MSE of 0.01 suggests that there are no extreme outliers in the predictions.

# R-squared: This metric measures the proportion of variance in the dependent variable that is predictable from the independent variables. 
# It ranges from 0 to 1, with 1 indicating a perfect fit. An R-squared of 0.97 means that 97% of the variability in the target variable can be explained by the features used in the model, which is an excellent result.

# These metrics suggest that the model has a strong predictive performance. 
# However, it is essential to consider other factors like overfitting and the nature of the dataset before concluding that the model is indeed excellent. 
# It's adequate to cross-validate your results using techniques like k-fold cross-validation to assess the model's performance on different subsets of the data. 
# This will give you a more reliable estimate of the model's ability to generalize to new data.

#Therefore, the model appears to have high accuracy and strong predictive performance. The predicted values are very close to the actual values and provides insights into believing they have low variance and are relatively precise. A high R-squared value indicates a good fit of the model to the data.


#Question B

#Print the feature importances and permutation importances to see which features are most important in the model.
print("Feature Importances:")
print(feature_importances)
print("Permutation Importances:")
print(permutation_importances)

# Save the decision tree as a PDF file
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(tree_model, out_file=None, feature_names=x_encoded.columns, filled=True)
graph = graphviz.Source(dot_data)
graph.format = 'pdf'
graph.render('decision_tree', format='pdf', view=False, cleanup=True)




#The feature importances provide insight into the relative importance of the descriptive features in the decision tree model. According to the feature importances, the most important feature is "Business Type_Burger store" with an importance score of 0.423, followed by "Drive-through Sales" (0.247) and "Counter Sales" (0.225). This suggests that "Business Type_Burger store" has the strongest impact on the model's predictions, followed by the other two features.

#The permutation importances further support these findings. The ranking of features based on permutation importances aligns with the feature importances, confirming the significance of "Business Type_Burger store" (1.069), "Drive-through Sales" (0.478), and "Counter Sales" (0.399). These results reinforce the importance of these features in the model.

#Business Type

#The "Business Type_Café" feature also plays a role in the decision tree splits. When the business type is classified as a café and other conditions are met, the predicted net profit ranges from 0.937 to 1.655. This indicates that being a café can have a moderate positive impact on net profit.

#The "Business Type_Pizza Store" feature is not involved in any splits in the decision tree, indicating that it may have less influence on the predicted net profit compared to burger stores and cafés.

#Sales Performance:

#The "Counter Sales" and "Drive-through Sales" features are important in distinguishing different subsets of data within specific business types. The decision tree splits on these features to capture variations in net profit.

#The values of "Counter Sales" and "Drive-through Sales" determine the predicted net profit in different ranges. For example, lower "Counter Sales" values (e.g., <= 4.6) tend to be associated with lower predicted net profit values (e.g., 1.325), while higher "Drive-through Sales" values (e.g., > 2.95) can result in higher predicted net profit values (e.g., 1.861).

#Other Descriptive Features:

#The "number of customers" feature also contributes to the decision tree splits, indicating its relevance in predicting net profit. However, its impact is not as prominent as the business type and sales performance features.

#The "Location_Richmond" and "Location_Vancouver" features do not appear in any splits, suggesting that they may have limited impact on the predicted net profit in the given decision tree.

#In summary, based on the decision tree structure, the business type (specifically burger stores and cafés) and sales performance (counter sales and drive-through sales) have the most significant impact on the predicted net profit. Other descriptive features such as the number of customers and location may have a lesser but still noticeable influence
