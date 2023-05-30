import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

#pedro
# Read the data
df = pd.read_excel('/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/Final Project/Suggested Databases/Customer-Churn-Records (1).xlsx')

# Split the data into X and y
y = df['Exited']
x = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Complain','Satisfaction Score', 'Point Earned',  'Card Type']]

# One-hot encode the data to make it usable by the model (strings to numbers)
x_encoded = pd.get_dummies(x, columns=['Gender', 'Geography', 'Card Type'])

# Get the feature names after one-hot encoding
feature_names = x_encoded.columns

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

# Create the model and fit it to the training data
forest_model = RandomForestRegressor(random_state=42, max_depth=10, max_features=8, n_estimators=500)
forest_model.fit(x_train, y_train)

# Predict the test data
y_pred = forest_model.predict(x_test)

# Calculate the metrics for the selected model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ape = abs(y_pred-y_test)/y_test *100
mape = statistics.mean(ape)

# Export df columns to excel
#x_encoded.to_excel('/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/Final Project/Suggested Databases/Customer-Churn-Records (Col).xlsx', index = False)

# Calculate the feature importances
importance_scores = forest_model.feature_importances_
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

# Calculate the permutation importances
result = permutation_importance(forest_model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance_scores = result.importances_mean
permutation_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
permutation_importances = permutation_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

# Print metrics for the selected model
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}")


# Plot the actual vs. predicted values
import matplotlib.pyplot as plt

plt.scatter(y_test, y_test, color='blue', marker='o', label="Actual Values")
plt.scatter(y_test, y_pred, color='red', marker='x', label="Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.show()

#Print the feature importances and permutation importances to see which features are most important in the model.
print("Feature Importances:")
print(feature_importances)
print("Permutation Importances:")
print(permutation_importances)

# Save the decision tree as a PDF file
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(forest_model.estimators_[0], out_file=None, feature_names=feature_names, filled=True)
graph = graphviz.Source(dot_data)
graph.format = 'pdf'
graph.render('random_forest', format='pdf', view=False, cleanup=True)

# Predict the churn for a customer with the following characteristics:
new_data = {
    'CreditScore': [553],
    'Age': [41],
    'Tenure': [9],
    'Balance': [110112.54],
    'NumOfProducts': [2],
    'IsActiveMember': [0],
    'EstimatedSalary': [81898.81],
    'Complain': [0],
    'Satisfaction Score': [3],
    'Point Earned': [611],
    'Gender_Female': [0],
    'Gender_Male': [1],
    'Geography_France': [0],
    'Geography_Germany': [1],
    'Geography_Spain': [0],
    'Card Type_DIAMOND': [1],
    'Card Type_GOLD': [0],
    'Card Type_PLATINUM': [0],
    'Card Type_SILVER': [0]
}

# Create a pandas DataFrame from the dictionary
input_data = pd.DataFrame.from_dict(new_data)

# Initialize and fit the StandardScaler object
STD = StandardScaler()
STD.fit(x_train)

# Standardize the input data using the same StandardScaler used during training
input_data_std = STD.transform(input_data)

# Predict the churn using the Random Forest model
rf_predicted_output = forest_model.predict(input_data_std)

print(f"Random Forest predicted Exited: {int(rf_predicted_output)}")
