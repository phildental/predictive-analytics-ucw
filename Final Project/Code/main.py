import pandas as pd
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# Read the data
df = pd.read_excel('/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics-ucw/Final Project/Suggested Databases/Customer-Churn-Records (1).xlsx')

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
forest_model = RandomForestClassifier(random_state=42, max_depth=10, max_features=8, n_estimators=500)
forest_model.fit(x_train, y_train)

# Predict the probabilities of the test data
y_pred_proba = forest_model.predict_proba(x_test)

# The second column corresponds to the probability that the customer exits
exit_probabilities = y_pred_proba[:, 1]

# Predict the test data
y_pred = forest_model.predict(x_test)

# Calculate the metrics for the selected model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ape = abs(y_pred-y_test)/y_test *100
mape = statistics.mean(ape)

# Predict on training data
y_train_pred = forest_model.predict(x_train)

# Calculate metrics for training data
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Print metrics for training data
print(f"Training Mean Absolute Error: {mae_train:.2f}")
print(f"Training Mean Squared Error: {mse_train:.2f}")
print(f"Training R-squared: {r2_train:.2f}")

# Print metrics for test data
print(f"Test Mean Absolute Error: {mae:.2f}")
print(f"Test Mean Squared Error: {mse:.2f}")
print(f"Test R-squared: {r2:.2f}")


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

from sklearn.metrics import confusion_matrix


# Convert predicted probabilities to binary predictions
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Plot confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Calculate correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


#Print the feature importances and permutation importances to see which features are most important in the model.
print("Feature Importances:")
print(feature_importances)
print("Permutation Importances:")
print(permutation_importances)

# Save the decision tree as a PDF file
from sklearn.tree import export_graphviz
import graphviz

# Predict the churn for a customer with the following characteristics:
new_data = {
    'CreditScore': [710],
    'Age': [43],
    'Tenure': [2],
    'Balance': [140080.32],
    'NumOfProducts': [3],
    'IsActiveMember': [1],
    'EstimatedSalary': [157908.19],
    'Complain': [0],
    'Satisfaction Score': [3],
    'Point Earned': [350],
    'Gender_Female': [0], # Assuming FALSE translates to 0
    'Gender_Male': [1], # Assuming TRUE translates to 1
    'Geography_France': [0], # Assuming FALSE translates to 0
    'Geography_Germany': [1], # Assuming TRUE translates to 1
    'Geography_Spain': [0], # Assuming FALSE translates to 0
    'Card Type_DIAMOND': [1], # Assuming TRUE translates to 1
    'Card Type_GOLD': [0], # Assuming FALSE translates to 0
    'Card Type_PLATINUM': [0], # Assuming FALSE translates to 0
    'Card Type_SILVER': [0] # Assuming FALSE translates to 0
}

# Create a pandas DataFrame from the dictionary
input_data = pd.DataFrame.from_dict(new_data)

# Predict the churn using the Random Forest model
rf_predicted_probabilities = forest_model.predict_proba(input_data)

# Print probabilities for no-exit and exit
print(f"Data point: No-exit probability = {rf_predicted_probabilities[0][0]*100:.2f}%, Exit probability = {rf_predicted_probabilities[0][1]*100:.2f}%")

# Age range for which we want to make predictions
age_range = list(range(18, 91))

# Initialize a dictionary to hold the predicted probabilities
predicted_probs = {}

# Loop through the age range and predict for each age
for age in age_range:
    # Change the age in the new_data dictionary
    new_data['Age'] = [age]
    # Convert the dictionary to a pandas DataFrame
    input_data = pd.DataFrame.from_dict(new_data)
    # Predict the churn using the Random Forest model
    rf_predicted_probabilities = forest_model.predict_proba(input_data)
    # Add the probabilities to the predicted_probs dictionary
    predicted_probs[age] = rf_predicted_probabilities[0][1]  # adding only the exit probability

# Convert the predicted probabilities to a DataFrame for easier viewing
predicted_probs_df = pd.DataFrame(list(predicted_probs.items()), columns=['Age', 'Exit_Probability'])

# Plot the predicted probabilities (Age)
plt.scatter(predicted_probs_df['Age'], predicted_probs_df['Exit_Probability'])
plt.xlabel('Age')
plt.ylabel('Exit Probability')
plt.title('Exit Probability vs. Age')
plt.show()

# Age range for which we want to make predictions
numproducts_range = list(range(1, 5))

# Initialize a dictionary to hold the predicted probabilities
predicted_probs = {}

# Loop through the age range and predict for each age
for numproducts in age_range:
    # Change the age in the new_data dictionary
    new_data['NumOfProducts'] = [numproducts]
    # Convert the dictionary to a pandas DataFrame
    input_data = pd.DataFrame.from_dict(new_data)
    # Predict the churn using the Random Forest model
    rf_predicted_probabilities = forest_model.predict_proba(input_data)
    # Add the probabilities to the predicted_probs dictionary
    predicted_probs[numproducts] = rf_predicted_probabilities[0][1]  # adding only the exit probability

# Convert the predicted probabilities to a DataFrame for easier viewing
predicted_probs_df = pd.DataFrame(list(predicted_probs.items()), columns=['NumOfProducts', 'Exit_Probability'])


# Plot the predicted probabilities (Age)
plt.scatter(predicted_probs_df['NumOfProducts'], predicted_probs_df['Exit_Probability'])
plt.xlabel('NumOfProducts')
plt.ylabel('Exit Probability')
plt.title('Exit Probability vs. Nº of Products')
plt.show()