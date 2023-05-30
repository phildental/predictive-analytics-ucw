# Question F

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(x_train, y_train)

# Create the Random Forest model and fit it to the training data
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(x_train, y_train)

new_data = {
    'Counter Sales': [500000],  #500k on counter sales
    'Drive-through Sales': [700000], #700k on drive-through sales
    'number of customers': [0],  
    'Business Type_Burger store': [0],  
    'Business Type_Café': [0],
    'Business Type_Pizza Store': [1], #Pizza franchise
    'Location _Richmond': [1],  #Located in Richmond
    'Location _Vancouver': [0]
    
}

# Create a pandas DataFrame from the dictionary
input_data = pd.DataFrame.from_dict(new_data)

# Use the model to predict the output for the new instance
dt_predicted_output = tree_model.predict(input_data)
rf_predicted_output = rf_model.predict(input_data)

# Print net profit in CAD dollars
print(f"Decision Tree predicted net profit: ${dt_predicted_output[0]:,.2f}")
print(f"Random Forest predicted net profit: ${rf_predicted_output[0]:,.2f}")
