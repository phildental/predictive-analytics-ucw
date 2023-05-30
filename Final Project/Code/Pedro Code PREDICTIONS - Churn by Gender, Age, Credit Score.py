import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statistics

from pandas import read_excel

# Read the data
df = pd.read_excel('/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/Final Project/Suggested Databases/Customer-Churn-Records.xlsx')

# Split the data into X and y
y = df ["Exited"]
x = df [["CreditScore" , "Gender" , "Age"]]

# One-hot encode the data to make it usable by the model (strings to numbers) Getting dummies for gender
x_encoded = pd.get_dummies(x, columns=['Gender'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

forest_model = RandomForestRegressor(random_state=42, max_depth=7, max_features=7, n_estimators=1000)
forest_model.fit(x_train, y_train)

# Predict the test data
y_pred = forest_model.predict(x_test)

# Calculate the metrics for the selected model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ape = abs(y_pred-y_test)/y_test *100
mape = statistics.mean(ape)


y_train = ["Exited"]
x_train = [["CreditScore" , "Gender_Male" , "Gender_Female", "Age"]]


#Predicting churn using the features "CreditScore" , "Gender_Male" , "Gender_Female", "Age"
#target low credit score 300, female, age between 18 and 68
forest_model.predict([[300,0,1,18]]) #Churn around 0.941
forest_model.predict([[300,0,1,28]]) #Churn around 0.941
forest_model.predict([[300,0,1,38]]) #Churn around 0.941
forest_model.predict([[300,0,1,48]]) #Churn around 0.941
forest_model.predict([[300,0,1,58]]) #Churn around 0.941
forest_model.predict([[300,0,1,68]]) #Churn around 0.941

#Predicting churn using the features "CreditScore" , "Gender_Male" , "Gender_Female", "Age"
#target high credit score 800, male, age between 18 and 68
forest_model.predict([[800,1,0,18]]) #Churn around 0.073
forest_model.predict([[800,1,0,28]]) #Churn around 0.073
forest_model.predict([[800,1,0,38]]) #Churn around 0.073
forest_model.predict([[800,1,0,48]]) #Churn around 0.073
forest_model.predict([[800,1,0,58]]) #Churn around 0.073
forest_model.predict([[800,1,0,68]]) #Churn around 0.073










