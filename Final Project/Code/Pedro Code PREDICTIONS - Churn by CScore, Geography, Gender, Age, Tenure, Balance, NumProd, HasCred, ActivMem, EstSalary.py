import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from hyperopt import hp,fmin,tpe,Trials,STATUS_OK


# Read the dataset into a DataFrame
df = pd.read_excel("/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/Final Project/Suggested Databases/Customer-Churn-Records.xlsx")

# Select relevant features for prediction
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Complain','Satisfaction Score', 'Point Earned',  'Card Type']
target = ['Exited']


# Prepare the data for modeling
X = df[features]
y = df[target]

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

print(X)
