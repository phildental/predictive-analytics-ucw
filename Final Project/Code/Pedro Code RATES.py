import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import read_excel
read_excel('/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/Final Project/Suggested Databases/Customer-Churn-Records.xlsx')

df = pd.read_excel('/Users/felipemarques/Documents/GitHub/ucw/predictive-analytics/Final Project/Suggested Databases/Customer-Churn-Records.xlsx')

description = df.describe()
print(description)

# Group by 'Geography' and calculate churn rate
churn_by_geography = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)

# Display the region leading in churn
leading_region = churn_by_geography.index[0]

print("Region leading in churn:", leading_region)

# Filter the data for customers who churned
churned_customers = df[df['Exited'] == 1]

# Create a histogram of the ages of churned customers
plt.hist(churned_customers['Age'], bins=10, edgecolor='black')

# Set the labels and title of the histogram
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram of Age for Churned Customers')

# Display the histogram
plt.show()

# Group by 'Gender' and calculate churn rate
churn_by_gender = df.groupby('Gender')['Exited'].mean()

# Display the churn rate by gender
print(churn_by_gender)

# Group by 'Age' and calculate churn rate
churn_by_age = df.groupby('Age')['Exited'].mean()

# Display the churn rate by age
print(churn_by_age)

# Group by 'Geography' and calculate churn rate
churn_by_geography = df.groupby('Geography')['Exited'].mean()

# Display the churn rate by customer geography
print(churn_by_geography)

# Filter the data for churned customers
churned_customers = df[df['Exited'] == 1]

# Count the churned customers by geography
churn_by_geography = churned_customers['Geography'].value_counts()

# Display the count of churned customers by geography
print(churn_by_geography)

# Female churn = 1139
# Male churn = 899
# Total churn = 2038
number_of_churned_customers = 2038
total_number_of_customers = 10000

# Calculate the churn rate
churn_rate = (number_of_churned_customers / total_number_of_customers) * 100

# Display the churn rate
print("Churn Rate:", churn_rate, "%")

# Group by 'Geography' and calculate the count of churned customers
churn_distribution = df[df['Exited'] == 1]['Geography'].value_counts()

# Plotting the churn distribution as a bar chart
churn_distribution.plot(kind='bar', edgecolor='black')

# Set the labels and title of the chart
plt.xlabel('Geography')
plt.ylabel('Number of Churned Customers')
plt.title('Churn Distribution by Customer Geography')

# Display the chart
plt.show()

# Group by 'NumOfProducts' and calculate churn rate
churn_by_products = df.groupby('NumOfProducts')['Exited'].mean()

# Display the churn rate by number of products
print(churn_by_products)

# Group by 'NumOfProducts' and 'Gender' and calculate churn rate - SOMETHING WENT WRONG HERE GUYS
churn_by_products_gender = df.groupby(['Gender', 'NumOfProducts'])['Exited'].mean().unstack()

# Plotting the bar graph
churn_by_products_gender.plot(kind='bar')

# Set the labels and title of the chart
plt.xlabel('Churn rate')
plt.ylabel('Number of Products')
plt.title('Churn Rate by Number of Products, Split by Gender')

# Display the chart
plt.show()

# Group by 'Card Type' and calculate churn rate
# How could it be possible for a card type SILVER to be attributed to client 
# who has not a credit card??
churn_by_card_type = df.groupby('Card Type')['Exited'].mean()

# Display the churn rate by card type
print(churn_by_card_type)

# Group by 'Status' and calculate churn rate
churn_by_status = df.groupby('IsActiveMember')['Exited'].mean()

# Display the churn rate by status
# I got different rates on EXCEL
print(churn_by_status)

# Define the balance ranges
balance_ranges = [0, 50000, 100000, 150000, float('inf')]
labels = ['< 50k', '50k - 100k', '100k - 150k', '> 150k']

# Categorize the balance into ranges
df['Balance Range'] = pd.cut(df['Balance'], bins=balance_ranges, labels=labels)

# Group by 'Balance Range' and calculate churn rate
churn_by_balance = df.groupby('Balance Range')['Exited'].mean()

# Display the churn rate by balance range
print(churn_by_balance)

# Categorize the balance into ranges
df['Balance Range'] = pd.cut(df['Balance'], bins=balance_ranges, labels=labels)

# Group by 'Balance Range' and calculate churn rate
churn_by_balance = df.groupby('Balance Range')['Exited'].mean()

# Plotting the bar graph
churn_by_balance.plot(kind='bar')

# Set the labels and title of the chart
plt.xlabel('Balance Range')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Balance Range')

# Display the chart
# I got different rates on EXCEL
plt.show()
