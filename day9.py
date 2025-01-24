#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

# Load the banking data
df = pd.read_csv('Day_9_banking_data.csv')

# Task 1: Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Task 2: Use .describe() to generate basic statistics for numerical columns
print("\nBasic statistics of the numerical columns:")
print(df.describe())

# Task 3: Check for missing values in the dataset
print("\nMissing values in the dataset:")
print(df.isnull().sum())


# In[6]:


# Task 1: Group by Account_Type and calculate the total sum of Transaction_Amount and average Account_Balance
account_type_grouped = df.groupby('Account_Type').agg(
    Total_Transaction_Amount=('Transaction_Amount', 'sum'),
    Average_Account_Balance=('Account_Balance', 'mean')
)

print("\nGrouped by Account_Type:")
print(account_type_grouped)

# Task 2: Group by Branch and calculate the total number of transactions and average transaction amount
branch_grouped = df.groupby('Branch').agg(
    Total_Transactions=('Transaction_Amount', 'size'),
    Average_Transaction_Amount=('Transaction_Amount', 'mean')
)

print("\nGrouped by Branch:")
print(branch_grouped)


# In[ ]:




