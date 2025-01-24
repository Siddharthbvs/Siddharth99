#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

df = pd.read_csv('Day_10_banking_data.csv')

filtered_transactions_2000 = df[df['Transaction_Amount'] <= 2000]
print("Transactions where Transaction_Amount <= 2000:")
print(filtered_transactions_2000)

loan_payment_high_balance = df[(df['Transaction_Type'] == 'Loan Payment') & (df['Account_Balance'] > 5000)]
print("\nTransactions with 'Loan Payment' and Account_Balance > 5000:")
print(loan_payment_high_balance)

uptown_transactions = df[df['Branch'] == 'Uptown']
print("\nTransactions made in 'Uptown' branch:")
print(uptown_transactions)


# In[12]:


account_type_grouped = df.groupby('Account_Type').agg(
    Total_Transaction_Amount=('Transaction_Amount', 'sum'),
    Average_Account_Balance=('Account_Balance', 'mean')
)

print("\nGrouped by Account_Type:")
print(account_type_grouped)

branch_grouped = df.groupby('Branch').agg(
    Total_Transactions=('Transaction_Amount', 'size'),
    Average_Transaction_Amount=('Transaction_Amount', 'mean')
)

print("\nGrouped by Branch:")
print(branch_grouped)


# In[ ]:




