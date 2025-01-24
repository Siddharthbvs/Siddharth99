#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
df = pd.read_csv('Day_11_banking_data.csv')

# Group by Account_Type and calculate the total sum of Transaction_Amount
account_type_sum = df.groupby('Account_Type')['Transaction_Amount'].sum()

# Plot a bar chart
account_type_sum.plot(kind='bar', color='skyblue', title='Total Transaction Amount per Account Type')

# Labeling the plot
plt.xlabel('Account Type')
plt.ylabel('Total Transaction Amount')

# Display the plot
plt.show()


# In[10]:


# Group by Branch and calculate the total sum of Transaction_Amount
branch_sum = df.groupby('Branch')['Transaction_Amount'].sum()

# Plot a pie chart
branch_sum.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue'])

# Set the title and display the plot
plt.title('Percentage of Transactions per Branch')
plt.ylabel('')  # Hides the y-axis label
plt.show()


# In[ ]:




