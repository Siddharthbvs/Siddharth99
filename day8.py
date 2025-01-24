#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Sample data: Replace with your own data source
data = {
    'Region': ['East', 'West', 'East', 'North', 'South'],
    'Sales': [1200, 800, 1500, 600, 2000],
    'Profit': [300, 100, 400, 50, 500],
    'Quantity': [10, 5, 15, 8, 20]
}

df = pd.DataFrame(data)

filtered_df = df[df['Sales'] > 1000]

east_region_sales = df[df['Region'] == 'East']

df['Profit_Per_Unit'] = df['Profit'] / df['Quantity']

df['High_Sales'] = df['Sales'].apply(lambda x: 'Yes' if x > 1000 else 'No')

print("Filtered Sales > 1000:")
print(filtered_df)
print("\nSales in East Region:")
print(east_region_sales)
print("\nData with new columns (Profit_Per_Unit and High_Sales):")
print(df)


# In[ ]:




