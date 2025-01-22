#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

# Load the sales_data.csv file into a DataFrame
df = pd.read_csv('sales_data.csv')


# In[10]:


# Task 1: Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())


# In[11]:


# Task 2: Print basic statistics of numerical columns
print("\nBasic statistics of the numerical columns:")
print(df.describe())


# In[12]:


# Task 3: Calculate the total sales for each region
total_sales_by_region = df.groupby('Region')['Sales'].sum()
print("\nTotal sales for each region:")
print(total_sales_by_region)


# In[13]:


# Task 4: Find the most sold product (based on quantity)
most_sold_product = df.groupby('Product')['Quantity'].sum().idxmax()
print(f"\nThe most sold product (based on quantity): {most_sold_product}")


# In[14]:


# Task 5: Compute the average profit margin for each product
# Profit margin = (Profit / Sales) * 100
df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100
avg_profit_margin_by_product = df.groupby('Product')['Profit Margin'].mean()
print("\nAverage profit margin for each product:")
print(avg_profit_margin_by_product)


# In[ ]:




