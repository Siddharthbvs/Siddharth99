#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("Groceries_dataset.csv")
data


# In[ ]:


data.info()


# In[ ]:


data["itemDescription"].value_counts()


# In[ ]:


#plot a bar chart to visualize the category of Groceries data on the movies
counts = data['itemDescription'].value_counts()
plt.bar(counts.index, counts.values)


# In[ ]:


counts = data['Date'].value_counts()
plt.bar(counts.index, counts.values)


# In[ ]:


counts = data['Member_number'].value_counts()
plt.bar(counts.index, counts.values)


# In[ ]:


counts =data['itemDescription'].value_counts().reset_index()
print(counts)
plt.xlabel("itemDescription")
plt.ylabel("Counts")
plt.xticks(rotation=45, ha='right')
sns.barplot(data=counts, x='index', y='itemDescription', hue = 'index',)


# In[ ]:


# Load the dataset
data = pd.read_csv("Groceries_dataset.csv")
# Get the counts of each unique item
item_counts = data['itemDescription'].value_counts().reset_index()
# Rename columns for clarity
item_counts.columns = ['Item', 'Count']
# Plot the top 10 items by frequency
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Item', data=item_counts.head(10), palette='viridis')
# Set plot labels and title
plt.xlabel('Frequency')
plt.ylabel('Item Description')
plt.title('Top 10 Most Purchased Grocery Items')
plt.show()


# In[ ]:


# Plot 1: Line Graph (Monthly Purchases Trend Over Time)
plt.figure(figsize=(12, 8))
sns.lineplot(x='Month', y='Item Count', data=monthly_sales, hue='Year', marker='o', palette='viridis')
plt.title('Monthly Grocery Purchases Over Time (Line Graph)')
plt.xlabel('Month')
plt.ylabel('Number of Items Purchased')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.legend(title='Year')
plt.show()


# In[ ]:


# Plot 2: Bar Graph (Monthly Purchases by Year)
plt.figure(figsize=(12, 8))
sns.barplot(x='Month', y='Item Count', data=monthly_sales, hue='Year', palette='viridis')
plt.title('Monthly Grocery Purchases by Year (Bar Graph)')
plt.xlabel('Month')
plt.ylabel('Number of Items Purchased')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.legend(title='Year')
plt.show()


# In[ ]:


# Plot 3: Multi-line Graph (Trend of Purchases for Each Year)
monthly_sales_pivot = monthly_sales.pivot(index='Month', columns='Year', values='Item Count')
plt.figure(figsize=(12, 8))
monthly_sales_pivot.plot(kind='line', marker='o', figsize=(12, 8), cmap='viridis')
plt.title('Multi-line Trend of Grocery Purchases (By Year)')
plt.xlabel('Month')
plt.ylabel('Number of Items Purchased')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.legend(title='Year')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'Movie_df' has 'Date' and 'rating' columns
# Convert 'Date' to datetime format (if it's not already)
Movie_df['Date'] = pd.to_datetime(Movie_df['Date'], format='%d-%m-%Y')

# Group the data by month or year
monthly_data = Movie_df.groupby(Movie_df['Date'].dt.to_period('M')).size().reset_index(name='Item Count')

# Plot a line graph showing the trend of items purchased over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Item Count', data=monthly_data, marker='o', color='b')
plt.title('Trends of Items Purchased Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Items Purchased')
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[ ]:




