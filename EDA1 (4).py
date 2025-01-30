#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
data


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)
print(data.size)

data1 = data.drop(['Unnamed: 0',"Temp C"],axis =1)
data1


# In[5]:


#drop dupplication column (temp c ) and unnamed column
data1 = data.drop(['Unnamed: 0', "Temp C"], axis = 1)
data1


# In[6]:


#convert th month column data type to float data type
data1['Month'] = pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[7]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[8]:


#checking for duplicated rows in the table
#print the duplicated row(one)only
data[data1.duplicated()]


# In[9]:


#drop duplicated data
data1.drop_duplicates(keep='first', inplace =True)
data1


# In[10]:


#chhange column names (Rename the columns)
data1.rename({'Solar.R': 'Solar'},axis=1, inplace = True)
data1


# In[11]:


data1.info()


# In[12]:


#display data1 missing values count in eachcolumn using is null().sum()
data1.isnull().sum()


# In[13]:


#visualize data1 missing values graph
cols = data1.columns
colors = ['yellow','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[14]:


#find the meanand median values of each numeric column
#Imputation of missing values with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[15]:


#Replace the ozone missing values with median values
data1["Ozone"] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


#find the meanand median values of each numeric column
#Imputation of missing values with median
median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ",median_solar)
print("Mean of Solar: ",median_solar)


# In[17]:


#Replace the solar missing values with median values
data1["Solar"] = data1['Solar'].fillna(mean_solar)
data1.isnull().sum()


# In[18]:


#print the data1 5 rows
data1.head()


# In[19]:


#find the mode values of categorical columns weather
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[20]:


#impute misssing values (replace nan with mode etc)using fillna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[21]:


#impute missing values (rReplace NaN mode etc.) of "Whether" using fillna()
data1["Weathher"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[22]:


#impute missing values (rReplace NaN mode etc.) of "Month" using fillna()
mode_month = data1["Month"].mode()[0]
data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[23]:


#Reset the index
data1.reset_index(drop = True)


# In[24]:


#Detection of outliers
#Create a figure with two subplots, stacked vertically


fig, axes = plt.subplots (2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
#Plot the boxplot in the first (top) subplot 
sns.boxplot(data=data1 ["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient='h') 
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
# Plot the histogram with KDE curve in the second (bottom) subplot 
sns.histplot(data1 ["Ozone"], kde=True, ax=axes [1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
#Adjust Layout for better spacing 
plt.tight_layout() 
# Show the plot 
plt.show()


# observations
# *The Ozone column has extream values beyond 81. 
# *THe same is conformed from  the below right=skewed Histogram.

# In[25]:


#Detection of outliers
#Create a figure with two subplots, stacked vertically


fig, axes = plt.subplots (2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
#Plot the boxplot in the first (top) subplot 
sns.boxplot(data=data1 ["Solar"], ax=axes[0], color='skyblue', width=0.5, orient='h') 
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")
# Plot the histogram with KDE curve in the second (bottom) subplot 
sns.histplot(data1 ["Solar"], kde=True, ax=axes [1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")
#Adjust Layout for better spacing 
plt.tight_layout() 
# Show the plot 
plt.show()


# -It has No outliers
# -It is not perfectly Symmetric
# -It is Slightly left skewed

# In[26]:


#Create a figure for violin plot
sns.violinplot(data=data1["Ozone"],color='lightgreen')
plt.title("Violin Plot")

#Show the plot
plt.show()


# In[27]:


#exteracy outliers from boxplot for ozone column
plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert = False)
[item.get_xdata() for item in boxplot_data ['fliers']]


# In[28]:


data1["Ozone"].describe()


# In[29]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or ( x > (mu + 3*sigma))):
        print(x)


# In[30]:


import scipy.stats as stats 
# Create Q-Q plot 
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[31]:


import scipy.stats as stats 
# Create Q-Q plot 
plt.figure(figsize=(8,6))
stats.probplot(data1["Solar"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# Observations from Q-Q plot
# The data does not follow normal distribution as the data points are deviating significantly away from
# The data shows a right-skwed distribution and possible outliers

# In[32]:


#create a figure for violin plot
sns.violinplot(data = data1["Ozone"], color = 'lightgreen')
plt.title("Voilin Plot")
#show the plot
plt.show


# In[33]:


sns.violinplot(data = data1, x = "Weather", y="Ozone",palette="Set2")


# In[34]:


sns.violinplot(data = data1, x = "Ozone", y="Solar",palette="Set2")


# In[35]:


sns.violinplot(data = data1, x = "Weather", y="Solar",palette="Set2")


# In[36]:


sns.swarmplot(data = data1, x = "Weather", y="Ozone",palette="Set2")


# In[37]:


sns.stripplot(data = data1, x = "Weather", y="Ozone",palette="Set1", size=6,jitter = True)


# In[38]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"],color="black")


# In[39]:


sns.kdeplot(data=data1["Solar"], fill=True, color="blue")
sns.rugplot(data=data1["Solar"],color="black")


# In[40]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[41]:


#Compute pearson correlation coefficient
#between Wind Speed and Temp
data1["Wind"].corr(data1["Temp"])


# Observation 
# * The correction between wind and temp is observed to be negatively correlated with mild strength

# In[44]:


data1.info()


# In[45]:


#read all numeric columns into a new table
data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[46]:


#print correlation corfficinets for all the above columns

data1_numeric.corr()


# Observation 
# * The highest correlation strength is observed between Ozone and Temperaure(0.597087)
# * The next higher correlation strength is observed between Ozone and wind (-0.523738)
# * The next higher correlation strength is observed between wind and Temp(-0.441228)
# * The least correlation strength is observed between Solar and Wind(-0.055874)

# In[47]:


#plot a pair plot between all numeric columns using seaborn
sns.pairplot(data1_numeric)


# Transformations

# In[48]:


#creating dummy variable for Weather column
data2=pd.get_dummies(data1,columns=['Month','Weather'])
data2


# In[ ]:




