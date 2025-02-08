#!/usr/bin/env python
# coding: utf-8

# # Asssumptipns in multiple Linear Regression
# 1. Linearity:The Relationship b/w the predicators and the response is linear
# 2. Independence:Obsevations are independent of each other.
# 3. Homoscendasticity: The residual(differeces b/w observed and predicated values) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Errors: The residuals of the model are normally distributed.
# 5. No Multicolinearity: The Independent variable should not be too highly correleated with ech other.

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


#Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


#rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Descrption of columns
# - HP : Horse power of the car
# - MPG : Millage of the car(Miles Per Gallon)
# - VOL : Volume of the car(Size)
# - SP : Top speed of the car(Miles per hour)
# - WT : Weight of the car

# In[4]:


cars.info()


# In[5]:


#check for missing values
cars.isna().sum()


# # observations
# - There are no missing values 
# - There are 81 obseravtions
# - The dataa types of the columns are relevant and valid

# In[6]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# In[7]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()            


# In[8]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# In[9]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# In[10]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# # Observation
# - There are some extreme values(outliers) observed in towards the right tail of SP and HP distrubitions.
# - In vol and WT mcolumns , a few outliers are observed in both tails of their distributions,
# - The extreme values of car data may have come from the specially designed nature of cars 
# - As this is multi-dimensional data,the outliers with respect to spatial dimensions may have  to be considered while bulding the regression 

# # CHECKING FOR DUBLICATE 

# In[11]:


cars[cars.duplicated()]


# In[12]:


# Pair plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[13]:


cars.corr()


# # Observations
# - High Positive Correlation between VOL and WT 
# - Negative Correlation between MPG and other variables
# - Strong Positive Correlation between HP and SP
# - The highest corr among x columns is not desirable as it might lead to multicollinearity problem

# In[14]:


#Build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~+VOL+SP+HP',data=cars).fit()


# In[15]:


model1.summary()


# # Observations
# - The R -squared and adjucent R-Suared value are Good and about 75% of variablity in Y is explained by X columns
# - The probability values with respect to F-satistic is close to zero, including that all or of X columns are significant 
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored 

# # Performance metrics for model 1

# In[16]:


#Find the performance matrics
#create a data frme with actual y and predicted y columns
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[17]:


# predict for the given X data columns
pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[18]:


# compute the Mean Squared Error(MSE) dor model1
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("MSE: ",mse)
print("RMSE: ",np.sqrt(mse))


# # Checking for multicollinearity among X-columns using VIF method

# In[19]:


cars.head()


# In[20]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# # Observation
# - The ideal range of VIF values shall be b/w 0 to 10. However slightly higher values can be tolerated
# - As seen from the very high VIF values for VOL and WT ,it is clear that they are prone to multicollinearity
# - hence it is decided to drop one of the column(either VOl ot WT) to overcome the multicollinearity
# - It is decided to drop WT and retain VOL column in further models.

# In[21]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[22]:


#Bulid model2
import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[23]:


model2.summary()


# # Performance metrics for model2

# In[24]:


#Find the performance metric for model2
#create a data frame with actual y and predicted y 
df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[25]:


# predict for the given X data columns
pred_y2 = model2.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[26]:


# compute the Mean Squared Error(MSE) dor model2
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"],df2["pred_y2"])
print("MSE: ",mse)
print("RMSE: ",np.sqrt(mse))


# # Observations

# - The adjucent R-squared value improved slightly to 0.76
# - All the p-values for mmodel parameters sre less thsn 5% hence  they are significanf
# - therefore the HP,VOL,SP columns are finalized as the significent predictor for the MPG
# - There is no improvement in MSE value

# # Identification of High Influence points(spatial outliers)

# In[27]:


cars1.shape


# In[28]:


k = 3 # No of X-columns in cars1
n = 81 # No of obserevstions
leverage_cutoff = 3 *((k + 1)/n)
leverage_cutoff


# In[29]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1,alpha=.05)
y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# # Observations
# - From the above plot, it is evidient that points 65,70,76,78,79,80 are the influencers.
# - As their H leverage values are higher and size is higher

# In[30]:


cars1[cars1.index.isin([65,70,78,79,80])]


# In[31]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[32]:


cars2


# # Build Model3 on cars2 dataset

# In[33]:


model3= smf.ols('MPG~VOL+SP+HP' ,data = cars2).fit()


# In[34]:


model3.summary()


# # Performance Metric for model3

# In[35]:


df3= pd.DataFrame()
df3["actual_y3"] = cars2["MPG"]
df3.head()


# In[37]:


#predict on all X data columns
pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[39]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :",mse )
print("RMSE :",np.sqrt(mse))


# # Comparision of moelds
#  |* Metrics        |* Model 1|* Model 2|* Model 3|
#  |-----------------|---------|---------|---------|
#  | R-squaredq      | 0.771   | 0.770   | 0.885   |
#  | Adj. R-squared  | 0.758   | 0.761   | 0.880   |
#  | MSE             | 18.89   | 18.91   | 8.68    |
#  | RMSE            | 4.34    | 4.34    | 2.94    |

# #### Comparison of models
# 
#  
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
#  
# 
# - *From the above comparison table, it is observed that Model 3 is the best among the three.*

# # Check the visibility of model assumptions for model3

# In[44]:


model3.resid


# In[45]:


model3.fittedvalues


# In[46]:


# The model is built with VOL,SP,HP,by ignoring WT
import statsmodels.api as sm
qqplot=sm.qqplot(model3.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[52]:


sns.displot(model3.resid, kde = True)


# In[54]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[55]:


plt.figure(figsize=(6,4))
plt.scatter(get_standardized_values(model3.fittedvalues),
            get_standardized_values(model3.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[ ]:




