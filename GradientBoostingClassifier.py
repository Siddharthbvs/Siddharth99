#!/usr/bin/env python
# coding: utf-8

# #### Gradient Boosting Classifier

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[6]:


# Load dataset
df = pd.read_csv('diabetes.csv')
df


# In[7]:


# Features and target
X = df.drop('class', axis=1)
y = df['class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[8]:


# Perform train, test split on the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.8, random_state = 42)


# In[9]:


# Instantiate the model and define the parameters
gbc = GradientBoostingClassifier(random_state=42)

# Set up KFold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=gbc,
                           param_grid=param_grid,
                           cv=kfold,
                           scoring='recall',
                           n_jobs=-1,
                           verbose=1)


# In[10]:


# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Recall:", grid_search.best_score_)


# In[11]:


# Evaluate on test data using best estimator
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# #### Identify feature importance scores using XGBClassifier

# In[12]:


best_model.feature_importances_


# In[13]:


features = pd.DataFrame(best_model.feature_importances_, index = df.iloc[:,:-1].columns, columns=["Importances"])
df1= features.sort_values(by = "Importances")


# In[14]:


import seaborn as sns
sns.barplot(data = df1, x= features.index, y= "Importances", hue = features.index,palette = "Set2")


# In[15]:


import seaborn as sns

# Sort the dataframe by Importances
df1 = features.sort_values(by="Importances")

# Create the bar plot with the correct x and hue values
sns.barplot(data=df1, x=df1.index, y="Importances", hue=df1.index, palette="Set2")


# In[ ]:




