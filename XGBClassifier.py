#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install xgboost


# In[10]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# In[20]:


#Load data
df = pd.read_csv('diabetes.csv')
df


# In[21]:


X=df.drop('class',axis=1)
y=df['class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=42)


# In[25]:


#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)
print("--------------------------------------------------------------------")
print(X_test_scaled)


# In[26]:


xgb = XGBClassifier(use_label_encode = False, eval_metric="logloss",random_state=42)
param_grid = {
    'n_estimators':[100,150,200,300],
    'learning_rate':[0.01,0.1,0.15],
    'max_depth':[2,3,4,5],
    'subsample':[0.8,1.0],
    'colsample_bytree':[0.8,1.0]
}
#Stratified K-Fold
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           cv=skf,
                           scoring='recall',
                           n_jobs=-1,
                           verbose=1)


# In[27]:


# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Find the best model_best cross validated recall score
best_model = grid_search.best_estimator_ 
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Recall:", grid_search.best_score_)
# Predictions on test set
y_pred = best_model.predict(X_test_scaled)


# In[30]:


# Evaluate on test data using best estimator
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:




