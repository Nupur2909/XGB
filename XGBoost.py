#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV


# In[6]:


data = pd.read_csv("C:/Users/nupur/Downloads/diabetes.csv")


# In[7]:


data.describe()


# In[8]:


X = data.drop(columns = 'Outcome')


# In[9]:


y = data['Outcome']


# In[13]:


train, val_train,test, val_test  = train_test_split(X, y, test_size = 0.5, random_state = 355)
#Dividing the dataset in training set and hold out set by 50%


# In[15]:


X_train, X_test, y_train, y_test  = train_test_split(train, test, test_size = 0.2, random_state = 355)


# In[16]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# In[17]:


knn.score(X_test, y_test)


# In[18]:


svm = SVC()


# In[19]:


svm.fit(X_train, y_train)


# In[20]:


svm.score(X_test, y_test)


# In[21]:


predict_val1 = knn.predict(val_train)
predict_val2 = svm.predict(val_train)


# In[23]:


predict_val = np.column_stack((predict_val1, predict_val2))


# In[30]:


predict_val2


# In[40]:


predict_test1 = knn.predict(X_test)
predict_test2 = svm.predict(X_test)
predict_test = np.column_stack((predict_test1, predict_test2))
predict_test


# In[31]:


rand_clf = RandomForestClassifier()


# In[34]:


rand_clf.fit(predict_val,val_test)


# In[41]:


rand_clf.score(predict_test, y_test)


# In[42]:


grid_param = {
    "n_estimators" : [90,100,115],
    "criterion"    : ['gini','entropy'],
    "min_samp;e_leaf": [1,2,3,4,5],
    "min_sample_split": [4,5,6,7,8],
    "max_features" :['auto','log2']
}


# In[46]:


grid_search = GridSearchCV(estimator = rand_clf, param_grid = grid_param,cv=5,n_jobs=-1,verbose = 3)


# In[47]:


val_test


# In[48]:


grid_search


# In[ ]:




