#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv("Simple_Linear_Regression_Sales_data.csv")


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


TV = df.TV.values.reshape(-1, 1)


# In[8]:


Sales = df.Sales.values.reshape(-1, 1)


# In[9]:


x, y = TV, Sales


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y)


# In[12]:


regressor = LinearRegression(normalize=True)


# In[13]:


regressor.fit(X_train, Y_train)


# In[15]:


y_pred = regressor.predict(X_test)


# In[18]:


y_pred[0]


# In[13]:


plt.scatter(X_train, Y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Sales_1 vs TV_1(Training set)')
plt.xlabel('TV_1')
plt.ylabel('Sales')
plt.show()


# In[19]:


regressor.score(X_test, Y_test)


# In[ ]:




