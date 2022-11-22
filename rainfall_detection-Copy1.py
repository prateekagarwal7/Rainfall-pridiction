#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[4]:


data = pd.read_csv("C:/Users/Lenovo/Desktop/austin_weather.csv")


# In[5]:


data


# In[7]:


#start with cleaning the data set 
#removing them as we to deal with the number
data = data.drop(["Events","Date","SeaLevelPressureAvgInches"],axis = 1)


# In[8]:


data = data.replace('T',0.0)


# In[9]:


data = data.replace('-',0.0)


# In[10]:


data


# In[11]:


#saving the data in new csv file
data.to_csv("austin_weather_final.csv")


# In[27]:


#imporitng the libraries for processing and analysis the data 
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[28]:


data = pd.read_csv("austin_weather_final.csv")
data


# In[29]:


X = data.drop(['PrecipitationSumInches'],axis = 1)# itPrecipitationSumInches  will serve as lebel y


# In[30]:


Y = data["PrecipitationSumInches"]


# In[31]:


#reshaping it into 2d vector

Y = Y.values.reshape(-1,1)


#The introduction of the official website of the Numpy, the -1 is understood to be unspecified value, meaning that it is not specified as given. If I only need a specific number of rows, how much the number of columns I don't care, I just need to specify the number of lines, then the number of columns directly use -1 instead of it.


# In[32]:


Y


# In[33]:


#taking a random day to observe the data in gra[h]


day_index = 798#random number
days = [i for i in range(Y.size)]
days


# In[34]:


#intialize the linear regression classifier
clf = LinearRegression()

# training the classifieer on basis of input
clf.fit(X,Y)


# In[38]:


#plot a graph 
print("the precipitation trend graph ")
plt.scatter(days,Y,color = 'g')
plt.scatter(days[day_index],Y[day_index],color = 'r')
plt.title("precipitation level")
plt.xlabel("days")
plt.ylabel("precipitation inches")
plt.show()
x_vis  =x.filter(['tempAvgF','DewPointAvgF','HumidityAvgPercent','SeaLevelPressureAvgInches','VisibilityAvgMiles','WindAvgMPH'])


# In[44]:


print("the precipitation vs attribute trend graph")

for i in range (x_vis.columns.size):
    plt.subplot(3,2,i+1)
    plt.scatter(days,x_vis[x_vis.columns.values[i][:100]], color='g')
    plt.scatter(days[day_index],x_vis[x_vis.columns.values[i]][day_index],color = 'r')
    plt.title(x_vis.columns.values[i])
plt.show()   


# In[ ]:




