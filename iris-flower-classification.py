#!/usr/bin/env python
# coding: utf-8

# <p style = "font-size : 50px; color : 'blue' ; font-family : 'Comic Sans MS'; text-align : center; background-color : #bedcfa; border-radius: 5px 5px;"><strong>IRIS FLOWER CLASSIFICATION</strong></p>

# ![image.png](attachment:aad7e54c-3ec1-4311-a381-778415030414.png)

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()


# In[3]:


#information about the dataset
df.info()   


# In[4]:


#describing about the dataset
df.describe()


# In[5]:


df.shape


# ### We drop the column id because it is not important.

# In[6]:


df.drop('Id',axis=1,inplace=True)


# In[7]:


df.head()


# In[8]:


#count the value
df['Species'].value_counts()


# In[9]:


#finding the null value
df.isnull().sum()


# In[10]:


import missingno as msno
msno.bar(df)


# In[11]:


df.drop_duplicates(inplace=True)


# # EDA 

# ### 1. Relationship between species and sepal length

# In[12]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Species',y='SepalLengthCm',data=df.sort_values('SepalLengthCm',ascending=False))


# ### 2. Relationship between species and sepal width

# In[13]:


df.plot(kind='scatter',x='SepalWidthCm',y='SepalLengthCm')


# ### 3. Relationship between sepal width and sepal length

# In[14]:


sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df, size=5)


# ### 4.Pairplot

# In[15]:


sns.pairplot(df, hue="Species", size=3)


# ### 5. Boxplot

# In[16]:


df.boxplot(by="Species", figsize=(12, 6))


# ### 5. Andrews_curves

# In[17]:


import pandas.plotting
from pandas.plotting import andrews_curves
andrews_curves(df, "Species")


# ### 6.CategoricalPlot

# In[18]:


plt.figure(figsize=(15,15))
sns.catplot(x='Species',y='SepalWidthCm',data=df.sort_values('SepalWidthCm',ascending=False),kind='boxen')


# ### 7.Violinplot

# In[19]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)


# # Neural Network

# In[20]:


X=df.drop('Species',axis=1)
y=df['Species']


# In[21]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


# In[22]:


df['Species'] = pd.Categorical(df.Species)
df['Species'] = df.Species.cat.codes
# Turn response variable into one-hot response vectory = to_categorical(df.response)
y = to_categorical(df.Species)


# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,stratify=y,random_state=123)


# In[24]:


model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(4,)))

model.add(Dense(3,activation='softmax'))


# In[25]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[26]:


history=model.fit(X_train,y_train,epochs=45,validation_data=(X_test, y_test))


# In[27]:


model.evaluate(X_test,y_test)


# In[28]:


pred = model.predict(X_test[:10])
print(pred)


# In[29]:


p=np.argmax(pred,axis=1)
print(p)
print(y_test[:10])


# In[30]:


history.history['accuracy']


# In[31]:


history.history['val_accuracy']


# In[32]:


plt.figure()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

