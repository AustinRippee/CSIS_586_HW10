#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn 
import keras


# In[2]:


from keras.datasets import cifar10
(X_train, y_train) , (X_test, y_test) = cifar10.load_data()


# In[3]:


X_train.shape


# In[4]:


y_train.shape


# In[5]:


y_train


# In[7]:


X_train[2]


# In[6]:


plt.imshow(X_train[2])
print(y_train[2])


# In[11]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
y_train


# In[12]:


X_train = X_train/255
X_test = X_test/255


# In[13]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam


# In[17]:


cnn = Sequential()
cnn.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', 
               input_shape = (32, 32, 3)))
cnn.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.25))
cnn.add(Flatten())

cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(units = 256, activation = 'relu'))
cnn.add(Dense(units = 10, activation = 'softmax'))
cnn.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam', metrics = ['accuracy'])


# In[2]:


import time

start = time.time()
history = cnn.fit(X_train, y_train, batch_size = 200, epochs = 20, shuffle = True)
end = time.time()
print(end - start)


# In[24]:


loss, accuracy = cnn.evaluate(X_test, y_test)
print(loss, accuracy)


# In[1]:


from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[ ]:





# In[ ]:




