#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn


# In[2]:


from keras.datasets import cifar10
(X_train, y_train) , (X_test, y_test) = cifar10.load_data()


# In[3]:


X_train.shape


# In[4]:


X_test.shape


# In[5]:


y_train.shape


# In[6]:


y_test.shape


# In[7]:


i = 2000 #image visualization
plt.imshow(X_train[i])
print(y_train[i])


# In[8]:


W_grid = 4 #visiualize images with target labels
L_grid = 4

fig, axes = plt.subplots(L_grid, W_grid, figsize = (25, 25))
axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training) #random number
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')
    
plt.subplots_adjust(hspace = 0.4)   


# In[9]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[10]:


number_cat = 10


# In[11]:


y_train


# In[12]:


import keras
y_train = keras.utils.to_categorical(y_train, number_cat) #conversion to catagory values
y_test = keras.utils.to_categorical(y_test, number_cat)


# In[13]:


y_test


# In[14]:


X_train = X_train/255 #normalize the data
X_test = X_test/255


# In[15]:


X_train


# In[16]:


X_train.shape


# In[17]:


Input_shape = X_train.shape[1:]


# In[18]:


Input_shape #extract input image size


# In[19]:


#model building
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout #convoulision - downsampling - downsampling - connect the network - flatten the array - data regularization
from keras.optimizers import Adam #obtain the weights
from keras.callbacks import TensorBoard


# In[20]:


cnn_model = Sequential()
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = Input_shape))
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.4))

cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.4))

cnn_model.add(Flatten())
cnn_model.add(Dense(units = 512, activation = 'relu'))

cnn_model.add(Dense(units = 512, activation = 'relu'))

cnn_model.add(Dense(units = 10, activation = 'softmax'))


# In[21]:


cnn_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.RMSprop(learning_rate = 0.001), metrics = ['accuracy'])


# In[22]:


history = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 1, shuffle = True)


# In[23]:


evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(evaluation[1]))


# In[24]:


predicted_classes = cnn_model.predict(X_test)
predicted_classes


# In[25]:


y_test


# In[26]:


y_test = y_test.argmax(1)


# In[27]:


y_test


# In[30]:


L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('True = {}'.format(y_test[i]))
    axes[i].axis('off')
    
plt.subplots_adjust(wspace = 1)


# In[ ]:




