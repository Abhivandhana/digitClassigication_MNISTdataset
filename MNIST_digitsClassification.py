#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns

np.random.seed(0)


# Data
# 

# In[2]:


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# Visualizing Examples

# In[10]:


num_classes = 10
f, ax = plt.subplots(1, num_classes, figsize = (20,20))
for i in range (0,num_classes):
    sample = x_train[y_train == i][0]
    ax[i].imshow(sample, cmap='gray')
    ax[i].set_title("Label: {}".format(i), fontsize=12)


# In[12]:


for i in range(10):
    print(y_train[i])


# In[13]:


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[14]:


for i in range(10):
    print(y_train[i])


# Data Preparation

# In[15]:


x_train = x_train/255.0
x_test = x_test/255.0


# In[16]:


x_train[0].shape


# In[18]:


x_train = x_train.reshape(x_train.shape[0], -1)


# In[19]:


x_train


# In[21]:


x_test = x_test.reshape(x_test.shape[0], -1)


# In[22]:


x_test


# In[23]:


x_train.shape


# In[24]:


x_test.shape


# Model - Fully Connected Neural Network

# In[27]:


model = Sequential()
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# Training Dataset

# In[29]:


batch_size = 512
epochs=10
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)


# In[30]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))


# In[31]:


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)


# In[32]:


y_pred


# In[33]:


y_pred_classes


# Testing data at random

# In[34]:


random_idx = np.random.choice(len(x_test))
x_sample = x_test[random_idx]
y_true = np.argmax(y_test, axis=1)
y_sample_true = y_true[random_idx]
y_sample_pred_class = y_pred_classes[random_idx]

plt.title("Predicted: {}, True: {}".format(y_sample_pred_class, y_sample_true), fontsize=16)
plt.imshow(x_sample.reshape(28, 28), cmap='gray')


# Plotting Confusion Matrix

# confusion_mtx = confusion_matrix(y_true, y_pred_classes)
# 
# fig, ax = plt.subplots(figsize=(15,10))
# ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues")
# ax.set_xlabel('Predicted Label')
# ax.set_ylabel('True Label')
# ax.set_title('Confusion Matrix')

# Error

# In[37]:


errors = (y_pred_classes - y_true != 0)
y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
x_test_errors = x_test[errors]


# In[38]:


y_pred_errors_probability = np.max(y_pred_errors, axis=1)
true_probability_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
diff_errors_pred_true = y_pred_errors_probability - true_probability_errors

sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)
top_idx_diff_errors = sorted_idx_diff_errors[-5:] 


# In[39]:


num = len(top_idx_diff_errors)
f, ax = plt.subplots(1, num, figsize=(30,30))

for i in range(0, num):
  idx = top_idx_diff_errors[i]
  sample = x_test_errors[idx].reshape(28,28)
  y_t = y_true_errors[idx]
  y_p = y_pred_classes_errors[idx]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title("Predicted label :{}\nTrue label: {}".format(y_p, y_t), fontsize=22)

