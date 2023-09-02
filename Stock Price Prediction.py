#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Libraries


# In[1]:


import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import yfinance as yf


# In[ ]:


# Import Data


# In[2]:


df= yf.download('SOLIMAC.BO', start='2012-01-01', end='2020-01-01')

df


# In[ ]:


# Train the Data


# In[3]:


scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(df ['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range (prediction_days, len(scaled_data)):
                      x_train.append(scaled_data[x-prediction_days:x, 0])
                      y_train.append(scaled_data[x,0])
                    


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[4]:


# Build the model


# In[5]:


model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Prediction of next day/price

model.compile(optimizer='adam', loss ='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


# In[6]:


test_data = yf.download('SOLIMAC.BO', start='2020-01-01', end='2023-09-02')
actual_prices = test_data['Close'].values

total_dataset = pd.concat((df['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)


# In[7]:


test_data.tail(10)


# In[8]:


# Predictions


# In[9]:


x_test = []

for x in range (prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


# In[10]:


# Plot the prediction


# In[11]:


plt.plot(actual_prices, color= "black", label = f"Actual {'SOLIMAC.BO'} Price")
plt.plot(predicted_prices, color= "green", label = f"Predicted {'SOLIMAC.BO'} Price")
plt.title(f"{'SOLIMAC.BO'} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{'SOLIMAC.BO'} Share Price")
plt.legend()
plt.show()


# In[12]:


# Prediction for next day


# In[13]:


real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

