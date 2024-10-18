# -*- coding: utf-8 -*-
"""ghk_stocktransformer_v1.3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1M-fm9ASo6UsGbOC7hXVyqsegL2H2CUWV
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class Time2Vec:
    def __init__(self, kernel_size=1):
        self.k = kernel_size

    def __call__(self, inputs):
        wb = tf.Variable(tf.random.normal([inputs.shape[-1], self.k]))
        bb = tf.Variable(tf.random.normal([self.k]))
        wa = tf.Variable(tf.random.normal([inputs.shape[-1], self.k]))
        ba = tf.Variable(tf.random.normal([self.k]))

        bias = tf.matmul(inputs, wb) + bb
        dp = tf.matmul(inputs, wa) + ba
        wgts = tf.math.sin(bias) + dp

        return tf.concat([inputs, wgts], axis=-1)

data = pd.read_csv('infosys.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

period= 5

# roc
data['roc'] = ((data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)) * 100

# rsi
data['change_wrt_prevday'] = data['Close']- data['Close'].shift(1)

data['gain']=np.where(data['change_wrt_prevday'] >= 0,data['change_wrt_prevday'], 0)
data['loss']=np.where(data['change_wrt_prevday'] < 0,-data['change_wrt_prevday'], 0)

data['avg_gain'] = data['gain'].rolling(window=period).mean()
data['avg_loss'] = data['loss'].rolling(window=period).mean()

data['rsi'] = 100 - (100 / (1 + (data['avg_gain'] / data['avg_loss'])))

# Bollinger bands

data['SMA'] = data['Close'].rolling(window=period).mean()
data['SD'] = data['Close'].rolling(window=period).std()

data['UB'] = data['SMA'] + 2* data['SD']
data['LB'] = data['SMA'] - 2* data['SD']
data['bb'] = data['UB']-data['LB']
# data['Signal'] = 0
# data.loc[data['Close'] <= data['LB'], 'Signal'] = 1
# data.loc[data['Close'] >= data['UB'], 'Signal'] = -1

data = data.dropna(subset=['rsi'])

data = data.fillna(0)

data=data.reset_index(drop=True)

data['index'] = range(len(data))

features = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close', 'roc', 'rsi', 'bb', 'index']
target = 'Close'

data.head()

data = data.dropna()
data = data.reset_index(drop=True)

sequence_length = 100





data['Index'] = np.arange(len(data))

time_data = data['Index'].values.reshape(-1, 1).astype(float)
time_tensor = tf.convert_to_tensor(time_data, dtype=tf.float32)

time2vec_layer = Time2Vec(kernel_size=4)
time2vec_output = time2vec_layer(time_tensor)

time2vec_df = pd.DataFrame(time2vec_output.numpy(), columns=[f'Time2Vec{i}' for i in range(time2vec_output.shape[1])])

final_data = pd.concat([data, time2vec_df], axis=1)

final_data.head()

X = []
y = []

for i in range(len(final_data) - sequence_length):
    X.append(final_data[features].iloc[i:i+sequence_length].values)
    y.append(final_data[target].iloc[i+sequence_length])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y)

split = 0.95
X_train, X_test = X_scaled[:int(split*len(X))], X_scaled[int(split*len(X)):]
y_train, y_test = y_scaled[:int(split*len(y))], y_scaled[int(split*len(y)):]



model=Sequential()
model.add(LSTM(128,return_sequences=False,input_shape = (X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train, y_train, batch_size=128, epochs=200, validation_split=0.2, verbose=1)

y_test_pred_scaled = model.predict(X_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_scaled)

plt.figure(figsize=(25, 15))
plt.plot(y_test_actual, label='orig', alpha=0.7)
plt.plot(y_test_pred, label='pred', alpha=0.7)
plt.title('orig vs pred')
plt.xlabel('time')
plt.ylabel('close')
plt.legend()
plt.show()

from sklearn.metrics import r2_score

r2 = r2_score(y_test_actual, y_test_pred)
print(r2)

y_test_pred_scaled = model.predict(X_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

plt.figure(figsize=(25, 5))
plt.plot(y_test_actual, label='orig', alpha=0.7)
plt.plot(y_test_pred, label='pred', alpha=0.7)
plt.title('orig vs pred')
plt.xlabel('time')
plt.ylabel('close')
plt.legend()
plt.show()

r2 = r2_score(y_test_actual, y_test_pred)
print(r2)

