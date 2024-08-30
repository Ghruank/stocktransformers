# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping

# %%
data = pd.read_csv('TCS.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# %%
period= 5

# %%
# roc
data['roc'] = ((data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)) * 100


# %%
# rsi
data['change_wrt_prevday'] = data['Close']- data['Close'].shift(1)

data['gain']=np.where(data['change_wrt_prevday'] >= 0,data['change_wrt_prevday'], 0)
data['loss']=np.where(data['change_wrt_prevday'] < 0,-data['change_wrt_prevday'], 0)

data['avg_gain'] = data['gain'].rolling(window=period).mean()
data['avg_loss'] = data['loss'].rolling(window=period).mean()

data['rsi'] = 100 - (100 / (1 + (data['avg_gain'] / data['avg_loss'])))

# %%
# Bollinger bands

data['SMA'] = data['Close'].rolling(window=period).mean()
data['SD'] = data['Close'].rolling(window=period).std()

data['UB'] = data['SMA'] + 2* data['SD']
data['LB'] = data['SMA'] - 2* data['SD']
data['bb'] = data['UB']-data['LB']
# data['Signal'] = 0
# data.loc[data['Close'] <= data['LB'], 'Signal'] = 1
# data.loc[data['Close'] >= data['UB'], 'Signal'] = -1

# %%
data.head(15)

# %%
data = data.dropna(subset=['rsi'])

# %%
data = data.fillna(0)

# %%
data=data.reset_index(drop=True)

# %%
data

# %%
features = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close', 'roc', 'rsi', 'bb']
target = 'Close'

# %%
data

# %%
sequence_length = 100

# %%
X = []
y = []

for i in range(len(data) - sequence_length):
    X.append(data[features].iloc[i:i+sequence_length].values)
    y.append(data[target].iloc[i+sequence_length])

# %%
X = np.array(X)
y = np.array(y).reshape(-1, 1)

# %%
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y)

# %%
s = 0.9
X_train = X_scaled[:int(s*len(X))]
X_test = X_scaled[int(s*len(X)):]
y_train = y_scaled[:int(s*len(y))]
y_test = y_scaled[int(s*len(y)):]


# %%
X_train.shape

# %%
  # model = Sequential()
# # model.add(InputLayer((100,6)))
# model.add(LSTM(128,activation='relu'))
# model.add(LSTM(128,activation='relu'))
# model.add(Dense(8,'relu'))
# model.add(Dense(1,'linear'))

model=Sequential()
model.add(LSTM(128,return_sequences=False,input_shape = (X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.005), loss='mean_squared_error')
# history=model.fit(X_train,y_train, batch_size = 128,epochs = 200,validation_split=0.2, verbose=1)

# %%
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

history = model.fit(X_train, y_train, batch_size=128, epochs=200, validation_split=0.2,
                    verbose=1, callbacks=[early_stopping])

# %% [markdown]
# 

# %%
y_test_pred_scaled = model.predict(X_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_scaled)

# %%
plt.figure(figsize=(25, 15))
plt.plot(y_test_actual, label='orig', alpha=0.7)
plt.plot(y_test_pred, label='pred', alpha=0.7)
plt.title('orig vs pred')
plt.xlabel('time')
plt.ylabel('close')
plt.legend()
plt.show()

# %%
y_test_pred_scaled = model.predict(X_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# %%
plt.figure(figsize=(25, 5))
plt.plot(y_test_actual, label='orig', alpha=0.7)
plt.plot(y_test_pred, label='pred', alpha=0.7)
plt.title('orig vs pred')
plt.xlabel('time')
plt.ylabel('close')
plt.legend()
plt.show()

# %%
train_loss = model.evaluate(X_train, y_train, verbose=0)
train_accuracy = 1 - train_loss

test_loss = model.evaluate(X_test, y_test, verbose=0)
test_accuracy = 1 - test_loss

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# %%



