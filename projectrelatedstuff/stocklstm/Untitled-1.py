# %%
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv('hul.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')


# %%
features = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close']
target = 'Close'

# %%
sequence_length = 100

# %%
data['Date']

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
X

# %%
y

# %%
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y)

# %%
X_scaled

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.05, random_state=42)

# %%
model = Sequential([
    LSTM(64, activation='relu', input_shape=(sequence_length, len(features)), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1)
])

# %%
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# %%
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2, verbose=1)

# %%
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_scaled)

# %%
y_test.shape

# %%
y_scaled.shape

# %%
X_scaled.shape

# %%
X_test.shape

# %%
n_test = len(y_test)

# test_dates = data['Date'].iloc[-n_test:]

plt.figure(figsize=(25, 5))
plt.subplot(1, 2, 1)
plt.plot( y_actual, label='Actual', alpha=0.7)
plt.plot( y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Close Price (Test Set)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()


# %%
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# %%
model.summary()


