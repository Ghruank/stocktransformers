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
data = pd.read_csv('monthly_milk_production_1.csv')

# %%
data['Production']

# %%
X = np.empty((0, 3), dtype=int)
y = np.empty((0, 1), dtype=int)

# %%
d = data['Production']

# %%
for i in range(3, d.size):
    temp = np.array([[d[i-1], d[i-2], d[i-3]]])
    X = np.vstack([X, temp])
    temp1 = np.array([[d[i]]])
    y = np.vstack([y, temp1])


# %%
X

# %%
y

# %%
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_reshaped = X.reshape(-1, X.shape[1])
y_reshaped = y.reshape(-1, 1)

X_scaled = scaler_X.fit_transform(X_reshaped)
y_scaled = scaler_y.fit_transform(y_reshaped)

# %%
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# %%
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled, test_size=0.2, random_state=42)

# %%
model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, 3), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

# %%
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# %%
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)


# %%
y_pred_scaled = model.predict(X_reshaped)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_scaled)

# %%
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(y_actual, label='y', alpha=0.7)
plt.plot(y_pred, label='y_pred', alpha=0.7)
plt.title('original vs pred y')
plt.xlabel('timesteps')
plt.ylabel('value')
plt.legend()

# %%
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.tight_layout()
plt.show()

# %%
model.summary()

# %%



