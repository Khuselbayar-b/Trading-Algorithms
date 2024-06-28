import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

def fetch_stock_data(ticker, period='10y', interval='1d'):
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data

ydata = fetch_stock_data('SPY')
stock = pd.DataFrame({
    'Date': ydata.index,
    'Price': ydata['Close'],
    'Volume': ydata['Volume'] 
})
def print_data():
    plt.subplot(3, 1, 1)
    plt.plot(stock['Date'], stock['Price'], linestyle='-', color='b')
    plt.title('SPY')
    plt.xlabel('Date')
    plt.ylabel('Price')

    # Plot Inflation and Fed Rates on the same subplot
    plt.subplot(3, 1, 2)
    plt.plot(stock['Date'], stock['Volume'], linestyle='-', color='g', label='Inflation')
    plt.title('Volume')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()

def df_to_X_y2(df, window_size=8):
  df_as_np = df.drop(columns=['Date']).to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size][0]
    y.append(label)
  return np.array(X), np.array(y)

fft_values = np.fft.fft(stock['Volume'])
fft_freq = np.fft.fftfreq(len(stock['Volume']))

ifft_values = np.fft.ifft(fft_values)

stock['Volume'] = ifft_values.real
X, Y = df_to_X_y2(stock)
def min_max_scaler(X):
    # Extract the columns of interest (assuming 2nd to 5th columns)
    price_col = X[:, :, 0]
    volume_col = X[:, :, 1]

    # Initialize the scaler for each column
    price_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()

    # Reshape columns to fit the scaler
    price_col = price_col.reshape(-1, 1)
    volume_col = volume_col.reshape(-1, 1)

    # Fit and transform each column
    price_scaled = price_scaler.fit_transform(price_col)
    volume_scaled = volume_scaler.fit_transform(volume_col)

    # Reshape back to original shape
    price_scaled = price_scaled.reshape(X[:, :, 0].shape)
    volume_scaled = volume_scaled.reshape(X[:, :, 1].shape)

    # Create a copy of the original array and assign the scaled values
    X_scaled = X.copy()
    X_scaled[:, :, 0] = price_scaled
    X_scaled[:, :, 1] = volume_scaled
    return X_scaled

x_train, y_train = X[:1800], Y[:1800]
x_val, y_val = X[1800:2300], Y[1800:2300]
x_test, y_test = X[2300:], Y[2300:]

x_train = min_max_scaler(x_train)
x_val = min_max_scaler(x_val)
x_test = min_max_scaler(x_test)

model = Sequential()
model.add(InputLayer((16, 4)))  # Input layer with shape matching the data
model.add(LSTM(16))
model.add(Dropout(0.1))  # Add dropout for regularization
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Print the model summary
model.summary()

cp1 = ModelCheckpoint('model1/', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01), metrics=[RootMeanSquaredError()])

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batch_size=32, callbacks=[cp1, early_stopping])