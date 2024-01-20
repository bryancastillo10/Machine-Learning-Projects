### Cryptocurrency Price Prediction by LSTM###
# For Data Processing/Wrangling
import pandas as pd
import numpy as np

# For Data Visualization
import matplotlib.pyplot as plt

# For Data Mining using API
import yfinance as yfin
from pandas_datareader import data as pdr
import datetime as dt

# For Machine Learning Algorithm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

crypto_currency = "BTC"
against_currency = "USD"

yfin.pdr_override()
start = dt.datetime(2019, 1, 1)
end = dt.datetime.now()

data = pdr.get_data_yahoo(f"{crypto_currency}-{against_currency}", start, end)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

prediction_days = 60
# if we want to predict, not on the next day after 60 days, add:
# future_days = 30

x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)):  # len(scaled_data)- future_days
    x_train.append(scaled_data[x - prediction_days : x, 0])
    y_train.append(scaled_data[x, 0])  # x + future_days
## Comments in the loop are for prediction for a desired future_days

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create a Neural Network
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Setting up the Test Dataset
test_start = dt.datetime(2019, 1, 1)
test_end = dt.datetime.now()

test_data = pdr.get_data_yahoo(
    f"{crypto_currency}-{against_currency}", test_start, test_end
)
actual_prices = test_data["Close"].values

total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

# Reshaping the Model Inputs
model_inputs = total_dataset[
    len(total_dataset) - len(test_data) - prediction_days :
].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days : x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Prediction of the Crypto Prices
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color="black", label="Actual Prices")
plt.plot(prediction_prices, color="green", label="Predicted Prices")
plt.title(f"{crypto_currency} Price Prediction")
plt.xlabel("Time")
plt.ylabel(f"Price in {crypto_currency} ")
plt.legend(loc="upper left")
plt.show()

# Predict the Next Day
real_data = [
    model_inputs[len(model_inputs) + 1 - prediction_days : len(model_inputs) + 1, 0]
]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data[1]), 1)

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)
