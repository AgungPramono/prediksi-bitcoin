import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Fungsi untuk preprocessing data
def preprocess_data(data):
    """Preprocess data dengan normalisasi dan memilih kolom relevan."""
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, data, scaler

# Fungsi untuk membagi data menjadi training dan testing
def split_data(scaled_data, sequence_length=1):
    """Membagi data menjadi sequence untuk training dan testing."""
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length, :-1])
        y.append(scaled_data[i + sequence_length, 3])
    X, y = np.array(X), np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

def explore_data(data,name="Dataset"):
    # Mengatur pandas untuk menampilkan semua baris
    pd.set_option('display.max_rows', None)  # Menampilkan semua baris
    pd.set_option('display.max_columns', None)  # Menampilkan semua kolom (opsional)
    pd.set_option('display.width', None)  # Menyesuaikan lebar tampilan

    print(data.head())
    print(f"\nInformasi {name}:")
    print(data.info())
    print("\nStatistik Deskriptif:")
    print(data.describe())
    print("\nPengecekan Missing Values:")
    print(data.isnull().sum())

# Fungsi untuk membuat spline pada data
def create_spline(x, y, num_points=300):
    """Membuat spline untuk data yang lebih halus."""
    x_new = np.linspace(x.min(), x.max(), num_points)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_new)
    return x_new, y_smooth

# Fungsi untuk membuat prediksi masa depan
def predict_future(model, last_scaled, scaler, raw_data, days):
    """Membuat prediksi masa depan berdasarkan model."""
    predictions = []
    for _ in range(days):
        next_pred_scaled = model.predict(last_scaled[:, :-1])
        predictions.append(next_pred_scaled[0])
        new_scaled = np.hstack((last_scaled[:, :-1], [[next_pred_scaled[0]]]))
        last_scaled = new_scaled

    future_scaled = np.column_stack([
        np.full((days, 1), raw_data['Open'].iloc[-1]),
        np.full((days, 1), raw_data['High'].iloc[-1]),
        np.full((days, 1), raw_data['Low'].iloc[-1]),
        np.array(predictions).reshape(-1, 1),
        np.full((days, 1), raw_data['Volume'].iloc[-1])
    ])
    future_prices = scaler.inverse_transform(future_scaled)[:, 3]
    return future_prices

# Fungsi utama untuk plotting hasil
def plot_results(data, y_test_actual, y_pred_actual, future_prices, days):
    """Plot hasil aktual, prediksi, dan prediksi masa depan."""
    x_actual = np.arange(len(data.index[-len(y_test_actual):]))
    x_future = np.arange(len(data.index[-len(y_test_actual):]), len(data.index[-len(y_test_actual):]) + days)

    x_actual_smooth, y_test_smooth = create_spline(x_actual, y_test_actual)
    x_actual_smooth_pred, y_pred_smooth = create_spline(x_actual, y_pred_actual)
    x_future_smooth, future_prices_smooth = create_spline(x_future, future_prices)


    plt.figure(figsize=(12, 8))
    plt.plot(x_actual_smooth, y_test_smooth, label="Actual Close (Spline)", color='blue')
    plt.plot(x_actual_smooth_pred, y_pred_smooth, label="Predicted Close (Spline)", color='red', linestyle='-')
    plt.plot(x_future_smooth, future_prices_smooth, label=f"Predicted Future ({days} Days, Spline)", color='green', linestyle='-')

    plt.title(f"Linear Regression: Bitcoin Close Price Prediction")
    plt.xlabel("Time Index")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

# Fungsi untuk plotting kolom Close
def plot_close_column(data):
    plt.figure(1)
    """Plot kolom Close dari data historis."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.title('Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Mengambil data dari Yahoo Finance
symbol = 'BTC-USD'
start_date = "2018-01-01"
end_date = "2024-12-29"
data = yf.Ticker(symbol).history(start=start_date, end=end_date)
del data['Dividends']
del data['Stock Splits']

explore_data(data)

print(data.head())

# Preprocessing data
scaled_data, raw_data, scaler = preprocess_data(data)

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = split_data(scaled_data)

# Melatih model regresi linear
model = LinearRegression()
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Prediksi dengan data testing
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Menghitung MAE dan MSE dalam persen
mae_percent = (mae / y_test.mean()) * 100
mse_percent = (mse / (y_test.mean()**2)) * 100

# Menampilkan hasil
print(f"Mean Squared Error (MSE): {mse_percent:.2f}%")
print(f"Mean Absolute Error (MAE): {mae_percent:.2f}%")
print(f"R-squared (R2): {r2:.4f}")

# Inverse transform untuk y_test dan y_pred
y_test_with_others = np.column_stack([
    np.full_like(y_test, raw_data['Open'].iloc[-1]),
    np.full_like(y_test, raw_data['High'].iloc[-1]),
    np.full_like(y_test, raw_data['Low'].iloc[-1]),
    y_test.reshape(-1, 1),
    np.full_like(y_test, raw_data['Volume'].iloc[-1])
])
y_test_actual = scaler.inverse_transform(y_test_with_others)[:, 3]

y_pred_with_others = np.column_stack([
    np.full_like(y_pred, raw_data['Open'].iloc[-1]),
    np.full_like(y_pred, raw_data['High'].iloc[-1]),
    np.full_like(y_pred, raw_data['Low'].iloc[-1]),
    y_pred.reshape(-1, 1),
    np.full_like(y_pred, raw_data['Volume'].iloc[-1])
])
y_pred_actual = scaler.inverse_transform(y_pred_with_others)[:, 3]

# Menampilkan data hasil prediksi dan actual
df_results = pd.DataFrame({
    'Actual Close Price': y_test_actual,
    'Predicted Close Price': y_pred_actual
})
pd.set_option('display.max_rows', None)
print("\nData Hasil Prediksi dan Actual:\n")
print(df_results)

# Prediksi 100 hari ke depan
days = 5
last_scaled = scaled_data[-1].reshape(1, -1)
future_prices = predict_future(model, last_scaled, scaler, raw_data, days)
print("Predicted Future Prices for the next 100 days:")
for i, price in enumerate(future_prices, 1):
    print(f"Day {i}: {price}")

# Plot hasil
plot_results(data, y_test_actual, y_pred_actual, future_prices, days)

# Plot kolom Close
plot_close_column(data)
