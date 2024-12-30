from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import yfinance as yf
from scipy.stats import alpha
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

#import data bitcoin
def download_bitcoin_data():
    start_date="2018-01-01"
    end_date="2024-12-29"

    data = yf.Ticker("BTC-USD")
    data = data.history(start=start_date, end=end_date)
    #hapus data yg tidak digunakan
    del data['Dividends']
    del data['Stock Splits']

    forecast_data = data.copy()
    return data, forecast_data

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

#close price visualization
def plot_close_price(data):
    plt.figure(figsize=(14, 5))
    sns.lineplot(data=data,x=data.index, y="Close", label="Close Price History", markers="o", color="blue")

    max_price = data["Close"].max()
    max_price_date=data["Close"].idxmax()

    #visualisasi harga penutupan
    plt.annotate(f'Highest Price: {max_price:.2f} USD',
                 (max_price_date, max_price),
                 xytext=(max_price_date, max_price+2),
                 # arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12,color='red')

    plt.axhline(max_price, color='red', linestyle="--", alpha=0.7)
    plt.title("Close Price History")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()

#preprecessing data
#min max scaller
def min_max_scaler(data):
    # scaler =  MinMaxScaler(feature_range=(0,1))
    # # scaled_data=scaler.fit_transform(data["Close"].values.reshape(-1,1))
    # scaled_data=scaler.fit_transform(data.values.reshape(-1,1))
    # print("\nMin-Max Scaler")
    # print(scaled_data)
    # Menginisialisasi Min-Max Scaler hanya untuk kolom 'Close'
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Melakukan scaling hanya pada kolom 'Close'
    data["Close_Scaled"] = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    return data

def create_sequence(data,sequence_length):
    sequence = []
    label = []

    for i in range(len(data)-sequence_length):
        sequence.append(data[i:(i+sequence_length)])
        label.append(data[i+sequence_length,0])

    return np.array(sequence),np.array(label)

def split_data(data,sequence_length=1):
    # sequence_length = 30
    #
    # train_size=int(len(scaled_data)*0.8)#split 80% data training
    # train_data=scaled_data[:train_size]
    # test_data=scaled_data[train_size:]#sisanya data testing
    #
    # X_train,y_train=create_sequence(train_data,sequence_length)
    # X_test,y_test=create_sequence(test_data,sequence_length)

    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data[['Close']]

    # Membagi data menjadi 80% training dan 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print("\nData Training")
    print(X_train)
    print("\nData Testing")
    print(X_test)

    return X_train,X_test,y_train,y_test

# def train_lstm_model(X_train):
#     model = Sequential()
#     model.add(Input(shape=(X_train.shape[1],1)))
#     model.add(LSTM(units=50,return_sequences=True))
#     model.add(Droupout=0.2)
#
#     model.add(LSTM(units=50,return_sequences=False))
#     model.add(Droupout=0.2)
#
#     model.add(Dense(units=25))
#     model.add(Dense(units=1))

    # model.compile(optimizer='adam',loss='mean_squared_error')
    # model.fit(X_train,y_train,epochs=100,batch_size=32)
    #
    # return model

#transformasi ke 2d
def transform_data_to_2d(data):
    data_2d = np.reshape(data, (data.shape[0],-1))
    return data_2d


def train_random_forest_model(X_train,y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    return model

def mse_evaluation(y_test,y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return mse

#predicition price vs actual price
def prediction_vs_actual_price(data):
    squence_lenght=30
    last_30_days=data['Close'][-squence_lenght:].values.reshape(-1,1)



def main():
    data,forecast_data = download_bitcoin_data()
    explore_data(data)
    # plot_close_price(data)
    scaled_data=min_max_scaler(data)
    X_train,X_test,y_train,x_test = split_data(data)

    data_training=transform_data_to_2d(X_train)
    data_testing=transform_data_to_2d(X_test)

    model = train_random_forest_model(transform_data_to_2d(data_training),transform_data_to_2d(y_train))

    prediction = model.predict(transform_data_to_2d(data_testing))
    #
    print("MSE:", mse_evaluation(data_testing,prediction))



if __name__ == "__main__":
    main()