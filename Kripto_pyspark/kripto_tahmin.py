import streamlit as st
import requests
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_ta as ta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.graphics.tsaplots as sg
from scipy.stats import zscore
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
st.set_page_config(layout="wide")
st.title("Kripto Para Fiyat Tahmin Uygulaması")
st.write("Belirlenen kripto para için geçmiş verileri kullanarak fiyat tahminleri yapın.")
st.write("---")
st.markdown("### 🔑 API Anahtarı")
st.write(
    "Kripto para listesi **CoinGecko API**'den çekilir ve bu bir anahtar gerektirmez. Ancak, detaylı fiyat verilerini çekmek için **CryptoCompare API** anahtarınıza ihtiyacınız vardır.")
st.markdown(
    "API anahtarınızı almak için lütfen bu linke tıklayın: **[CryptoCompare API Kayıt](https://www.cryptocompare.com/cryptopian/api-keys)**")
st.write("---")

@st.cache_data(ttl=3600)
def get_top_100_coins():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        coins = {coin['name']: coin['symbol'].upper() for coin in data}
        st.success("Kripto para listesi CoinGecko API'den başarıyla çekildi.")
        return coins
    except (requests.exceptions.RequestException, Exception) as e:
        st.warning(f"CoinGecko API'den kripto para listesi çekilemedi: {e}")
        st.warning("Yedek liste kullanılıyor.")
        return {
            "Bitcoin": "BTC", "Ethereum": "ETH", "Tether": "USDT", "BNB": "BNB",
            "Solana": "SOL", "XRP": "XRP", "USD Coin": "USDC", "Cardano": "ADA",
            "Dogecoin": "DOGE", "Avalanche": "AVAX", "Shiba Inu": "SHIB"
        }

def get_crypto_data(crypto_symbol, time_frame, limit, api_key):
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histo{time_frame}?fsym={crypto_symbol}&tsym=USD&limit={limit}&api_key={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'Response' in data and data['Response'] == 'Error':
            st.error(f"CryptoCompare API hatası: {data['Message']}")
            return pd.DataFrame()
        ohlc_data = data['Data']['Data']
        ohlc_df = pd.DataFrame(ohlc_data)
        ohlc_df['timestamp'] = pd.to_datetime(ohlc_df['time'], unit='s')
        columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volumeto']
        ohlc_df = ohlc_df[columns_to_keep]
        ohlc_df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volumeto': 'volume'},
                       inplace=True)
        return ohlc_df
    except requests.exceptions.RequestException as e:
        st.error(f"Veri çekilirken API hatası: {e}. Lütfen API anahtarınızı kontrol edin.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Genel hata: {e}")
        return pd.DataFrame()

def prepare_data(df, lags=[1, 7, 14, 21], rolling_windows=[7, 14, 21]):
    df_copy = df.copy()
    price_zscore = np.abs(zscore(df_copy['close'].dropna()))
    df_copy['close'] = np.where(price_zscore > 3, df_copy['close'].median(), df_copy['close'])
    df_copy.index = pd.to_datetime(df_copy['timestamp'])
    df_copy['dayofweek'] = df_copy.index.dayofweek.astype(int)
    df_copy['dayofmonth'] = df_copy.index.day.astype(int)
    df_copy['month'] = df_copy.index.month.astype(int)
    for lag in lags:
        df_copy[f'lag_{lag}'] = df_copy['close'].shift(lag)
    for window in rolling_windows:
        df_copy[f'rolling_mean_{window}'] = df_copy['close'].rolling(window).mean().shift(1)
    df_copy.ta.bbands(close='close', append=True, length=20)
    df_copy.ta.rsi(close='close', append=True, length=14)
    df_copy.ta.macd(close='close', append=True)
    df_copy.drop(['timestamp', 'open', 'high', 'low', 'volume'], axis=1, inplace=True, errors='ignore')
    df_copy.dropna(inplace=True)
    return df_copy

def prepare_lstm_data(df, n_steps=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:(i + n_steps)])
        y.append(scaled_data[i + n_steps, 0])
    return np.array(X), np.array(y), scaler, df.columns

def build_lstm_model(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model_name, ohlc_df):
    min_data_points = 50
    if len(ohlc_df) < min_data_points:
        st.error(f"Model eğitimi için yeterli veri yok. En az {min_data_points} veri noktası gerekli.")
        return None

    if model_name in ["Linear Regression", "LSTM"]:
        features_df = prepare_data(ohlc_df.copy())
        if features_df.empty:
            st.error("Özellik mühendisliği sonrası veri seti boş kaldı. Veri miktarını artırın.")
            return None
        train_size = int(len(features_df) * 0.8)
        train_df = features_df.iloc[:train_size].copy()
        test_df = features_df.iloc[train_size:].copy()
    else:
        train_size = int(len(ohlc_df) * 0.8)
        train_df = ohlc_df.iloc[:train_size].copy()
        test_df = ohlc_df.iloc[train_size:].copy()

    if len(test_df) == 0:
        st.error("Test veri seti boş. Lütfen daha fazla veri çekmeyi deneyin.")
        return None

    y_pred = None
    y_test = None
    model = None

    if model_name == "Linear Regression":
        X_train = train_df.drop('close', axis=1)
        y_train = train_df['close']
        X_test = test_df.drop('close', axis=1)
        y_test = test_df['close']
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_name == "ARIMA":
        try:
            ts_train = train_df['close']
            ts_test = test_df['close']
            model = ARIMA(ts_train, order=(5, 1, 0))
            model_fit = model.fit()
            y_pred = model_fit.forecast(steps=len(ts_test))
            y_test = ts_test
        except Exception as e:
            st.error(f"ARIMA modeli eğitilirken bir hata oluştu: {e}. Veri setini küçültmeyi deneyin.")
            return None
    elif model_name == "SARIMA":
        try:
            ts_train = train_df['close']
            ts_test = test_df['close']
            model = SARIMAX(ts_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            model_fit = model.fit(disp=False)
            y_pred = model_fit.forecast(steps=len(ts_test))
            y_test = ts_test
        except Exception as e:
            st.error(
                f"SARIMA modeli eğitilirken bir hata oluştu: {e}. Veri setini veya model parametrelerini değiştirmeyi deneyin.")
            return None
    elif model_name == "LSTM":
        try:
            n_steps = 30
            X_train, y_train, scaler, features_cols = prepare_lstm_data(train_df, n_steps)
            X_test, y_test, _, _ = prepare_lstm_data(test_df, n_steps)

            if X_test.size == 0:
                st.error("LSTM için test verisi boş. Lütfen daha fazla veri çekmeyi deneyin.")
                return None

            model = build_lstm_model(n_steps, X_train.shape[2])
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping], verbose=0)
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler.inverse_transform(
                np.concatenate((y_pred_scaled, np.zeros((y_pred_scaled.shape[0], features_cols.shape[0] - 1))),
                               axis=1))[:, 0]
            y_test = scaler.inverse_transform(
                np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], features_cols.shape[0] - 1))),
                               axis=1))[:, 0]
            model.scaler = scaler
            model.n_steps = n_steps
            model.features_cols = features_cols

        except Exception as e:
            st.error(f"LSTM modeli eğitilirken bir hata oluştu: {e}. Veri setini kontrol edin.")
            return None

    if y_pred is not None and len(y_test) > 0:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.success(f"**{model_name}** modeli eğitimi tamamlandı! 🎉")
        st.info(f"RMSE: **{rmse:.2f}**, MAE: **{mae:.2f}**, R2 Skoru: **{r2:.2f}**")
        if model_name in ["ARIMA", "SARIMA"]:
            return model_fit
        return model
    else:
        return None

def run_prediction(trained_model, model_name, historical_data, forecast_steps, time_frame):
    forecast_list = []
    forecast_dates = []
    temp_df = historical_data.copy()
    confidence_intervals = None
    last_date = temp_df['timestamp'].iloc[-1]
    if time_frame == 'minute':
        delta = timedelta(minutes=1)
    elif time_frame == 'hour':
        delta = timedelta(hours=1)
    elif time_frame == 'day':
        delta = timedelta(days=1)

    if model_name == "Linear Regression":
        for i in range(forecast_steps):
            features_df = prepare_data(temp_df)
            if features_df.empty:
                st.warning("Gelecek tahmini için gerekli özellikler oluşturulamadı.")
                return [], [], None
            latest_features = features_df.iloc[-1].drop('close').to_frame().T
            latest_features = latest_features.apply(pd.to_numeric, errors='ignore')
            prediction = trained_model.predict(latest_features)[0]
            forecast_list.append(prediction)
            new_date = last_date + delta * (i + 1)
            forecast_dates.append(new_date)
            new_row = pd.DataFrame(
                [{'timestamp': new_date, 'close': prediction, 'open': prediction, 'high': prediction, 'low': prediction,
                  'volume': 0}])
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)
    elif model_name in ["ARIMA", "SARIMA"]:
        forecast_result = trained_model.get_forecast(steps=forecast_steps)
        forecast_list = forecast_result.predicted_mean.tolist()
        confidence_intervals = forecast_result.conf_int().values
        for i in range(forecast_steps):
            new_date = last_date + delta * (i + 1)
            forecast_dates.append(new_date)
    elif model_name == "LSTM":
        features_df = prepare_data(historical_data.copy())

        scaler = trained_model.scaler
        n_steps = trained_model.n_steps

        scaled_data = scaler.fit_transform(features_df.values)
        current_batch = scaled_data[-n_steps:].reshape(1, n_steps, scaled_data.shape[1])

        for i in range(forecast_steps):
            predicted_price_scaled = trained_model.predict(current_batch, verbose=0)[0][0]

            dummy_row_scaled = np.zeros((1, scaled_data.shape[1]))
            dummy_row_scaled[0, 0] = predicted_price_scaled

            predicted_price_unscaled = scaler.inverse_transform(dummy_row_scaled)[:, 0][0]

            forecast_list.append(predicted_price_unscaled)
            new_date = last_date + delta * (i + 1)
            forecast_dates.append(new_date)

            new_input_scaled = np.append(current_batch[0, 1:, :], scaled_data[-1, :].reshape(1, -1), axis=0)
            new_input_scaled[-1, 0] = predicted_price_scaled

            current_batch = new_input_scaled.reshape(1, n_steps, scaled_data.shape[1])

    return forecast_list, forecast_dates, confidence_intervals

def plot_data(historical_df, forecast_list, time_frame, forecast_dates, confidence_intervals):
    historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
    forecast_df = pd.DataFrame({'price': forecast_list, 'timestamp': forecast_dates})
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical_df['timestamp'], historical_df['close'], label='Tarihsel Fiyat', color='b', linewidth=2)
    ax.plot(forecast_df['timestamp'], forecast_df['price'], label='Tahmin', color='r', linestyle='--', marker='o',
            markersize=4)
    if confidence_intervals is not None:
        ax.fill_between(forecast_dates, confidence_intervals[:, 0], confidence_intervals[:, 1], color='r', alpha=0.1,
                        label='Tahmin Aralığı (%95 CI)')
    ax.set_title(f'{st.session_state.selected_name} Fiyat Tahmini ({time_frame.capitalize()} Bazlı)', fontsize=16)
    ax.set_xlabel('Tarih', fontsize=12)
    ax.set_ylabel('Fiyat (USD)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if time_frame == 'day':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
    elif time_frame == 'hour':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    elif time_frame == 'minute':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_analysis_graphs(prices_df, time_frame):
    ts_data = prices_df.copy()
    ts_data['timestamp'] = pd.to_datetime(ts_data['timestamp'])
    ts_data.set_index('timestamp', inplace=True)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 12))
    window = 60 if time_frame == 'minute' else 24 if time_frame == 'hour' else 30
    rolling_mean = ts_data['close'].rolling(window=window).mean()
    rolling_std = ts_data['close'].rolling(window=window).std()
    axes[0].plot(ts_data['close'], color='b', label='Fiyat')
    axes[0].plot(rolling_mean, color='r', label=f'{window} {time_frame}lık MO')
    axes[0].plot(rolling_std, color='g', label=f'{window} {time_frame}lık SS')
    axes[0].set_title('Durağanlık Testi (Hareketli İstatistikler)')
    axes[0].legend(loc='best')
    axes[0].grid(True)
    sg.plot_acf(ts_data['close'], lags=window, ax=axes[1])
    axes[1].set_title('Otorelasyon Fonksiyonu (ACF)')
    sg.plot_pacf(ts_data['close'], lags=window, ax=axes[2])
    axes[2].set_title('Kısmi Otorelasyon Fonksiyonu (PACF)')
    plt.tight_layout()
    st.pyplot(fig)

st.sidebar.header("Uygulama Ayarları")
api_key = st.sidebar.text_input("CryptoCompare API Anahtarınızı Girin:", type="password")

top_100_coins = get_top_100_coins()
if not top_100_coins:
    st.error("Kritik hata: Kripto para listesi çekilemedi ve yedek liste boş.")
    st.stop()
sorted_coin_names = sorted(list(top_100_coins.keys()))
if 'Bitcoin' in sorted_coin_names:
    default_index = sorted_coin_names.index("Bitcoin")
else:
    default_index = 0
st.session_state.selected_name = st.sidebar.selectbox("Kripto Para Seçin", sorted_coin_names, index=default_index)
crypto_symbol = top_100_coins.get(st.session_state.selected_name)
time_frame = st.sidebar.radio("Veri Periyodunu Seçin", ["day", "hour", "minute"])
if time_frame == 'day':
    limit = st.sidebar.number_input("Kaç günlük veri istersiniz?", min_value=1, max_value=2000, value=1825)
    forecast_steps = st.sidebar.number_input("Kaç günlük tahmin istersiniz?", min_value=1, max_value=30, value=7)
elif time_frame == 'hour':
    limit = st.sidebar.number_input("Kaç saatlik veri istersiniz?", min_value=1, max_value=2000, value=720)
    forecast_steps = st.sidebar.number_input("Kaç saatlik tahmin istersiniz?", min_value=1, max_value=48, value=24)
else:
    limit = st.sidebar.number_input("Kaç dakikalık veri istersiniz?", min_value=1, max_value=2000, value=2000)
    forecast_steps = st.sidebar.number_input("Kaç dakikalık tahmin istersiniz?", min_value=1, max_value=60, value=10)
models = ["LSTM","Linear Regression", "ARIMA", "SARIMA"]
model_name = st.sidebar.selectbox("Model Seçin", models)
if st.button("Analiz ve Tahmin Yap"):
    if not api_key:
        st.error("Lütfen devam etmek için CryptoCompare API anahtarınızı girin.")
        st.stop()
    with st.spinner('Veriler çekiliyor ve model eğitiliyor...'):
        ohlc_df = get_crypto_data(crypto_symbol, time_frame, limit, api_key)
        if not ohlc_df.empty:
            st.subheader("Zaman Serisi Analiz Grafikleri")
            plot_analysis_graphs(ohlc_df.copy(), time_frame)
            trained_model = train_model(model_name, ohlc_df.copy())
            if trained_model:
                st.subheader("Gelecek Fiyat Tahmini")
                forecast, forecast_dates, conf_int = run_prediction(trained_model, model_name, ohlc_df.copy(),
                                                                    forecast_steps,
                                                                    time_frame)
                if forecast:
                    plot_data(ohlc_df.copy(), forecast, time_frame, forecast_dates, conf_int)
                    st.write("### Tahminler")
                    if conf_int is not None:
                        forecast_table = pd.DataFrame({
                            'Tarih': [d.strftime('%Y-%m-%d %H:%M:%S') for d in forecast_dates],
                            'Tahmin': [f'${val:.2f}' for val in forecast],
                            'Düşük Aralığı': [f'${val:.2f}' for val in conf_int[:, 0]],
                            'Yüksek Aralığı': [f'${val:.2f}' for val in conf_int[:, 1]]
                        })
                    else:
                        forecast_table = pd.DataFrame({
                            'Tarih': [d.strftime('%Y-%m-%d %H:%M:%S') for d in forecast_dates],
                            'Tahmin': [f'${val:.2f}' for val in forecast]
                        })
                    st.dataframe(forecast_table)
                else:
                    st.warning("Tahmin yapılamadı.")
            else:
                st.error("Model eğitilemedi. Lütfen ayarları kontrol edin.")
        else:

            st.error("Veri çekilemedi. Lütfen ayarları kontrol edin.")

