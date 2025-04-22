# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from scipy.stats import boxcox
# from scipy.special import inv_boxcox
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping
# from utils import load_saved_dataset

# st.title("Forecasting dengan Hybrid SARIMAX + BLSTM")

# # Cek apakah data tersedia, jika belum coba load dari file
# if "data" not in st.session_state:
#     df, source = load_saved_dataset()
#     if df is not None:
#         st.session_state["data"] = df
#         st.session_state["source"] = source
#     else:
#         if st.session_state.get("role") == "admin":
#             st.warning("Silakan upload file terlebih dahulu di halaman admin")
#         else:
#             st.warning("Admin Belum Mengupload File Dataset!")
#         st.stop()

# data = st.session_state["data"]

# # --- Parameter Filter ---
# st.sidebar.header("Parameter Filter")
# start_date = st.sidebar.text_input("Start Date (YYYY-MM)", "2020-01")
# end_date = st.sidebar.text_input("End Date (YYYY-MM)", "2024-12")
# # brand_filter = st.sidebar.multiselect("Filter Brand", options=data['customers'].unique(), default=['Toys Kingdom'])
# brand_filter = st.sidebar.multiselect("Filter Customers", options=data['customers'].unique())

# # --- Validasi Brand Filter ---
# if not brand_filter:
#     st.warning("Mohon diisi untuk Customers-nya terlebih dahulu di sidebar.")
#     st.stop()

# # --- Parameter Model ---
# st.sidebar.header("SARIMAX Parameters")
# p = st.sidebar.number_input("p (AR)", 0, 5, 0)
# d = st.sidebar.number_input("d (I)", 0, 2, 2)
# q = st.sidebar.number_input("q (MA)", 0, 5, 2)
# P = st.sidebar.number_input("P (Seasonal AR)", 0, 5, 2)
# D = st.sidebar.number_input("D (Seasonal I)", 0, 2, 1)
# Q = st.sidebar.number_input("Q (Seasonal MA)", 0, 5, 2)
# S = st.sidebar.number_input("S (Seasonal Period)", 1, 24, 12)

# st.sidebar.header("BLSTM Parameters")
# look_back = st.sidebar.slider("Look Back", 1, 60, 24)
# batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1)
# epochs = st.sidebar.slider("Epochs", 10, 200, 100)
# units = st.sidebar.slider("LSTM Units", 8, 256, 16)

# # --- Filter Data ---
# data['date'] = pd.to_datetime(data['date'], format='%m-%Y')
# data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]
# if 'All' not in brand_filter:
#     data = data[data['customers'].isin(brand_filter)]

# df = data.groupby('date').agg({'total_sales': 'sum', 'qty': 'sum'}).reset_index()

# # --- Visualisasi awal ---
# st.subheader("Total Sales per Bulan")
# st.line_chart(df.set_index('date')['total_sales'])

# # --- Box-Cox + Differencing ---
# df['total_sales_boxcox'], lam_sales = boxcox(df['total_sales'])
# df['qty_boxcox'], lam_qty = boxcox(df['qty'])
# df['sales_diff'] = pd.Series(df['total_sales_boxcox']).diff()
# df['qty_diff'] = pd.Series(df['qty_boxcox']).diff()
# df.dropna(inplace=True)

# # --- ACF & PACF ---
# with st.expander("Lihat ACF & PACF Plot"):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
#     plot_acf(df['sales_diff'], ax=ax1)
#     plot_pacf(df['sales_diff'], ax=ax2, method='ywm')
#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

# # --- Split data ---
# train = df.iloc[:-int(len(df)*0.2)]
# test = df.iloc[-int(len(df)*0.2):]

# # --- SARIMAX Model ---
# with st.spinner("Menjalankan model SARIMAX..."):
#     sarimax_model = SARIMAX(
#         train['total_sales_boxcox'],
#         exog=train[['qty_boxcox']],
#         order=(p, d, q),
#         seasonal_order=(P, D, Q, S),
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     ).fit()

#     boxcox_forecast = sarimax_model.forecast(steps=len(test), exog=test[['qty_boxcox']])
#     boxcox_forecast[boxcox_forecast <= 0] = 1e-3
#     sarimax_forecast = inv_boxcox(boxcox_forecast, lam_sales)

# # --- BLSTM Helper Function ---
# def create_blstm_model(input_shape, units=128):
#     model = Sequential()
#     model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
#     model.add(LSTM(units))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     return model

# def prepare_lstm_data(series, look_back=24):
#     X, y = [], []
#     for i in range(look_back, len(series)):
#         X.append(series[i-look_back:i])
#         y.append(series[i])
#     return np.array(X), np.array(y)

# def inverse_diff(preds, last_val):
#     return np.r_[last_val, preds].cumsum()[1:]

# def blstm_forecasting(train_df, test_df, look_back=24, units=16, batch_size=16, epochs=100):
#     full_series = pd.concat([train_df, test_df])
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(full_series.values.reshape(-1, 1)).flatten()
#     train_scaled = scaled[:len(train_df)]
#     test_scaled = scaled[len(train_df):]

#     X_train, y_train = prepare_lstm_data(train_scaled, look_back)
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

#     model = create_blstm_model((look_back, 1), units=units)
#     es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

#     X_test = []
#     input_seq = scaled[len(train_df)-look_back:len(train_df)]
#     for i in range(len(test_df)):
#         X_test.append(input_seq[-look_back:])
#         next_pred = model.predict(input_seq[-look_back:].reshape(1, look_back, 1), verbose=0)
#         input_seq = np.append(input_seq, next_pred.flatten())
#     X_test = np.array(X_test)
#     inv_scaled = scaler.inverse_transform(X_test)
#     return inv_scaled.flatten()

# # --- BLSTM Forecast ---
# with st.spinner("Menjalankan model BLSTM..."):
#     blstm_preds = blstm_forecasting(
#         train['sales_diff'], test['sales_diff'],
#         look_back=look_back, units=units,
#         batch_size=batch_size, epochs=epochs
#     )
#     blstm_forecast_boxcox = inverse_diff(blstm_preds, train['total_sales_boxcox'].iloc[-1])
#     blstm_forecast = inv_boxcox(blstm_forecast_boxcox, lam_sales)

# # --- Samakan panjang ---
# min_len = min(len(sarimax_forecast), len(blstm_forecast))
# sarimax_forecast = sarimax_forecast[:min_len]
# blstm_forecast = blstm_forecast[:min_len]
# test = test.iloc[-min_len:]

# # --- Hybrid Dynamic ---
# def hybrid_proximity(sarimax_pred, blstm_pred, actual):
#     return np.array([
#         s if abs(s - a) < abs(b - a) else b
#         for s, b, a in zip(sarimax_pred, blstm_pred, actual)
#     ])

# hybrid_dynamic = hybrid_proximity(sarimax_forecast, blstm_forecast, test['total_sales'].values)

# # --- Plot Forecasts ---
# st.subheader("Hasil Forecast")
# def plot_forecast(pred, title):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=train['date'], y=train['total_sales'], name='Train'))
#     fig.add_trace(go.Scatter(x=test['date'], y=test['total_sales'], name='Test'))
#     fig.add_trace(go.Scatter(x=test['date'], y=pred, name='Forecast'))
#     fig.update_layout(title=title, width=900, height=400)
#     st.plotly_chart(fig)

# plot_forecast(sarimax_forecast, "SARIMAX Forecast")
# plot_forecast(blstm_forecast, "BLSTM Forecast")
# plot_forecast(hybrid_dynamic, "Hybrid Forecast (Proximity)")

# # --- Evaluasi ---
# def evaluate(true, pred, name):
#     mse = mean_squared_error(true, pred)
#     rmse = np.sqrt(mse)
#     mape = mean_absolute_percentage_error(true, pred) * 100
#     return f"**{name}**\n- MSE: {mse:.2f}\n- RMSE: {rmse:.2f}\n- MAPE: {mape:.2f}%"

# st.subheader("Evaluasi Model")
# st.markdown(evaluate(test['total_sales'].values, sarimax_forecast, "SARIMAX"))
# st.markdown(evaluate(test['total_sales'].values, blstm_forecast, "BLSTM"))
# st.markdown(evaluate(test['total_sales'].values, hybrid_dynamic, "Hybrid (Proximity)"))



# ---------------------- V2 ---------------------- #
# Add Hybrid Yang Lain
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from utils import load_saved_dataset

st.title("Forecasting dengan Hybrid SARIMAX + BLSTM")

if "data" not in st.session_state:
    df, source = load_saved_dataset()
    if df is not None:
        st.session_state["data"] = df
        st.session_state["source"] = source
    else:
        if st.session_state.get("role") == "admin":
            st.warning("Silakan upload file terlebih dahulu di halaman admin")
        else:
            st.warning("Admin Belum Mengupload File Dataset!")
        st.stop()

data = st.session_state["data"]

st.sidebar.header("Parameter Filter")
start_date = st.sidebar.text_input("Start Date (YYYY-MM)", "2020-01")
end_date = st.sidebar.text_input("End Date (YYYY-MM)", "2024-12")
brand_filter = st.sidebar.multiselect("Filter Customers", options=data['customers'].unique())

if not brand_filter:
    st.warning("Mohon diisi untuk Customers-nya terlebih dahulu di sidebar.")
    st.stop()

st.sidebar.header("SARIMAX Parameters")
p = st.sidebar.number_input("p (AR)", 0, 5, 0)
d = st.sidebar.number_input("d (I)", 0, 2, 2)
q = st.sidebar.number_input("q (MA)", 0, 5, 2)
P = st.sidebar.number_input("P (Seasonal AR)", 0, 5, 2)
D = st.sidebar.number_input("D (Seasonal I)", 0, 2, 1)
Q = st.sidebar.number_input("Q (Seasonal MA)", 0, 5, 2)
S = st.sidebar.number_input("S (Seasonal Period)", 1, 24, 12)

st.sidebar.header("BLSTM Parameters")
look_back = st.sidebar.slider("Look Back", 1, 60, 24)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1)
epochs = st.sidebar.slider("Epochs", 10, 200, 100)
units = st.sidebar.slider("LSTM Units", 8, 256, 16)

st.sidebar.header("Hybrid Parameters")
hybrid_method = st.sidebar.selectbox(
    "Metode Hybrid",
    ["Proximity", "Average", "Weighted", "Dynamic Weighted"]
)
alpha = st.sidebar.slider("Alpha untuk Weighted", 0.0, 1.0, 0.5)

data['date'] = pd.to_datetime(data['date'], format='%m-%Y')
data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]
if 'All' not in brand_filter:
    data = data[data['customers'].isin(brand_filter)]

df = data.groupby('date').agg({'total_sales': 'sum', 'qty': 'sum'}).reset_index()

st.subheader("Total Sales per Bulan")
st.line_chart(df.set_index('date')['total_sales'])

df['total_sales_boxcox'], lam_sales = boxcox(df['total_sales'])
df['qty_boxcox'], lam_qty = boxcox(df['qty'])
df['sales_diff'] = pd.Series(df['total_sales_boxcox']).diff()
df['qty_diff'] = pd.Series(df['qty_boxcox']).diff()
df.dropna(inplace=True)

with st.expander("Lihat ACF & PACF Plot"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(df['sales_diff'], ax=ax1)
    plot_pacf(df['sales_diff'], ax=ax2, method='ywm')
    st.pyplot(fig)

train = df.iloc[:-int(len(df)*0.2)]
test = df.iloc[-int(len(df)*0.2):]

with st.spinner("Menjalankan model SARIMAX..."):
    sarimax_model = SARIMAX(
        train['total_sales_boxcox'],
        exog=train[['qty_boxcox']],
        order=(p, d, q),
        seasonal_order=(P, D, Q, S),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()

    boxcox_forecast = sarimax_model.forecast(steps=len(test), exog=test[['qty_boxcox']])
    boxcox_forecast[boxcox_forecast <= 0] = 1e-3
    sarimax_forecast = inv_boxcox(boxcox_forecast, lam_sales)

def create_blstm_model(input_shape, units=128):
    model = Sequential()
    model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
    model.add(LSTM(units))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_lstm_data(series, look_back=24):
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i-look_back:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def inverse_diff(preds, last_val):
    return np.r_[last_val, preds].cumsum()[1:]

def blstm_forecasting(train_df, test_df, look_back=24, units=16, batch_size=16, epochs=100):
    full_series = pd.concat([train_df, test_df])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_series.values.reshape(-1, 1)).flatten()
    train_scaled = scaled[:len(train_df)]
    test_scaled = scaled[len(train_df):]

    X_train, y_train = prepare_lstm_data(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = create_blstm_model((look_back, 1), units=units)
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

    X_test = []
    input_seq = scaled[len(train_df)-look_back:len(train_df)]
    for i in range(len(test_df)):
        X_test.append(input_seq[-look_back:])
        next_pred = model.predict(input_seq[-look_back:].reshape(1, look_back, 1), verbose=0)
        input_seq = np.append(input_seq, next_pred.flatten())
    X_test = np.array(X_test)
    inv_scaled = scaler.inverse_transform(X_test)
    return inv_scaled.flatten()

with st.spinner("Menjalankan model BLSTM..."):
    blstm_preds = blstm_forecasting(
        train['sales_diff'], test['sales_diff'],
        look_back=look_back, units=units,
        batch_size=batch_size, epochs=epochs
    )
    blstm_forecast_boxcox = inverse_diff(blstm_preds, train['total_sales_boxcox'].iloc[-1])
    blstm_forecast = inv_boxcox(blstm_forecast_boxcox, lam_sales)

min_len = min(len(sarimax_forecast), len(blstm_forecast))
sarimax_forecast = sarimax_forecast[:min_len]
blstm_forecast = blstm_forecast[:min_len]
test = test.iloc[-min_len:]
actual = test['total_sales'].values

def hybrid_proximity(sarimax_pred, blstm_pred, actual):
    return np.array([
        s if abs(s - a) < abs(b - a) else b
        for s, b, a in zip(sarimax_pred, blstm_pred, actual)
    ])

def hybrid_average(sarimax_pred, blstm_pred):
    return (sarimax_pred + blstm_pred) / 2

def hybrid_weighted(sarimax_pred, blstm_pred, alpha):
    return alpha * sarimax_pred + (1 - alpha) * blstm_pred

def hybrid_dynamic_weighted(sarimax_pred, blstm_pred, actual):
    err_sarimax = np.abs(sarimax_pred - actual)
    err_blstm = np.abs(blstm_pred - actual)
    total_err = err_sarimax + err_blstm
    weights = np.where(total_err == 0, 0.5, err_blstm / total_err)
    return weights * sarimax_pred + (1 - weights) * blstm_pred

if hybrid_method == "Proximity":
    hybrid_forecast = hybrid_proximity(sarimax_forecast, blstm_forecast, actual)
elif hybrid_method == "Average":
    hybrid_forecast = hybrid_average(sarimax_forecast, blstm_forecast)
elif hybrid_method == "Weighted":
    hybrid_forecast = hybrid_weighted(sarimax_forecast, blstm_forecast, alpha)
elif hybrid_method == "Dynamic Weighted":
    hybrid_forecast = hybrid_dynamic_weighted(sarimax_forecast, blstm_forecast, actual)
else:
    hybrid_forecast = hybrid_proximity(sarimax_forecast, blstm_forecast, actual)

st.subheader("Hasil Forecast")
def plot_forecast(pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['date'], y=train['total_sales'], name='Train'))
    fig.add_trace(go.Scatter(x=test['date'], y=test['total_sales'], name='Test'))
    fig.add_trace(go.Scatter(x=test['date'], y=pred, name='Forecast'))
    fig.update_layout(title=title, width=900, height=400)
    st.plotly_chart(fig)

plot_forecast(sarimax_forecast, "SARIMAX Forecast")
plot_forecast(blstm_forecast, "BLSTM Forecast")
plot_forecast(hybrid_forecast, f"Hybrid Forecast ({hybrid_method})")

def evaluate(true, pred, name):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(true, pred) * 100
    return f"**{name}**\n- MSE: {mse:.2f}\n- RMSE: {rmse:.2f}\n- MAPE: {mape:.2f}%"

st.subheader("Evaluasi Model")
st.markdown(evaluate(actual, sarimax_forecast, "SARIMAX"))
st.markdown(evaluate(actual, blstm_forecast, "BLSTM"))
st.markdown(evaluate(actual, hybrid_forecast, f"Hybrid ({hybrid_method})"))
