
# # ---------------------- V1 ---------------------- #
# # Add Hybrid Yang Lain

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

# available_dates = data['date'].dt.to_period("M").drop_duplicates().astype(str)

# start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
# end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-01"))

# start_date = pd.to_datetime(start_date).to_period("M").to_timestamp()
# end_date = pd.to_datetime(end_date).to_period("M").to_timestamp()

# # add 'All' di awal
# customer_options = ['All'] + list(data['customers'].unique())

# brand_filter = st.sidebar.multiselect("Filter Customers", options=customer_options, default=['All'])

# # Jika 'All' dipilih, isi selected_customers dengan seluruh customer
# if 'All' in brand_filter:
#     selected_customers = list(data['customers'].unique())
# else:
#     selected_customers = brand_filter

# # Validasi jika tidak ada yang dipilih sama sekali
# if not selected_customers:
#     st.warning("Mohon diisi untuk Customers-nya terlebih dahulu di sidebar.")
#     st.stop()

# # Filter data
# data = data[data['customers'].isin(selected_customers)]


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

# st.sidebar.header("Hybrid Parameters")
# hybrid_method = st.sidebar.selectbox(
#     "Metode Hybrid",
#     # ["Proximity", "Average", "Weighted", "Dynamic Weighted"]
#     ["Dynamic Weighted"]
# )
# # alpha = st.sidebar.slider("Alpha untuk Weighted", 0.0, 1.0, 0.5)
# alpha = 1.0

# data['date'] = pd.to_datetime(data['date'], format='%m-%Y')
# data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]
# if 'All' not in brand_filter:
#     data = data[data['customers'].isin(brand_filter)]

# df = data.groupby('date').agg({'total_sales': 'sum', 'qty': 'sum'}).reset_index()

# st.subheader("Total Sales per Bulan")
# st.line_chart(df.set_index('date')['total_sales'])

# df['total_sales_boxcox'], lam_sales = boxcox(df['total_sales'])
# df['qty_boxcox'], lam_qty = boxcox(df['qty'])
# df['sales_diff'] = pd.Series(df['total_sales_boxcox']).diff()
# df['qty_diff'] = pd.Series(df['qty_boxcox']).diff()
# df.dropna(inplace=True)

# with st.expander("Lihat ACF & PACF Plot"):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
#     plot_acf(df['sales_diff'], ax=ax1)
#     plot_pacf(df['sales_diff'], ax=ax2, method='ywm')
#     st.pyplot(fig)

# train = df.iloc[:-int(len(df)*0.2)]
# test = df.iloc[-int(len(df)*0.2):]

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

# with st.spinner("Menjalankan model BLSTM..."):
#     blstm_preds = blstm_forecasting(
#         train['sales_diff'], test['sales_diff'],
#         look_back=look_back, units=units,
#         batch_size=batch_size, epochs=epochs
#     )
#     blstm_forecast_boxcox = inverse_diff(blstm_preds, train['total_sales_boxcox'].iloc[-1])
#     blstm_forecast = inv_boxcox(blstm_forecast_boxcox, lam_sales)

# min_len = min(len(sarimax_forecast), len(blstm_forecast))
# sarimax_forecast = sarimax_forecast[:min_len]
# blstm_forecast = blstm_forecast[:min_len]
# test = test.iloc[-min_len:]
# actual = test['total_sales'].values

# def hybrid_proximity(sarimax_pred, blstm_pred, actual):
#     return np.array([
#         s if abs(s - a) < abs(b - a) else b
#         for s, b, a in zip(sarimax_pred, blstm_pred, actual)
#     ])

# def hybrid_average(sarimax_pred, blstm_pred):
#     return (sarimax_pred + blstm_pred) / 2

# def hybrid_weighted(sarimax_pred, blstm_pred, alpha):
#     return alpha * sarimax_pred + (1 - alpha) * blstm_pred

# def hybrid_dynamic_weighted(sarimax_pred, blstm_pred, actual):
#     err_sarimax = np.abs(sarimax_pred - actual)
#     err_blstm = np.abs(blstm_pred - actual)
#     total_err = err_sarimax + err_blstm
#     weights = np.where(total_err == 0, 0.5, err_blstm / total_err)
#     return weights * sarimax_pred + (1 - weights) * blstm_pred

# if hybrid_method == "Proximity":
#     hybrid_forecast = hybrid_proximity(sarimax_forecast, blstm_forecast, actual)
# elif hybrid_method == "Average":
#     hybrid_forecast = hybrid_average(sarimax_forecast, blstm_forecast)
# elif hybrid_method == "Weighted":
#     hybrid_forecast = hybrid_weighted(sarimax_forecast, blstm_forecast, alpha)
# elif hybrid_method == "Dynamic Weighted":
#     hybrid_forecast = hybrid_dynamic_weighted(sarimax_forecast, blstm_forecast, actual)
# else:
#     hybrid_forecast = hybrid_proximity(sarimax_forecast, blstm_forecast, actual)

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
# plot_forecast(hybrid_forecast, f"Hybrid Forecast ({hybrid_method})")

# def evaluate(true, pred, name):
#     mse = mean_squared_error(true, pred)
#     rmse = np.sqrt(mse)
#     mape = mean_absolute_percentage_error(true, pred) * 100

#     mse_trillion = mse / 1e12
#     rmse_million = rmse / 1e6

#     return (
#         f"**{name}**\n"
#         f"- MSE: {mse_trillion:.2f} \n"
#         f"- RMSE: {rmse_million:.2f} \n"
#         f"- MAPE: {mape:.2f}%"
#     )


# st.subheader("Evaluasi Model")
# st.markdown(evaluate(actual, sarimax_forecast, "SARIMAX"))
# st.markdown(evaluate(actual, blstm_forecast, "BLSTM"))
# st.markdown(evaluate(actual, hybrid_forecast, f"Hybrid ({hybrid_method})"))


# ---------------------- V2 ---------------------- #
# Add Prediksi tahun depan dan filter per brand
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

# class ForecastingApp:
#     def __init__(self):
#         self.initialize_session_state()
#         self.setup_ui()
        
#     def initialize_session_state(self):
#         if "data" not in st.session_state:
#             df, source = load_saved_dataset()
#             if df is not None:
#                 st.session_state["data"] = df
#                 st.session_state["source"] = source
#             else:
#                 if st.session_state.get("role") == "admin":
#                     st.warning("Silakan upload file terlebih dahulu di halaman admin")
#                 else:
#                     st.warning("Admin Belum Mengupload File Dataset!")
#                 st.stop()

#     def setup_ui(self):
#         st.title("Forecasting dengan Hybrid SARIMAX + BLSTM")
#         self.setup_sidebar()
#         self.process_data()
#         self.run_models()
#         self.display_results()
#         self.display_future_predictions()  # Tambahan untuk menampilkan prediksi masa depan

#     def setup_sidebar(self):
#         st.sidebar.header("Data Parameters")
#         self.start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
#         self.end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-01"))
        
#         # Convert to monthly periods
#         self.start_date = pd.to_datetime(self.start_date).to_period("M").to_timestamp()
#         self.end_date = pd.to_datetime(self.end_date).to_period("M").to_timestamp()
        
#         if self.start_date >= self.end_date:
#             st.error("Start date must be before end date")
#             st.stop()

#         # Customer and brand filters
#         data = st.session_state["data"]
#         self.setup_filters(data)
        
#         # Model parameters
#         st.sidebar.header("SARIMAX Parameters")
#         self.sarimax_params = {
#             'p': st.sidebar.number_input("p (AR)", 0, 5, 0),
#             'd': st.sidebar.number_input("d (I)", 0, 2, 2),
#             'q': st.sidebar.number_input("q (MA)", 0, 5, 2),
#             'P': st.sidebar.number_input("P (Seasonal AR)", 0, 5, 2),
#             'D': st.sidebar.number_input("D (Seasonal I)", 0, 2, 1),
#             'Q': st.sidebar.number_input("Q (Seasonal MA)", 0, 5, 2),
#             'S': st.sidebar.number_input("S (Seasonal Period)", 1, 24, 12)
#         }

#         st.sidebar.header("BLSTM Parameters")
#         self.blstm_params = {
#             'look_back': st.sidebar.slider("Look Back", 1, 60, 24),
#             'batch_size': st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1),
#             'epochs': st.sidebar.slider("Epochs", 10, 200, 100),
#             'units': st.sidebar.slider("LSTM Units", 8, 256, 16)
#         }

#         st.sidebar.header("Hybrid Parameters")
#         self.hybrid_method = st.sidebar.selectbox(
#             "Metode Hybrid",
#             ["Dynamic Weighted"]
#         )
        
#         # Hapus checkbox dan buat otomatis
#         self.predict_future = True  # Selalu aktifkan prediksi masa depan

#     def setup_filters(self, data):
#         customer_options = ['All'] + list(data['customers'].unique())
#         selected_customers = st.sidebar.multiselect("Filter Customers", options=customer_options, default=['All'])
        
#         if 'All' in selected_customers:
#             self.selected_customers = list(data['customers'].unique())
#         else:
#             self.selected_customers = selected_customers

#         if not self.selected_customers:
#             st.warning("Mohon diisi untuk Customers-nya terlebih dahulu di sidebar.")
#             st.stop()

#         brand_options = ['All'] + list(data['brand'].unique())
#         selected_brands = st.sidebar.multiselect("Filter Brand", options=brand_options, default=['All'])

#         if 'All' in selected_brands:
#             self.selected_brands = list(data['brand'].unique())
#         else:
#             self.selected_brands = selected_brands

#         if not self.selected_brands:
#             st.warning("Mohon pilih minimal satu Brand di sidebar.")
#             st.stop()

#     def process_data(self):
#         data = st.session_state["data"]
        
#         # Filter data based on selections
#         data = data[
#             (data['customers'].isin(self.selected_customers)) & 
#             (data['brand'].isin(self.selected_brands)) &
#             (data['date'] >= pd.to_datetime(self.start_date)) & 
#             (data['date'] <= pd.to_datetime(self.end_date))
#         ]
        
#         # Aggregate by date
#         self.df = data.groupby('date').agg({'total_sales': 'sum', 'qty': 'sum'}).reset_index()
        
#         # Validate sufficient data
#         if len(self.df) < self.blstm_params['look_back'] * 2:
#             st.error(f"Not enough data points. Need at least {self.blstm_params['look_back']*2} points for look_back={self.blstm_params['look_back']}")
#             st.stop()
        
#         # Display data
#         st.subheader("Total Sales per Bulan")
#         st.line_chart(self.df.set_index('date')['total_sales'])
        
#         # Transform data
#         self.prepare_data_for_modeling()

#     def prepare_data_for_modeling(self):
#         # Box-Cox transformations
#         self.df['total_sales_boxcox'], self.lam_sales = boxcox(self.df['total_sales'])
#         self.df['qty_boxcox'], self.lam_qty = boxcox(self.df['qty'])
        
#         # Differencing
#         self.df['sales_diff'] = pd.Series(self.df['total_sales_boxcox']).diff()
#         self.df['qty_diff'] = pd.Series(self.df['qty_boxcox']).diff()
#         self.df.dropna(inplace=True)
        
#         # Show ACF/PACF plots
#         with st.expander("Lihat ACF & PACF Plot"):
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
#             plot_acf(self.df['sales_diff'], ax=ax1)
#             plot_pacf(self.df['sales_diff'], ax=ax2, method='ywm')
#             st.pyplot(fig)

#         # Split data
#         self.train = self.df.iloc[:-int(len(self.df)*0.2)]
#         self.test = self.df.iloc[-int(len(self.df)*0.2):]
#         self.actual = self.test['total_sales'].values

#         # Selalu siapkan data untuk prediksi masa depan
#         self.prepare_future_data()

#     def prepare_future_data(self):
#         last_date = self.df['date'].max()
#         future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
#         avg_qty = self.df['qty'].mean()
        
#         self.future_test = pd.DataFrame({
#             'date': future_dates,
#             'qty': avg_qty
#         })
        
#         # Transform qty future using lambda from training
#         self.future_test['qty_boxcox'] = boxcox(self.future_test['qty'], self.lam_qty)

#     def run_models(self):
#         with st.spinner("Menjalankan model SARIMAX..."):
#             self.run_sarimax()
            
#         with st.spinner("Menjalankan model BLSTM..."):
#             self.run_blstm()
            
#         self.combine_forecasts()
        
#         # Jalankan juga prediksi untuk masa depan
#         with st.spinner("Mempersiapkan prediksi masa depan..."):
#             self.run_future_predictions()

#     def run_sarimax(self):
#         model = SARIMAX(
#             self.train['total_sales_boxcox'],
#             exog=self.train[['qty_boxcox']],
#             order=(self.sarimax_params['p'], self.sarimax_params['d'], self.sarimax_params['q']),
#             seasonal_order=(self.sarimax_params['P'], self.sarimax_params['D'], 
#                           self.sarimax_params['Q'], self.sarimax_params['S']),
#             enforce_stationarity=False,
#             enforce_invertibility=False
#         ).fit()

#         # Prediksi untuk test set
#         boxcox_forecast = model.forecast(steps=len(self.test), exog=self.test[['qty_boxcox']])
#         boxcox_forecast[boxcox_forecast <= 0] = 1e-3
#         self.sarimax_forecast = inv_boxcox(boxcox_forecast, self.lam_sales)
        
#         # Simpan model untuk prediksi masa depan
#         self.sarimax_model = model

#     def run_blstm(self):
#         blstm_preds = self.blstm_forecasting(
#             self.train['sales_diff'], self.test['sales_diff'],
#             look_back=self.blstm_params['look_back'],
#             units=self.blstm_params['units'],
#             batch_size=self.blstm_params['batch_size'],
#             epochs=self.blstm_params['epochs']
#         )
        
#         blstm_forecast_boxcox = self.inverse_diff(blstm_preds, self.train['total_sales_boxcox'].iloc[-1])
#         self.blstm_forecast = inv_boxcox(blstm_forecast_boxcox, self.lam_sales)
        
#         # Simpan data untuk prediksi masa depan
#         self.blstm_train_series = self.train['sales_diff']
#         self.last_train_value = self.train['total_sales_boxcox'].iloc[-1]

#     def run_future_predictions(self):
#         # Prediksi SARIMAX untuk masa depan
#         boxcox_future = self.sarimax_model.forecast(steps=12, exog=self.future_test[['qty_boxcox']])
#         boxcox_future[boxcox_future <= 0] = 1e-3
#         self.sarimax_future = inv_boxcox(boxcox_future, self.lam_sales)
        
#         # Prediksi BLSTM untuk masa depan
#         future_dummy = pd.Series([0] * 12)  # Dummy series untuk prediksi masa depan
#         blstm_future_preds = self.blstm_forecasting(
#             self.blstm_train_series, future_dummy,
#             look_back=self.blstm_params['look_back'],
#             units=self.blstm_params['units'],
#             batch_size=self.blstm_params['batch_size'],
#             epochs=self.blstm_params['epochs']
#         )
        
#         blstm_future_boxcox = self.inverse_diff(blstm_future_preds, self.last_train_value)
#         self.blstm_future = inv_boxcox(blstm_future_boxcox, self.lam_sales)
        
#         # Gabungkan prediksi hybrid untuk masa depan
#         self.hybrid_future = self.hybrid_dynamic_weighted(
#             self.sarimax_future, 
#             self.blstm_future, 
#             None  # Tidak ada actual untuk masa depan
#         )

#     def blstm_forecasting(self, train_df, test_df, look_back=24, units=16, batch_size=16, epochs=100):
#         full_series = pd.concat([train_df, test_df])
#         scaler = MinMaxScaler()
#         scaled = scaler.fit_transform(full_series.values.reshape(-1, 1)).flatten()
        
#         if len(test_df) == 12 and all(test_df == 0):  # Untuk prediksi masa depan
#             # For future predictions
#             X_train, y_train = self.prepare_lstm_data(scaled[:len(train_df)], look_back)
#             X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
#             model = self.create_blstm_model((look_back, 1), units=units)
#             es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
#             model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
            
#             predictions = []
#             input_seq = scaled[-look_back:]
            
#             for _ in range(len(test_df)):
#                 next_pred = model.predict(input_seq[-look_back:].reshape(1, look_back, 1), verbose=0)
#                 predictions.append(next_pred[0,0])
#                 input_seq = np.append(input_seq, next_pred.flatten())
                
#             inv_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
#             return inv_scaled.flatten()
#         else:
#             # For test set predictions
#             X_train, y_train = self.prepare_lstm_data(scaled[:len(train_df)], look_back)
#             X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
#             model = self.create_blstm_model((look_back, 1), units=units)
#             es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
#             model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
            
#             X_test = []
#             input_seq = scaled[len(train_df)-look_back:len(train_df)]
            
#             for i in range(len(test_df)):
#                 X_test.append(input_seq[-look_back:])
#                 next_pred = model.predict(input_seq[-look_back:].reshape(1, look_back, 1), verbose=0)
#                 input_seq = np.append(input_seq, next_pred.flatten())
                
#             X_test = np.array(X_test)
#             inv_scaled = scaler.inverse_transform(X_test)
#             return inv_scaled.flatten()

#     @staticmethod
#     def create_blstm_model(input_shape, units=128):
#         model = Sequential()
#         model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
#         model.add(LSTM(units))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mse')
#         return model

#     @staticmethod
#     def prepare_lstm_data(series, look_back=24):
#         X, y = [], []
#         for i in range(look_back, len(series)):
#             X.append(series[i-look_back:i])
#             y.append(series[i])
#         return np.array(X), np.array(y)

#     @staticmethod
#     def inverse_diff(preds, last_val):
#         return np.r_[last_val, preds].cumsum()[1:]

#     def combine_forecasts(self):
#         min_len = min(len(self.sarimax_forecast), len(self.blstm_forecast))
#         self.sarimax_forecast = self.sarimax_forecast[:min_len]
#         self.blstm_forecast = self.blstm_forecast[:min_len]
#         self.test = self.test.iloc[-min_len:]
#         self.actual = self.test['total_sales'].values[:min_len]
        
#         if self.hybrid_method == "Dynamic Weighted":
#             self.hybrid_forecast = self.hybrid_dynamic_weighted(
#                 self.sarimax_forecast, 
#                 self.blstm_forecast, 
#                 self.actual
#             )

#     def hybrid_dynamic_weighted(self, sarimax_pred, blstm_pred, actual=None):
#         if actual is None:
#             # For future predictions, use equal weights
#             return (sarimax_pred + blstm_pred) / 2
#         else:
#             err_sarimax = np.abs(sarimax_pred - actual)
#             err_blstm = np.abs(blstm_pred - actual)
#             total_err = err_sarimax + err_blstm
#             weights = np.where(total_err == 0, 0.5, err_blstm / total_err)
#             return weights * sarimax_pred + (1 - weights) * blstm_pred

#     def display_results(self):
#         st.subheader("Hasil Forecast")
#         self.plot_forecast(self.sarimax_forecast, "SARIMAX Forecast")
#         self.plot_forecast(self.blstm_forecast, "BLSTM Forecast")
#         self.plot_forecast(self.hybrid_forecast, f"Hybrid Forecast ({self.hybrid_method})")
        
#         self.show_evaluation()

#     def display_future_predictions(self):
#         st.subheader("Prediksi untuk Tahun yang Akan Datang (12 Bulan Kedepan)")
        
#         fig = go.Figure()
        
#         # Tambahkan data historis
#         fig.add_trace(go.Scatter(
#             x=self.df['date'], 
#             y=self.df['total_sales'], 
#             name='Data Historis',
#             line=dict(color='blue')
#         ))
        
#         # Tambahkan prediksi SARIMAX
#         fig.add_trace(go.Scatter(
#             x=self.future_test['date'], 
#             y=self.sarimax_future, 
#             name='SARIMAX Future',
#             line=dict(color='red', dash='dash')
#         ))
        
#         # Tambahkan prediksi BLSTM
#         fig.add_trace(go.Scatter(
#             x=self.future_test['date'], 
#             y=self.blstm_future, 
#             name='BLSTM Future',
#             line=dict(color='green', dash='dash')
#         ))
        
#         # Tambahkan prediksi Hybrid
#         fig.add_trace(go.Scatter(
#             x=self.future_test['date'], 
#             y=self.hybrid_future, 
#             name=f'Hybrid Future ({self.hybrid_method})',
#             line=dict(color='purple', dash='dash')
#         ))
        
#         fig.update_layout(
#             title='Prediksi 12 Bulan Kedepan',
#             xaxis_title='Date',
#             yaxis_title='Total Sales',
#             width=900,
#             height=500,
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#         )
#         st.plotly_chart(fig)

#     def plot_forecast(self, pred, title):
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=self.train['date'], 
#             y=self.train['total_sales'], 
#             name='Train',
#             line=dict(color='blue')
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=self.test['date'], 
#             y=self.actual, 
#             name='Actual',
#             line=dict(color='green')
#         ))
            
#         fig.add_trace(go.Scatter(
#             x=self.test['date'], 
#             y=pred, 
#             name='Forecast',
#             line=dict(color='red')
#         ))
        
#         fig.update_layout(
#             title=title,
#             xaxis_title='Date',
#             yaxis_title='Total Sales',
#             width=900,
#             height=500,
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#         )
#         st.plotly_chart(fig)

#     def show_evaluation(self):
#         st.subheader("Evaluasi Model")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.markdown(self.evaluate_model(self.actual, self.sarimax_forecast, "SARIMAX"))
        
#         with col2:
#             st.markdown(self.evaluate_model(self.actual, self.blstm_forecast, "BLSTM"))
        
#         with col3:
#             st.markdown(self.evaluate_model(
#                 self.actual, 
#                 self.hybrid_forecast, 
#                 f"Hybrid ({self.hybrid_method})"
#             ))

#     @staticmethod
#     def evaluate_model(true, pred, name):
#         mse = mean_squared_error(true, pred)
#         rmse = np.sqrt(mse)
#         mape = mean_absolute_percentage_error(true, pred) * 100

#         mse_trillion = mse / 1e12
#         rmse_million = rmse / 1e6

#         return (
#             f"**{name}**\n\n"
#             f"- **MSE:** {mse_trillion:.2f}\n"
#             f"- **RMSE:** {rmse_million:.2f}\n"
#             f"- **MAPE:** {mape:.2f}%"
#         )
    
#     # ---------------------- Add Prediksi 12 Bulan Kedepan ---------------------- #

#     def prepare_future_data(self):
#         """Mempersiapkan data untuk prediksi 12 bulan ke depan"""
#         last_date = self.df['date'].max()
#         future_dates = pd.date_range(
#             start=last_date + pd.DateOffset(months=1),
#             periods=12,
#             freq='MS'
#         )
        
#         # Gunakan rata-rata quantity sebagai input eksogen
#         avg_qty = self.df['qty'].mean()
        
#         self.future_df = pd.DataFrame({
#             'date': future_dates,
#             'qty': avg_qty
#         })
        
#         # Transformasi Box-Cox untuk quantity masa depan
#         self.future_df['qty_boxcox'] = boxcox(self.future_df['qty'], self.lam_qty)

#     def calculate_sarimax_future(self):
#         """Menghitung prediksi SARIMAX untuk 12 bulan ke depan"""
#         boxcox_future = self.sarimax_model.forecast(
#             steps=12,
#             exog=self.future_df[['qty_boxcox']]
#         )
#         boxcox_future[boxcox_future <= 0] = 1e-3
#         self.sarimax_future = inv_boxcox(boxcox_future, self.lam_sales)
#         return self.sarimax_future

#     def calculate_blstm_future(self):
#         """Menghitung prediksi BLSTM untuk 12 bulan ke depan"""
#         # Scale data training
#         scaler = MinMaxScaler()
#         scaled_train = scaler.fit_transform(self.train['sales_diff'].values.reshape(-1, 1)).flatten()
        
#         # Siapkan data untuk prediksi multi-step
#         X_train, y_train = [], []
#         for i in range(self.blstm_params['look_back'], len(scaled_train)-12+1):
#             X_train.append(scaled_train[i-self.blstm_params['look_back']:i])
#             y_train.append(scaled_train[i:i+12])
        
#         X_train = np.array(X_train).reshape(-1, self.blstm_params['look_back'], 1)
#         y_train = np.array(y_train)
        
#         # Bangun model multi-output
#         model = Sequential()
#         model.add(Bidirectional(
#             LSTM(self.blstm_params['units'], return_sequences=True),
#             input_shape=(self.blstm_params['look_back'], 1)
#         ))
#         model.add(Bidirectional(LSTM(self.blstm_params['units'])))
#         model.add(Dense(12))
#         model.compile(optimizer='adam', loss='mse')
        
#         # Latih model
#         es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
#         model.fit(
#             X_train, y_train,
#             epochs=self.blstm_params['epochs'],
#             batch_size=self.blstm_params['batch_size'],
#             verbose=0,
#             callbacks=[es]
#         )
        
#         # Buat prediksi
#         last_window = scaled_train[-self.blstm_params['look_back']:]
#         future_pred = model.predict(last_window.reshape(1, self.blstm_params['look_back'], 1))[0]
        
#         # Kembalikan ke skala asli
#         future_pred = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        
#         # Transformasi kembali ke nilai asli
#         blstm_future_boxcox = self.inverse_diff(future_pred, self.train['total_sales_boxcox'].iloc[-1])
#         self.blstm_future = inv_boxcox(blstm_future_boxcox, self.lam_sales)
#         return self.blstm_future

#     def calculate_hybrid_future(self):
#         """Menghitung prediksi hybrid untuk 12 bulan ke depan"""
#         self.hybrid_future = self.hybrid_dynamic_weighted(
#             self.sarimax_future,
#             self.blstm_future,
#             None  # Tidak ada actual untuk prediksi masa depan
#         )
#         return self.hybrid_future

#     def run_future_predictions(self):
#         """Menjalankan seluruh proses prediksi 12 bulan ke depan"""
#         with st.spinner("Mempersiapkan data masa depan..."):
#             self.prepare_future_data()
            
#         with st.spinner("Menghitung prediksi SARIMAX..."):
#             self.calculate_sarimax_future()
            
#         with st.spinner("Menghitung prediksi BLSTM..."):
#             self.calculate_blstm_future()
            
#         with st.spinner("Menghitung prediksi Hybrid..."):
#             self.calculate_hybrid_future()

#     def display_future_predictions(self):
#         """Menampilkan visualisasi prediksi 12 bulan ke depan"""
#         st.subheader("Prediksi untuk 12 Bulan Kedepan")
        
#         fig = go.Figure()
        
#         # Tambahkan data historis (2 tahun terakhir)
#         hist_df = self.df[self.df['date'] >= (self.df['date'].max() - pd.DateOffset(years=2))]
#         fig.add_trace(go.Scatter(
#             x=hist_df['date'], y=hist_df['total_sales'],
#             name='Data Historis', line=dict(color='blue')
#         ))
        
#         # Tambahkan prediksi
#         fig.add_trace(go.Scatter(
#             x=self.future_df['date'], y=self.sarimax_future,
#             name='SARIMAX Future', line=dict(color='red', dash='dash')
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=self.future_df['date'], y=self.blstm_future,
#             name='BLSTM Future', line=dict(color='green', dash='dash')
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=self.future_df['date'], y=self.hybrid_future,
#             name=f'Hybrid Future ({self.hybrid_method})', line=dict(color='purple')
#         ))
        
#         fig.update_layout(
#             title='Prediksi 12 Bulan Kedepan',
#             xaxis_title='Date',
#             yaxis_title='Total Sales',
#             width=900,
#             height=500,
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#         )
#         st.plotly_chart(fig)

# # Run the app
# if __name__ == "__main__":
#     app = ForecastingApp()


# ---------------------- V3 ---------------------- #
# Add Prediksi dari awal tahun - Revisi After Sidank
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

class DataHandler:
    def __init__(self):
        self.load_data()
        
    def load_data(self):
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
    
    def get_filtered_data(self, params):
        data = st.session_state["data"]
        
        # Filter dates
        filtered = data[
            (data['date'] >= pd.to_datetime(params['start_date'])) & 
            (data['date'] <= pd.to_datetime(params['end_date']))
        ]
        
        # Filter customers and brands
        if params['customers'] != ['All']:
            filtered = filtered[filtered['customers'].isin(params['customers'])]
        if params['brands'] != ['All']:
            filtered = filtered[filtered['brand'].isin(params['brands'])]
            
        return filtered

class ModelBuilder:
    @staticmethod
    def create_blstm_model(input_shape, units=128):
        model = Sequential()
        model.add(Bidirectional(
            LSTM(units, return_sequences=True), 
            input_shape=input_shape
        ))
        model.add(LSTM(units))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

class ForecastingModels:
    def __init__(self, data, params):
        self.df = data
        self.params = params
        self.prepare_data()
        
    def prepare_data(self):
        # Transformasi data
        self.df['total_sales_boxcox'], self.lam_sales = boxcox(self.df['total_sales'])
        self.df['qty_boxcox'], self.lam_qty = boxcox(self.df['qty'])
        
        # Differencing
        self.df['sales_diff'] = self.df['total_sales_boxcox'].diff()
        self.df['qty_diff'] = self.df['qty_boxcox'].diff()
        self.df.dropna(inplace=True)
        
        # Split data
        split_index = -int(len(self.df) * 0.2)
        self.train = self.df.iloc[:split_index]
        self.test = self.df.iloc[split_index:]
        self.actual = self.test['total_sales'].values
        
        # Siapkan data masa depan
        self.prepare_future_data()
    
    def prepare_future_data(self):
        last_date = self.df['date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=12, 
            freq='MS'
        )
        
        # Gunakan rata-rata quantity sebagai input eksogen
        avg_qty = self.df['qty'].mean()
        
        self.future_df = pd.DataFrame({
            'date': future_dates,
            'qty': avg_qty
        })
        
        # Transformasi Box-Cox untuk quantity masa depan
        self.future_df['qty_boxcox'] = boxcox(self.future_df['qty'], self.lam_qty)
    
    def run_sarimax(self):
        model = SARIMAX(
            self.train['total_sales_boxcox'],
            exog=self.train[['qty_boxcox']],
            order=(
                self.params['sarimax']['p'], 
                self.params['sarimax']['d'], 
                self.params['sarimax']['q']
            ),
            seasonal_order=(
                self.params['sarimax']['P'], 
                self.params['sarimax']['D'], 
                self.params['sarimax']['Q'], 
                self.params['sarimax']['S']
            ),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        
        # Prediksi untuk test set
        boxcox_forecast = model.forecast(
            steps=len(self.test), 
            exog=self.test[['qty_boxcox']]
        )
        boxcox_forecast[boxcox_forecast <= 0] = 1e-3
        self.sarimax_forecast = inv_boxcox(boxcox_forecast, self.lam_sales)
        
        # Simpan model untuk prediksi masa depan
        self.sarimax_model = model
        
        # Simpan prediksi in-sample untuk training set
        train_fitted_boxcox = model.fittedvalues
        train_fitted = inv_boxcox(train_fitted_boxcox, self.lam_sales)
        self.sarimax_train_pred = train_fitted
    
    def run_blstm(self):
        # Persiapkan data
        full_series = pd.concat([self.train['sales_diff'], self.test['sales_diff']])
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(full_series.values.reshape(-1, 1)).flatten()
        train_scaled = scaled[:len(self.train)]
        test_scaled = scaled[len(self.train):]
        
        # Buat dataset
        X_train, y_train = self.prepare_lstm_data(train_scaled, self.params['blstm']['look_back'])
        X_train = X_train.reshape((X_train.shape[0], self.params['blstm']['look_back'], 1))
        
        # Bangun dan latih model
        model = ModelBuilder.create_blstm_model(
            (self.params['blstm']['look_back'], 1), 
            units=self.params['blstm']['units']
        )
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=self.params['blstm']['epochs'],
            batch_size=self.params['blstm']['batch_size'],
            verbose=0,
            callbacks=[es]
        )
        
        # Prediksi untuk test set
        X_test = []
        input_seq = scaled[len(self.train)-self.params['blstm']['look_back']:len(self.train)]
        
        for _ in range(len(self.test)):
            X_test.append(input_seq[-self.params['blstm']['look_back']:])
            next_pred = model.predict(
                input_seq[-self.params['blstm']['look_back']:].reshape(1, self.params['blstm']['look_back'], 1), 
                verbose=0
            )
            input_seq = np.append(input_seq, next_pred.flatten())
        
        X_test = np.array(X_test)
        inv_scaled = scaler.inverse_transform(X_test)
        blstm_preds = inv_scaled.flatten()
        
        # Transformasi kembali
        blstm_forecast_boxcox = self.inverse_diff(
            blstm_preds, 
            self.train['total_sales_boxcox'].iloc[-1]
        )
        self.blstm_forecast = inv_boxcox(blstm_forecast_boxcox, self.lam_sales)
        
        # Hitung prediksi in-sample untuk training set
        train_pred_scaled = model.predict(X_train, verbose=0).flatten()
        train_pred = scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
        base = self.train['total_sales_boxcox'].iloc[self.params['blstm']['look_back']-1]
        train_pred_boxcox = self.inverse_diff(train_pred, base)
        self.blstm_train_pred = inv_boxcox(train_pred_boxcox, self.lam_sales)
    
    def run_hybrid(self):
        min_len = min(len(self.sarimax_forecast), len(self.blstm_forecast))
        sarimax_forecast = self.sarimax_forecast[:min_len]
        blstm_forecast = self.blstm_forecast[:min_len]
        actual = self.actual[:min_len]
        
        if self.params['hybrid_method'] == "Dynamic Weighted":
            err_sarimax = np.abs(sarimax_forecast - actual)
            err_blstm = np.abs(blstm_forecast - actual)
            total_err = err_sarimax + err_blstm
            weights = np.where(total_err == 0, 0.5, err_blstm / total_err)
            self.hybrid_forecast = weights * sarimax_forecast + (1 - weights) * blstm_forecast
        else:
            self.hybrid_forecast = (sarimax_forecast + blstm_forecast) / 2
    
    def run_future_predictions(self):
        # Prediksi SARIMAX untuk masa depan
        boxcox_future = self.sarimax_model.forecast(
            steps=12, 
            exog=self.future_df[['qty_boxcox']]
        )
        boxcox_future[boxcox_future <= 0] = 1e-3
        self.sarimax_future = inv_boxcox(boxcox_future, self.lam_sales)
        
        # Prediksi BLSTM untuk masa depan
        # Scale data training
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(
            self.train['sales_diff'].values.reshape(-1, 1)
        ).flatten()
        
        # Siapkan data untuk prediksi multi-step
        look_back = self.params['blstm']['look_back']
        X_train, y_train = [], []
        for i in range(look_back, len(scaled_train)-12+1):
            X_train.append(scaled_train[i-look_back:i])
            y_train.append(scaled_train[i:i+12])
        
        if len(X_train) == 0:
            # Jika tidak cukup data, gunakan pendekatan recursive
            self.blstm_future = self.run_recursive_blstm_future(scaler, scaled_train, look_back)
        else:
            X_train = np.array(X_train).reshape(-1, look_back, 1)
            y_train = np.array(y_train)
            
            # Bangun model multi-output
            model = Sequential()
            model.add(Bidirectional(
                LSTM(self.params['blstm']['units'], return_sequences=True),
                input_shape=(look_back, 1)
            ))
            model.add(Bidirectional(LSTM(self.params['blstm']['units'])))
            model.add(Dense(12))
            model.compile(optimizer='adam', loss='mse')
            
            # Latih model
            es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(
                X_train, y_train,
                epochs=self.params['blstm']['epochs'],
                batch_size=self.params['blstm']['batch_size'],
                verbose=0,
                callbacks=[es]
            )
            
            # Buat prediksi
            last_window = scaled_train[-look_back:]
            future_pred = model.predict(last_window.reshape(1, look_back, 1))[0]
            
            # Kembalikan ke skala asli
            future_pred = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
            
            # Transformasi kembali ke nilai asli
            blstm_future_boxcox = self.inverse_diff(
                future_pred, 
                self.train['total_sales_boxcox'].iloc[-1]
            )
            self.blstm_future = inv_boxcox(blstm_future_boxcox, self.lam_sales)
        
        # Gabungkan prediksi hybrid untuk masa depan
        self.hybrid_future = (self.sarimax_future + self.blstm_future) / 2
    
    def run_recursive_blstm_future(self, scaler, scaled_train, look_back):
        # Jika tidak cukup data untuk multi-output, gunakan recursive
        model = ModelBuilder.create_blstm_model(
            (look_back, 1), 
            units=self.params['blstm']['units']
        )
        
        # Latih model dengan data yang ada (hanya untuk satu step)
        X_train, y_train = self.prepare_lstm_data(scaled_train, look_back)
        X_train = X_train.reshape((X_train.shape[0], look_back, 1))
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=self.params['blstm']['epochs'],
            batch_size=self.params['blstm']['batch_size'],
            verbose=0,
            callbacks=[es]
        )
        
        # Prediksi recursive untuk 12 bulan
        predictions = []
        input_seq = scaled_train[-look_back:]
        
        for _ in range(12):
            next_pred = model.predict(input_seq.reshape(1, look_back, 1), verbose=0)
            predictions.append(next_pred[0,0])
            input_seq = np.append(input_seq[1:], next_pred.flatten())
        
        # Inverse scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Inverse transform: diff and boxcox
        future_boxcox = self.inverse_diff(predictions, self.train['total_sales_boxcox'].iloc[-1])
        return inv_boxcox(future_boxcox, self.lam_sales)
    
    @staticmethod
    def prepare_lstm_data(series, look_back=24):
        X, y = [], []
        for i in range(look_back, len(series)):
            X.append(series[i-look_back:i])
            y.append(series[i])
        return np.array(X), np.array(y)
    
    @staticmethod
    def inverse_diff(preds, last_val):
        return np.r_[last_val, preds].cumsum()[1:]
    
    def get_full_predictions(self):
        "Mengembalikan DataFrame dengan semua prediksi dari awal hingga akhir"
        # Gabungkan semua tanggal
        train_dates = self.train['date']
        test_dates = self.test['date']
        future_dates = self.future_df['date']
        
        all_dates = pd.concat([train_dates, test_dates, future_dates]).reset_index(drop=True)
        
        # Buat DataFrame untuk hasil
        results = pd.DataFrame({'date': all_dates})
        
        # SARIMAX
        sarimax_train = np.full(len(train_dates), np.nan)
        sarimax_train[self.params['sarimax']['p']:] = self.sarimax_train_pred  # Adjust for AR order
        
        sarimax_test = self.sarimax_forecast
        sarimax_future = self.sarimax_future
        
        results['sarimax'] = np.concatenate([
            sarimax_train[:len(train_dates)], 
            sarimax_test[:len(test_dates)],
            sarimax_future[:len(future_dates)]
        ])
        
        # BLSTM
        look_back = self.params['blstm']['look_back']
        blstm_train = np.full(len(train_dates), np.nan)
        blstm_train[look_back:] = self.blstm_train_pred[:len(train_dates)-look_back]
        
        blstm_test = self.blstm_forecast[:len(test_dates)]
        blstm_future = self.blstm_future[:len(future_dates)]
        
        results['blstm'] = np.concatenate([
            blstm_train,
            blstm_test,
            blstm_future
        ])
        
        # Hybrid
        hybrid_train = np.full(len(train_dates), np.nan)
        hybrid_test = self.hybrid_forecast[:len(test_dates)]
        hybrid_future = self.hybrid_future[:len(future_dates)]
        
        results['hybrid'] = np.concatenate([
            hybrid_train,
            hybrid_test,
            hybrid_future
        ])
        
        # Actual values
        actual_train = self.train['total_sales'].values[:len(train_dates)]
        actual_test = self.test['total_sales'].values[:len(test_dates)]
        actual_future = np.full(len(future_dates), np.nan)
        
        results['actual'] = np.concatenate([
            actual_train,
            actual_test,
            actual_future
        ])
    
        return results

class Visualizer:
    @staticmethod
    def plot_forecast(train_dates, train_values, test_dates, test_values, pred_values, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_dates, y=train_values, 
            name='Train', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=test_values, 
            name='Actual', line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=pred_values, 
            name='Forecast', line=dict(color='red')
        ))
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Total Sales',
            width=900,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    
    @staticmethod
    def plot_full_comparison(dates, actual, pred, title, color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=actual,
            name='Actual', line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=pred,
            name=title, line=dict(color=color, dash='dash', width=2)
        ))
        fig.update_layout(
            title=f'{title} vs Actual',
            xaxis_title='Tanggal',
            yaxis_title='Total Penjualan',
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified',
            width=900,
            height=500
        )
        return fig
    
    @staticmethod
    def plot_future_predictions(hist_dates, hist_values, future_dates, sarimax_future, blstm_future, hybrid_future, hybrid_method):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_values, 
            name='Data Historis', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=sarimax_future, 
            name='SARIMAX Future', line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=blstm_future, 
            name='BLSTM Future', line=dict(color='green', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=hybrid_future, 
            name=f'Hybrid Future ({hybrid_method})', line=dict(color='purple', dash='dash')
        ))
        fig.update_layout(
            title='Prediksi 12 Bulan Kedepan',
            xaxis_title='Date',
            yaxis_title='Total Sales',
            width=900,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    
    @staticmethod
    def evaluate_model(true, pred, name):
        # Pastikan prediksi dan true memiliki panjang yang sama
        min_len = min(len(true), len(pred))
        true = true[:min_len]
        pred = pred[:min_len]
        
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(true, pred) * 100

        mse_trillion = mse / 1e12
        rmse_million = rmse / 1e6

        return (
            f"**{name}**\n\n"
            f"- **MSE:** {mse_trillion:.2f}\n"
            f"- **RMSE:** {rmse_million:.2f}\n"
            f"- **MAPE:** {mape:.2f}%"
        )
    
    @staticmethod
    def plot_complete_timeline(df, title="Perbandingan Prediksi vs Aktual"):
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['actual'],
            name='Aktual', line=dict(color='blue', width=2)
        ))
        
        # SARIMAX
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['sarimax'],
            name='SARIMAX', line=dict(color='red', dash='dash')
        ))
        
        # BLSTM
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['blstm'],
            name='BLSTM', line=dict(color='green', dash='dash')
        ))
        
        # Hybrid
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['hybrid'],
            name='Hybrid', line=dict(color='purple', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Tanggal',
            yaxis_title='Total Penjualan',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            width=900,
            height=600
        )
        return fig

class ForecastingApp:
    def __init__(self):
        self.data_handler = DataHandler()
        self.params = self.setup_sidebar()
        self.process_data()
        self.run_models()
        self.display_results()
        
    def setup_sidebar(self):
        st.sidebar.header("Data Parameters")
        
        # Date inputs
        start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-01"))
        
        # Convert to monthly periods
        start_date = pd.to_datetime(start_date).to_period("M").to_timestamp()
        end_date = pd.to_datetime(end_date).to_period("M").to_timestamp()
        
        if start_date >= end_date:
            st.error("Start date must be before end date")
            st.stop()
        
        # Customer and brand filters
        data = st.session_state["data"]
        customer_options = ['All'] + list(data['customers'].unique())
        selected_customers = st.sidebar.multiselect(
            "Filter Customers", 
            options=customer_options, 
            default=['All']
        )
        
        brand_options = ['All'] + list(data['brand'].unique())
        selected_brands = st.sidebar.multiselect(
            "Filter Brand", 
            options=brand_options, 
            default=['All']
        )
        
        # Model parameters
        st.sidebar.header("SARIMAX Parameters")
        sarimax_params = {
            'p': st.sidebar.number_input("p (AR)", 0, 5, 0),
            'd': st.sidebar.number_input("d (I)", 0, 2, 2),
            'q': st.sidebar.number_input("q (MA)", 0, 5, 2),
            'P': st.sidebar.number_input("P (Seasonal AR)", 0, 5, 2),
            'D': st.sidebar.number_input("D (Seasonal I)", 0, 2, 1),
            'Q': st.sidebar.number_input("Q (Seasonal MA)", 0, 5, 2),
            'S': st.sidebar.number_input("S (Seasonal Period)", 1, 24, 12)
        }

        st.sidebar.header("BLSTM Parameters")
        blstm_params = {
            'look_back': st.sidebar.slider("Look Back", 1, 60, 24),
            'batch_size': st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1),
            'epochs': st.sidebar.slider("Epochs", 10, 200, 100),
            'units': st.sidebar.slider("LSTM Units", 8, 256, 16)
        }

        st.sidebar.header("Hybrid Parameters")
        hybrid_method = st.sidebar.selectbox(
            "Metode Hybrid",
            ["Dynamic Weighted"]
        )
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'customers': selected_customers,
            'brands': selected_brands,
            'sarimax': sarimax_params,
            'blstm': blstm_params,
            'hybrid_method': hybrid_method
        }
    
    def process_data(self):
        # Dapatkan data yang sudah difilter
        raw_data = self.data_handler.get_filtered_data(self.params)
        
        # Aggregate by date
        self.df = raw_data.groupby('date').agg({'total_sales': 'sum', 'qty': 'sum'}).reset_index()
        
        # Validasi data yang cukup
        if len(self.df) < self.params['blstm']['look_back'] * 2:
            st.error(f"Not enough data points. Need at least {self.params['blstm']['look_back']*2} points")
            st.stop()
        
        # Tampilkan data
        st.subheader("Total Sales per Bulan")
        st.line_chart(self.df.set_index('date')['total_sales'])
        
        # Tampilkan ACF/PACF plots
        with st.expander("Lihat ACF & PACF Plot"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            plot_acf(self.df['total_sales'], ax=ax1)
            plot_pacf(self.df['total_sales'], ax=ax2, method='ywm')
            st.pyplot(fig)
    
    def run_models(self):
        # Inisialisasi model
        self.forecaster = ForecastingModels(self.df, self.params)
        
        with st.spinner("Menjalankan model SARIMAX..."):
            self.forecaster.run_sarimax()
            
        with st.spinner("Menjalankan model BLSTM..."):
            self.forecaster.run_blstm()
            
        with st.spinner("Menggabungkan prediksi..."):
            self.forecaster.run_hybrid()
            
        with st.spinner("Mempersiapkan prediksi masa depan..."):
            self.forecaster.run_future_predictions()
            
        # Dapatkan semua prediksi dalam satu DataFrame
        self.full_predictions = self.forecaster.get_full_predictions()
    
    def display_results(self):
        self.display_forecast_results()
        self.display_evaluation()
        self.display_complete_timeline()
        self.display_individual_comparisons()
        self.display_future_predictions()
    
    def display_forecast_results(self):
        st.subheader("Hasil Forecast untuk Data Uji")
        
        # SARIMAX
        sarimax_fig = Visualizer.plot_forecast(
            self.forecaster.train['date'],
            self.forecaster.train['total_sales'],
            self.forecaster.test['date'],
            self.forecaster.actual,
            self.forecaster.sarimax_forecast,
            "SARIMAX Forecast"
        )
        st.plotly_chart(sarimax_fig)
        
        # BLSTM
        blstm_fig = Visualizer.plot_forecast(
            self.forecaster.train['date'],
            self.forecaster.train['total_sales'],
            self.forecaster.test['date'],
            self.forecaster.actual,
            self.forecaster.blstm_forecast,
            "BLSTM Forecast"
        )
        st.plotly_chart(blstm_fig)
        
        # Hybrid
        hybrid_fig = Visualizer.plot_forecast(
            self.forecaster.train['date'],
            self.forecaster.train['total_sales'],
            self.forecaster.test['date'],
            self.forecaster.actual,
            self.forecaster.hybrid_forecast,
            f"Hybrid Forecast ({self.params['hybrid_method']})"
        )
        st.plotly_chart(hybrid_fig)
    
    def display_evaluation(self):
        st.subheader("Evaluasi Model untuk Data Uji")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(Visualizer.evaluate_model(
                self.forecaster.actual,
                self.forecaster.sarimax_forecast[:len(self.forecaster.actual)],
                "SARIMAX"
            ))
        
        with col2:
            st.markdown(Visualizer.evaluate_model(
                self.forecaster.actual,
                self.forecaster.blstm_forecast[:len(self.forecaster.actual)],
                "BLSTM"
            ))
        
        with col3:
            st.markdown(Visualizer.evaluate_model(
                self.forecaster.actual,
                self.forecaster.hybrid_forecast[:len(self.forecaster.actual)],
                f"Hybrid ({self.params['hybrid_method']})"
            ))
    
    def display_complete_timeline(self):
        st.subheader("Perbandingan Lengkap: Aktual vs Semua Prediksi")
        fig = Visualizer.plot_complete_timeline(
            self.full_predictions,
            "Perbandingan Prediksi vs Aktual (2020-2024)"
        )
        st.plotly_chart(fig)
    
    def display_individual_comparisons(self):
        st.subheader("Perbandingan Individual Prediksi vs Aktual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # SARIMAX vs Actual
            sarimax_fig = Visualizer.plot_full_comparison(
                self.full_predictions['date'],
                self.full_predictions['actual'],
                self.full_predictions['sarimax'],
                "SARIMAX",
                "red"
            )
            st.plotly_chart(sarimax_fig)
            
        with col2:
            # BLSTM vs Actual
            blstm_fig = Visualizer.plot_full_comparison(
                self.full_predictions['date'],
                self.full_predictions['actual'],
                self.full_predictions['blstm'],
                "BLSTM",
                "green"
            )
            st.plotly_chart(blstm_fig)
        
        # Hybrid vs Actual
        hybrid_fig = Visualizer.plot_full_comparison(
            self.full_predictions['date'],
            self.full_predictions['actual'],
            self.full_predictions['hybrid'],
            f"Hybrid ({self.params['hybrid_method']})",
            "purple"
        )
        st.plotly_chart(hybrid_fig)
    
    def display_future_predictions(self):
        st.subheader("Prediksi untuk 12 Bulan Kedepan")
        
        # Ambil data 2 tahun terakhir untuk konteks historis
        hist_df = self.df[self.df['date'] >= (self.df['date'].max() - pd.DateOffset(years=2))]
        
        fig = Visualizer.plot_future_predictions(
            hist_df['date'],
            hist_df['total_sales'],
            self.forecaster.future_df['date'],
            self.forecaster.sarimax_future,
            self.forecaster.blstm_future,
            self.forecaster.hybrid_future,
            self.params['hybrid_method']
        )
        st.plotly_chart(fig)

# Run the app
if __name__ == "__main__":
    app = ForecastingApp()