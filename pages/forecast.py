
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

class ForecastingApp:
    def __init__(self):
        self.initialize_session_state()
        self.setup_ui()
        
    def initialize_session_state(self):
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

    def setup_ui(self):
        st.title("Forecasting dengan Hybrid SARIMAX + BLSTM")
        self.setup_sidebar()
        self.process_data()
        self.run_models()
        self.display_results()
        self.display_future_predictions()  # Tambahan untuk menampilkan prediksi masa depan

    def setup_sidebar(self):
        st.sidebar.header("Data Parameters")
        self.start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        self.end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-01"))
        
        # Convert to monthly periods
        self.start_date = pd.to_datetime(self.start_date).to_period("M").to_timestamp()
        self.end_date = pd.to_datetime(self.end_date).to_period("M").to_timestamp()
        
        if self.start_date >= self.end_date:
            st.error("Start date must be before end date")
            st.stop()

        # Customer and brand filters
        data = st.session_state["data"]
        self.setup_filters(data)
        
        # Model parameters
        st.sidebar.header("SARIMAX Parameters")
        self.sarimax_params = {
            'p': st.sidebar.number_input("p (AR)", 0, 5, 0),
            'd': st.sidebar.number_input("d (I)", 0, 2, 2),
            'q': st.sidebar.number_input("q (MA)", 0, 5, 2),
            'P': st.sidebar.number_input("P (Seasonal AR)", 0, 5, 2),
            'D': st.sidebar.number_input("D (Seasonal I)", 0, 2, 1),
            'Q': st.sidebar.number_input("Q (Seasonal MA)", 0, 5, 2),
            'S': st.sidebar.number_input("S (Seasonal Period)", 1, 24, 12)
        }

        st.sidebar.header("BLSTM Parameters")
        self.blstm_params = {
            'look_back': st.sidebar.slider("Look Back", 1, 60, 24),
            'batch_size': st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1),
            'epochs': st.sidebar.slider("Epochs", 10, 200, 100),
            'units': st.sidebar.slider("LSTM Units", 8, 256, 16)
        }

        st.sidebar.header("Hybrid Parameters")
        self.hybrid_method = st.sidebar.selectbox(
            "Metode Hybrid",
            ["Dynamic Weighted"]
        )
        
        # Hapus checkbox dan buat otomatis
        self.predict_future = True  # Selalu aktifkan prediksi masa depan

    def setup_filters(self, data):
        customer_options = ['All'] + list(data['customers'].unique())
        selected_customers = st.sidebar.multiselect("Filter Customers", options=customer_options, default=['All'])
        
        if 'All' in selected_customers:
            self.selected_customers = list(data['customers'].unique())
        else:
            self.selected_customers = selected_customers

        if not self.selected_customers:
            st.warning("Mohon diisi untuk Customers-nya terlebih dahulu di sidebar.")
            st.stop()

        brand_options = ['All'] + list(data['brand'].unique())
        selected_brands = st.sidebar.multiselect("Filter Brand", options=brand_options, default=['All'])

        if 'All' in selected_brands:
            self.selected_brands = list(data['brand'].unique())
        else:
            self.selected_brands = selected_brands

        if not self.selected_brands:
            st.warning("Mohon pilih minimal satu Brand di sidebar.")
            st.stop()

    def process_data(self):
        data = st.session_state["data"]
        
        # Filter data based on selections
        data = data[
            (data['customers'].isin(self.selected_customers)) & 
            (data['brand'].isin(self.selected_brands)) &
            (data['date'] >= pd.to_datetime(self.start_date)) & 
            (data['date'] <= pd.to_datetime(self.end_date))
        ]
        
        # Aggregate by date
        self.df = data.groupby('date').agg({'total_sales': 'sum', 'qty': 'sum'}).reset_index()
        
        # Validate sufficient data
        if len(self.df) < self.blstm_params['look_back'] * 2:
            st.error(f"Not enough data points. Need at least {self.blstm_params['look_back']*2} points for look_back={self.blstm_params['look_back']}")
            st.stop()
        
        # Display data
        st.subheader("Total Sales per Bulan")
        st.line_chart(self.df.set_index('date')['total_sales'])
        
        # Transform data
        self.prepare_data_for_modeling()

    def prepare_data_for_modeling(self):
        # Box-Cox transformations
        self.df['total_sales_boxcox'], self.lam_sales = boxcox(self.df['total_sales'])
        self.df['qty_boxcox'], self.lam_qty = boxcox(self.df['qty'])
        
        # Differencing
        self.df['sales_diff'] = pd.Series(self.df['total_sales_boxcox']).diff()
        self.df['qty_diff'] = pd.Series(self.df['qty_boxcox']).diff()
        self.df.dropna(inplace=True)
        
        # Show ACF/PACF plots
        with st.expander("Lihat ACF & PACF Plot"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            plot_acf(self.df['sales_diff'], ax=ax1)
            plot_pacf(self.df['sales_diff'], ax=ax2, method='ywm')
            st.pyplot(fig)

        # Split data
        self.train = self.df.iloc[:-int(len(self.df)*0.2)]
        self.test = self.df.iloc[-int(len(self.df)*0.2):]
        self.actual = self.test['total_sales'].values

        # Selalu siapkan data untuk prediksi masa depan
        self.prepare_future_data()

    def prepare_future_data(self):
        last_date = self.df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
        avg_qty = self.df['qty'].mean()
        
        self.future_test = pd.DataFrame({
            'date': future_dates,
            'qty': avg_qty
        })
        
        # Transform qty future using lambda from training
        self.future_test['qty_boxcox'] = boxcox(self.future_test['qty'], self.lam_qty)

    def run_models(self):
        with st.spinner("Menjalankan model SARIMAX..."):
            self.run_sarimax()
            
        with st.spinner("Menjalankan model BLSTM..."):
            self.run_blstm()
            
        self.combine_forecasts()
        
        # Jalankan juga prediksi untuk masa depan
        with st.spinner("Mempersiapkan prediksi masa depan..."):
            self.run_future_predictions()

    def run_sarimax(self):
        model = SARIMAX(
            self.train['total_sales_boxcox'],
            exog=self.train[['qty_boxcox']],
            order=(self.sarimax_params['p'], self.sarimax_params['d'], self.sarimax_params['q']),
            seasonal_order=(self.sarimax_params['P'], self.sarimax_params['D'], 
                          self.sarimax_params['Q'], self.sarimax_params['S']),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()

        # Prediksi untuk test set
        boxcox_forecast = model.forecast(steps=len(self.test), exog=self.test[['qty_boxcox']])
        boxcox_forecast[boxcox_forecast <= 0] = 1e-3
        self.sarimax_forecast = inv_boxcox(boxcox_forecast, self.lam_sales)
        
        # Simpan model untuk prediksi masa depan
        self.sarimax_model = model

    def run_blstm(self):
        blstm_preds = self.blstm_forecasting(
            self.train['sales_diff'], self.test['sales_diff'],
            look_back=self.blstm_params['look_back'],
            units=self.blstm_params['units'],
            batch_size=self.blstm_params['batch_size'],
            epochs=self.blstm_params['epochs']
        )
        
        blstm_forecast_boxcox = self.inverse_diff(blstm_preds, self.train['total_sales_boxcox'].iloc[-1])
        self.blstm_forecast = inv_boxcox(blstm_forecast_boxcox, self.lam_sales)
        
        # Simpan data untuk prediksi masa depan
        self.blstm_train_series = self.train['sales_diff']
        self.last_train_value = self.train['total_sales_boxcox'].iloc[-1]

    def run_future_predictions(self):
        # Prediksi SARIMAX untuk masa depan
        boxcox_future = self.sarimax_model.forecast(steps=12, exog=self.future_test[['qty_boxcox']])
        boxcox_future[boxcox_future <= 0] = 1e-3
        self.sarimax_future = inv_boxcox(boxcox_future, self.lam_sales)
        
        # Prediksi BLSTM untuk masa depan
        future_dummy = pd.Series([0] * 12)  # Dummy series untuk prediksi masa depan
        blstm_future_preds = self.blstm_forecasting(
            self.blstm_train_series, future_dummy,
            look_back=self.blstm_params['look_back'],
            units=self.blstm_params['units'],
            batch_size=self.blstm_params['batch_size'],
            epochs=self.blstm_params['epochs']
        )
        
        blstm_future_boxcox = self.inverse_diff(blstm_future_preds, self.last_train_value)
        self.blstm_future = inv_boxcox(blstm_future_boxcox, self.lam_sales)
        
        # Gabungkan prediksi hybrid untuk masa depan
        self.hybrid_future = self.hybrid_dynamic_weighted(
            self.sarimax_future, 
            self.blstm_future, 
            None  # Tidak ada actual untuk masa depan
        )

    def blstm_forecasting(self, train_df, test_df, look_back=24, units=16, batch_size=16, epochs=100):
        full_series = pd.concat([train_df, test_df])
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(full_series.values.reshape(-1, 1)).flatten()
        
        if len(test_df) == 12 and all(test_df == 0):  # Untuk prediksi masa depan
            # For future predictions
            X_train, y_train = self.prepare_lstm_data(scaled[:len(train_df)], look_back)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            model = self.create_blstm_model((look_back, 1), units=units)
            es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
            
            predictions = []
            input_seq = scaled[-look_back:]
            
            for _ in range(len(test_df)):
                next_pred = model.predict(input_seq[-look_back:].reshape(1, look_back, 1), verbose=0)
                predictions.append(next_pred[0,0])
                input_seq = np.append(input_seq, next_pred.flatten())
                
            inv_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            return inv_scaled.flatten()
        else:
            # For test set predictions
            X_train, y_train = self.prepare_lstm_data(scaled[:len(train_df)], look_back)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            model = self.create_blstm_model((look_back, 1), units=units)
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

    @staticmethod
    def create_blstm_model(input_shape, units=128):
        model = Sequential()
        model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
        model.add(LSTM(units))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

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

    def combine_forecasts(self):
        min_len = min(len(self.sarimax_forecast), len(self.blstm_forecast))
        self.sarimax_forecast = self.sarimax_forecast[:min_len]
        self.blstm_forecast = self.blstm_forecast[:min_len]
        self.test = self.test.iloc[-min_len:]
        self.actual = self.test['total_sales'].values[:min_len]
        
        if self.hybrid_method == "Dynamic Weighted":
            self.hybrid_forecast = self.hybrid_dynamic_weighted(
                self.sarimax_forecast, 
                self.blstm_forecast, 
                self.actual
            )

    def hybrid_dynamic_weighted(self, sarimax_pred, blstm_pred, actual=None):
        if actual is None:
            # For future predictions, use equal weights
            return (sarimax_pred + blstm_pred) / 2
        else:
            err_sarimax = np.abs(sarimax_pred - actual)
            err_blstm = np.abs(blstm_pred - actual)
            total_err = err_sarimax + err_blstm
            weights = np.where(total_err == 0, 0.5, err_blstm / total_err)
            return weights * sarimax_pred + (1 - weights) * blstm_pred

    def display_results(self):
        st.subheader("Hasil Forecast")
        self.plot_forecast(self.sarimax_forecast, "SARIMAX Forecast")
        self.plot_forecast(self.blstm_forecast, "BLSTM Forecast")
        self.plot_forecast(self.hybrid_forecast, f"Hybrid Forecast ({self.hybrid_method})")
        
        self.show_evaluation()

    def display_future_predictions(self):
        st.subheader("Prediksi untuk Tahun yang Akan Datang (12 Bulan Kedepan)")
        
        fig = go.Figure()
        
        # Tambahkan data historis
        fig.add_trace(go.Scatter(
            x=self.df['date'], 
            y=self.df['total_sales'], 
            name='Data Historis',
            line=dict(color='blue')
        ))
        
        # Tambahkan prediksi SARIMAX
        fig.add_trace(go.Scatter(
            x=self.future_test['date'], 
            y=self.sarimax_future, 
            name='SARIMAX Future',
            line=dict(color='red', dash='dash')
        ))
        
        # Tambahkan prediksi BLSTM
        fig.add_trace(go.Scatter(
            x=self.future_test['date'], 
            y=self.blstm_future, 
            name='BLSTM Future',
            line=dict(color='green', dash='dash')
        ))
        
        # Tambahkan prediksi Hybrid
        fig.add_trace(go.Scatter(
            x=self.future_test['date'], 
            y=self.hybrid_future, 
            name=f'Hybrid Future ({self.hybrid_method})',
            line=dict(color='purple', dash='dash')
        ))
        
        fig.update_layout(
            title='Prediksi 12 Bulan Kedepan',
            xaxis_title='Date',
            yaxis_title='Total Sales',
            width=900,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig)

    def plot_forecast(self, pred, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.train['date'], 
            y=self.train['total_sales'], 
            name='Train',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.test['date'], 
            y=self.actual, 
            name='Actual',
            line=dict(color='green')
        ))
            
        fig.add_trace(go.Scatter(
            x=self.test['date'], 
            y=pred, 
            name='Forecast',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Total Sales',
            width=900,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig)

    def show_evaluation(self):
        st.subheader("Evaluasi Model")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(self.evaluate_model(self.actual, self.sarimax_forecast, "SARIMAX"))
        
        with col2:
            st.markdown(self.evaluate_model(self.actual, self.blstm_forecast, "BLSTM"))
        
        with col3:
            st.markdown(self.evaluate_model(
                self.actual, 
                self.hybrid_forecast, 
                f"Hybrid ({self.hybrid_method})"
            ))

    @staticmethod
    def evaluate_model(true, pred, name):
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
    
    # ---------------------- Add Prediksi 12 Bulan Kedepan ---------------------- #

    def prepare_future_data(self):
        """Mempersiapkan data untuk prediksi 12 bulan ke depan"""
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

    def calculate_sarimax_future(self):
        """Menghitung prediksi SARIMAX untuk 12 bulan ke depan"""
        boxcox_future = self.sarimax_model.forecast(
            steps=12,
            exog=self.future_df[['qty_boxcox']]
        )
        boxcox_future[boxcox_future <= 0] = 1e-3
        self.sarimax_future = inv_boxcox(boxcox_future, self.lam_sales)
        return self.sarimax_future

    def calculate_blstm_future(self):
        """Menghitung prediksi BLSTM untuk 12 bulan ke depan"""
        # Scale data training
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(self.train['sales_diff'].values.reshape(-1, 1)).flatten()
        
        # Siapkan data untuk prediksi multi-step
        X_train, y_train = [], []
        for i in range(self.blstm_params['look_back'], len(scaled_train)-12+1):
            X_train.append(scaled_train[i-self.blstm_params['look_back']:i])
            y_train.append(scaled_train[i:i+12])
        
        X_train = np.array(X_train).reshape(-1, self.blstm_params['look_back'], 1)
        y_train = np.array(y_train)
        
        # Bangun model multi-output
        model = Sequential()
        model.add(Bidirectional(
            LSTM(self.blstm_params['units'], return_sequences=True),
            input_shape=(self.blstm_params['look_back'], 1)
        ))
        model.add(Bidirectional(LSTM(self.blstm_params['units'])))
        model.add(Dense(12))
        model.compile(optimizer='adam', loss='mse')
        
        # Latih model
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=self.blstm_params['epochs'],
            batch_size=self.blstm_params['batch_size'],
            verbose=0,
            callbacks=[es]
        )
        
        # Buat prediksi
        last_window = scaled_train[-self.blstm_params['look_back']:]
        future_pred = model.predict(last_window.reshape(1, self.blstm_params['look_back'], 1))[0]
        
        # Kembalikan ke skala asli
        future_pred = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        
        # Transformasi kembali ke nilai asli
        blstm_future_boxcox = self.inverse_diff(future_pred, self.train['total_sales_boxcox'].iloc[-1])
        self.blstm_future = inv_boxcox(blstm_future_boxcox, self.lam_sales)
        return self.blstm_future

    def calculate_hybrid_future(self):
        """Menghitung prediksi hybrid untuk 12 bulan ke depan"""
        self.hybrid_future = self.hybrid_dynamic_weighted(
            self.sarimax_future,
            self.blstm_future,
            None  # Tidak ada actual untuk prediksi masa depan
        )
        return self.hybrid_future

    def run_future_predictions(self):
        """Menjalankan seluruh proses prediksi 12 bulan ke depan"""
        with st.spinner("Mempersiapkan data masa depan..."):
            self.prepare_future_data()
            
        with st.spinner("Menghitung prediksi SARIMAX..."):
            self.calculate_sarimax_future()
            
        with st.spinner("Menghitung prediksi BLSTM..."):
            self.calculate_blstm_future()
            
        with st.spinner("Menghitung prediksi Hybrid..."):
            self.calculate_hybrid_future()

    def display_future_predictions(self):
        """Menampilkan visualisasi prediksi 12 bulan ke depan"""
        st.subheader("Prediksi untuk 12 Bulan Kedepan")
        
        fig = go.Figure()
        
        # Tambahkan data historis (2 tahun terakhir)
        hist_df = self.df[self.df['date'] >= (self.df['date'].max() - pd.DateOffset(years=2))]
        fig.add_trace(go.Scatter(
            x=hist_df['date'], y=hist_df['total_sales'],
            name='Data Historis', line=dict(color='blue')
        ))
        
        # Tambahkan prediksi
        fig.add_trace(go.Scatter(
            x=self.future_df['date'], y=self.sarimax_future,
            name='SARIMAX Future', line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.future_df['date'], y=self.blstm_future,
            name='BLSTM Future', line=dict(color='green', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.future_df['date'], y=self.hybrid_future,
            name=f'Hybrid Future ({self.hybrid_method})', line=dict(color='purple')
        ))
        
        fig.update_layout(
            title='Prediksi 12 Bulan Kedepan',
            xaxis_title='Date',
            yaxis_title='Total Sales',
            width=900,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig)

# Run the app
if __name__ == "__main__":
    app = ForecastingApp()
