import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
import joblib
import os

class MLEngine:
    def __init__(self, db_engine):
        self.db_engine = db_engine
        self.models_path = "models/saved/"
        os.makedirs(self.models_path, exist_ok=True)

    def load_data(self):
        print("[INFO] Loading data from SQL...")
        query = "SELECT * FROM master_orders_view"
        df = pd.read_sql(query, self.db_engine)
        
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        return df

    def train_forecaster(self): 
        # XGBoost to predict Daily Sales Volume.
        
        print("[INFO] Starting Sales Forecast Training...")
        df = self.load_data()

        daily_sales = df.set_index('order_purchase_timestamp').resample('D').size().reset_index(name='order_count')
        
        daily_sales['order_purchase_timestamp'] = pd.to_datetime(daily_sales['order_purchase_timestamp'])

        daily_sales['lag_1'] = daily_sales['order_count'].shift(1) # Sales yesterday
        daily_sales['lag_7'] = daily_sales['order_count'].shift(7) # Sales last week
        
        daily_sales['day_of_week'] = daily_sales['order_purchase_timestamp'].dt.dayofweek # type: ignore
        daily_sales['month'] = daily_sales['order_purchase_timestamp'].dt.month # type: ignore
        
        daily_sales = daily_sales.dropna()

        # Train/Test Split
        X = daily_sales[['lag_1', 'lag_7', 'day_of_week', 'month']]
        y = daily_sales['order_count']
        
        # last 30 days for testing
        split_idx = len(daily_sales) - 30
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)


        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"[SUCCESS] Forecasting Model Trained. MAE: {mae:.2f}")


        joblib.dump(model, f"{self.models_path}forecast_xgb.pkl")
        return model, mae

    def train_anomaly_detector(self):
        
        # Train Isolation Forest to detect anomalous orders.

        print("[INFO] Starting Anomaly Detection Training...")
        df = self.load_data()
        
        features = ['price', 'freight_value', 'delivery_delay_days', 'review_score'] # Features- Anomaly Detection
        X = df[features].fillna(0) #fill NaNs with 0 to prevent errors


        model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42) #assuming ~3% of data is anomalous
        model.fit(X)

        print("[SUCCESS] Anomaly Detector Trained.")
        joblib.dump(model, f"{self.models_path}anomaly_isoforest.pkl")
        return model