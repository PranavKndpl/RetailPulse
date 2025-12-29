# MODEL CONTRACT v1.0
# Do not modify features, targets, or metrics without bumping version.

import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


class MLEngine:
    def __init__(self):
        pass

    def _prepare_data(self, df, date_col, target_col, context):
        data = df[[date_col, target_col]].copy()
        data.columns = ['ds', 'y']
        data['ds'] = pd.to_datetime(data['ds'])
        data = data.sort_values('ds').dropna()

        # Finance 
        if context == "Financial":
            data['y'] = np.log(data['y'] / data['y'].shift(1))

            data = data.replace([np.inf, -np.inf], np.nan)

            data['y'] = data['y'].clip(lower=-0.2, upper=0.2)

        return data.dropna()

    def _features(self, df, context, dropna=True):
        df = df.copy()

        df['lag_1'] = df['y'].shift(1)
        df['lag_7'] = df['y'].shift(7)
        df['roll_7'] = df['y'].shift(1).rolling(7).mean()

        if context == "Retail":
            df['dow'] = df['ds'].dt.dayofweek
            df['is_weekend'] = (df['dow'] >= 5).astype(int)

        if context == "Financial":
            df['vol_7'] = df['y'].shift(1).rolling(7).std()

        features = [c for c in df.columns if c not in ['ds', 'y']]
        if dropna:
            return df.dropna(), features
        else:
            return df, features

    def _direction_accuracy(self, y_true, y_pred):
        return (
            np.sign(y_true.diff().iloc[1:]) ==
            np.sign(y_pred.diff().iloc[1:])
        ).mean() * 100


    def _xgboost(self, data, context, mode):
        data, feats = self._features(data, context)

        if len(data) < 14:
            raise ValueError("Not enough data for recursive forecasting (min 14 rows).")


        split = int(len(data) * 0.8)
        train, test = data.iloc[:split], data.iloc[split:]

        model = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(train[feats], train['y'])

        if mode == "Validation":
            test['yhat'] = model.predict(test[feats])
            return train, test, test[['ds', 'yhat']]

        # 30-day forecast
        history = data.copy()
        future = []

        for _ in range(30):
            history, feats = self._features(history, context, dropna=False)

            last = history.iloc[-1:]
            last = last.dropna()

            if last.empty:
                break
            pred = model.predict(last[feats])[0]
            next_date = last['ds'].values[0] + pd.Timedelta(days=1)

            history = pd.concat(
                [history, pd.DataFrame({'ds': [next_date], 'y': [pred]})],
                ignore_index=True
            )
            future.append({'ds': next_date, 'yhat': pred})

        return history, None, pd.DataFrame(future)

    def _prophet(self, data, mode):
        m = Prophet()
        split = int(len(data) * 0.8)
        train, test = data.iloc[:split], data.iloc[split:]

        if mode == "Validation":
            m.fit(train)
            fc = m.predict(test[['ds']])
            return train, test, fc[['ds', 'yhat']]

        m.fit(data)
        fc = m.predict(m.make_future_dataframe(30))
        return data, None, fc[['ds', 'yhat']]


    # main entry
    def run_forecast(self, df, date_col, target_col, model, mode, context):
        data = self._prepare_data(df, date_col, target_col, context)

        if model == "XGBoost":
            train, test, forecast = self._xgboost(data, context, mode)
        else:
            train, test, forecast = self._prophet(data, mode)

        metrics = {}
        if mode == "Validation" and test is not None:
            y_true = test['y'].reset_index(drop=True)
            y_pred = forecast['yhat'].reset_index(drop=True)

            metrics['MAE'] = mean_absolute_error(y_true, y_pred)

            if context == "Retail":
                metrics['SMAPE'] = (
                    100 * np.mean(
                        2 * np.abs(y_pred - y_true) /
                        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
                    )
                )

            if context == "Financial":
                metrics['Direction'] = self._direction_accuracy(y_true, y_pred)

        return train, test, forecast, metrics
