
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sktime.datasets import load_airline
from sktime.datasets import load_shampoo_sales

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sktime.datasets import load_lynx
from sktime.utils.plotting import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split

from sktime.utils.plotting import plot_series

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS


y = load_airline()

#plot_series(y)

# forecasting
fh = ForecastingHorizon(
    pd.period_range("1956-02", periods=48, freq="M"), is_relative=False
)

cutoff = pd.Period("1956-01", freq="M")
fh = fh.to_relative(cutoff)

print(fh)

y_train, y_test = temporal_train_test_split(y, fh=fh)

plot_series(y_train, y_test, labels=["y_train", "y_test"])

forecaster = NaiveForecaster(strategy="drift", window_length=10)
forecaster.fit(y_train)

y_pred = forecaster.predict(fh)
#plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

forecaster = AutoARIMA(sp=12, suppress_warnings=True)
forecaster.fit(y_train)
y_pred_arima = forecaster.predict(fh)

forecaster = AutoETS(auto=True, sp=12, n_jobs=-1)
forecaster.fit(y_train)
y_pred_autoETS = forecaster.predict(fh)

plot_series(y_train, y_test, y_pred, y_pred_arima, y_pred_autoETS, labels=["y_train", "y_test", "y_pred", "y_pred_arima", 'y_pred_autoETS'])











