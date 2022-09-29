import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta

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
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA

import statsmodels.datasets
data = statsmodels.datasets.co2.load_pandas().data
data.co2.interpolate(inplace=True)


price_data = 'INDEX_S5FI, 1D.csv'
#price_data = 'BITSTAMP_BTCUSD, 1W.csv'
file = path.join('data', price_data)
df = pd.read_csv(file)
df['sma5_log'] = ta.sma(np.log(df["close"]), length=1)
df['sma5'] = ta.sma(df["close"], length=1)


y = df["close"]#df['sma5']
y = y.dropna()

x = list(range(len(y)))

#plot_series(y)

# convert into x and y
#x = list(range(len(data.index)))
#y = data.co2


# apply fast fourier transform and take absolute values
f=abs(np.fft.fft(y))

# get the list of frequencies
num=np.size(y)
freq = [i / num for i in list(range(num))]

# get the list of spectrums
spectrum=f.real*f.real+f.imag*f.imag
nspectrum=spectrum/spectrum[0]

# plot nspectrum per frequency, with a semilog scale on nspectrum
results = pd.DataFrame({'freq': freq, 'nspectrum': nspectrum})
results['period'] = results['freq'] / (1/52)
#plt.semilogy(results['period'], results['nspectrum'])

# improve the plot by convertint the data into grouped per week to avoid peaks
results['period_round'] = results['period'].round()
grouped_week = results.groupby('period_round')['nspectrum'].sum()
#plt.semilogy(grouped_week.index, grouped_week)
#plt.xticks([1, 13, 26, 39, 52])



from math import cos,pi
import numpy as np
import matplotlib.pyplot as plt

def plot_train(y, coeff, color, do_plot = True):
    offset = int(coeff*len(y))
    y_train = y[0:-offset]
    y_future = y
    fft = np.fft.fft(y_train)

    y_predicted = None
    for num_ in [50]:
        fft_list = np.copy(fft)
        fft_list[num_:-num_] = 0
        # Inverse Fast Fourier transform
        t = np.fft.ifft(fft_list)
        # The trend is your friend
        y_predicted = np.concatenate([t,t])
        
        if do_plot:
            plt.plot(y_predicted, color = color)

    return y_train, y_future, y_predicted, offset
    #plt.plot(np.arange(0, len(y_train)), y_train, color='blue')

# Generate Seasonal Data 
# X = [i for i in range(360 * 3)]
# slope = 1 / 365
# t_Y = [i * slope for i in X]
# s_Y = [60 * cos(2 * pi * i /365) for i in X]
# m_Y = [30 * cos(2 * pi * i /30) for i in X]

# c_Y = [a + b + c for (a,b,c) in zip(t_Y, s_Y, m_Y)]
# x = np.array(c_Y)

# Fast Fourier Transform 
y_train = y[0:-int(0.31*len(y))]
fft = np.fft.fft(y_train)
num=np.size(y_train)
freq = [i / num for i in list(range(num))]
spectrum=fft.real*fft.real+fft.imag*fft.imag
nspectrum=spectrum/spectrum[0]

results = pd.DataFrame({'freq': freq, 'nspectrum': nspectrum})
results['period'] = results['freq'] / (1/52)
#plt.semilogy(results['period'], results['nspectrum'])

plt.figure(figsize=(14, 7), dpi=100)

max_future = 5
max_tries = 3
start_from = 0.4
mean_err = 0
y_pred_av = pd.DataFrame()

for i in range(0, max_tries):
    y_train, y_future, y_pred, offset = plot_train(y, start_from + 0.01*i, 'red', do_plot=False)
    
    column_values = pd.Series(y_pred)
    y_pred_av.insert(loc=0, column=i, value=column_values)
    
    error = 0
    for k in range(0, max_future):
        bar = len(y) - offset + k+1
        error += abs(y[bar] - y_pred[bar])
    #error = error / max_future
    mean_err += error

mean_err = mean_err / max_tries

y_pred_mean = y_pred_av.sum(axis=1)/max_tries


y_train, y_future, y_pred, offset = plot_train(y, start_from, 'red', do_plot=False);
plt.plot(np.arange(0, len(y_pred_mean)), y_pred_mean, color='red', linewidth = 2)
plt.plot(np.arange(0, len(y_train)), y_train, color='blue', linewidth = 2)

#plot_train(y, 0.42, 'orange');
#plot_train(y, 0.44, 'magenta');
plt.plot(np.arange(0, len(y_future)), y_future, 'b', label = 'x', linewidth = 1, color='green')

plt.show()