import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta

from sktime.utils.plotting import plot_series

import math
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
price_data = 'INDEX_S5FI, 1W.csv'
price_data = 'BATS_TSLA, 1W.csv'
file = path.join('data', price_data)
df = pd.read_csv(file)
df['sma5'] = ta.sma(np.log(df["close"]), length=2)


y = df['sma5']
y = y.dropna()

plot_series(y)

total = len(y)
total_slice = int(round(total / 2, 0))
training_set = y[:total_slice]
test_set = y[total_slice:]




