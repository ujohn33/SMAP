import polars as pl
import numpy as np
from statsmodels.tsa.stattools import acf
from typing import List

def calculate_lags_for_3h(df):
    # Calculate the time difference between the first two points to determine the frequency
    time_diff = df['dt'][1] - df['dt'][0]
    # Calculate the number of lags needed for a 3-hour period
    lags = int(3 * 60 / time_diff.total_seconds() * 60)
    return lags