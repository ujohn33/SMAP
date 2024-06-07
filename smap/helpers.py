import polars as pl
from smap import * 

def calc_features30_consumption(df):
    # Initial feature calculations
    features = pl.DataFrame()
    
    # Calculate average, max, min weekly consumption
    avg_week = c_week(df)
    max_week = s_max(df)
    min_week = s_min(df)
    
    # Calculate consumption for different parts of the day and week
    avg_morning = c_morning(df)
    avg_noon = c_noon(df)
    avg_afternoon = c_afternoon(df)
    avg_evening = c_evening(df)
    avg_night = c_night(df)
    avg_weekday = c_weekday(df)
    avg_weekend = c_weekend(df)
    
    # Combine all basic consumption stats into a single DataFrame
    # copy a dataframe in polars
    features = avg_week.clone()
    features = features.join(max_week, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(min_week, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(avg_morning, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(avg_noon, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(avg_afternoon, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(avg_evening, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(avg_night, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(avg_weekday, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(avg_weekend, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    
    # Variance and correlation calculations
    variance = s_variance(df)
    correlation = s_cor(df)  # Adjust this call if specific correlation calculations are needed
    features = features.join(variance, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(correlation, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    
    # Peak detection related features
    wide_peaks = t_wide_peaks(df)
    width_peaks = t_width_peaks(df)
    features = features.join(wide_peaks, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(width_peaks, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])

    # Temporal features like the first time daily consumption exceeds 1 kW
    time_above_1kw = t_above_1kw(df)
    features = features.join(time_above_1kw, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])

    # Seasonal and autocorrelation features (if implemented)
    # Example: STL decomposition variance and ACF
    stl_var_rem = ts_stl_varRem(df)  # Implement this function if STL decomposition is needed
    acf_mean3h = ts_acf_mean3h(df)
    features = features.join(stl_var_rem, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    features = features.join(acf_mean3h, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    return features