import polars as pl
from smap import * 

def calc_features_consumption(df):
    # Add year and week columns
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])

    # Calculate basic features
    avg_week = c_week(df)
    max_week = s_max(df)
    min_week = s_min(df)

    avg_morning = c_morning(df)
    avg_noon = c_noon(df)
    avg_afternoon = c_afternoon(df)
    avg_evening = c_evening(df)
    avg_night = c_night(df)
    avg_weekday = c_weekday(df)
    avg_weekend = c_weekend(df)

    # Variance and correlation calculations
    variance = s_variance(df)
    correlation = s_cor(df)

    # Peak detection related features
    wide_peaks = t_wide_peaks(df)
    width_peaks = t_width_peaks(df)

    # Temporal features like the first time daily consumption exceeds 1 kW
    time_above_1kw = t_above_1kw(df)

    # Seasonal and autocorrelation features
    stl_var_rem = ts_stl_varRem(df)  # Implement this function if STL decomposition is needed
    acf_mean3h = ts_acf_mean3h(df)

    # Combine all features into a single DataFrame
    features_list = [
        avg_week, max_week, min_week,
        avg_morning, avg_noon, avg_afternoon, avg_evening, avg_night,
        avg_weekday, avg_weekend, variance, correlation,
        wide_peaks, width_peaks, time_above_1kw,
        stl_var_rem, acf_mean3h
    ]

    # Start with the first DataFrame and join the rest
    features = features_list[0].clone()
    for feature_df in features_list[1:]:
        features = features.join(feature_df, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])
    return features