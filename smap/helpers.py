import polars as pl
from smap import * 

def calc_features_consumption(df):
    # Add year and week columns
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])

    # Calculate basic weekly consumption statistics
    avg_weekly = c_week(df)  # Calculate average consumption per week
    max_weekly = s_max(df)   # Calculate maximum consumption per week
    min_weekly = s_min(df)   # Calculate minimum consumption per week

    # Calculate ratios and other derived features for further analysis
    ratio_mean_max = r_mean_max(df)
    ratio_min_mean = r_min_mean(df)
    ratio_night_mean = r_night_mean(df)
    ratio_morning_noon = r_morning_noon(df)
    ratio_evening_noon = r_evening_noon(df)
    ratio_mean_max_no_min = r_mean_max_no_min(df)
    ratio_evening_noon_no_min = r_evening_noon_no_min(df)
    ratio_morning_noon_no_min = r_morning_noon_no_min(df)
    ratio_day_night_no_min = r_day_night_no_min(df)
    ratio_var_wd_we = r_var_wd_we(df)
    ratio_min_wd_we = r_min_wd_we(df)
    ratio_max_wd_we = r_max_wd_we(df)
    ratio_evening_wd_we = r_evening_wd_we(df)
    ratio_night_wd_we = r_night_wd_we(df)
    ratio_noon_wd_we = r_noon_wd_we(df)
    ratio_morning_wd_we = r_morning_wd_we(df)
    ratio_afternoon_wd_we = r_afternoon_wd_we(df)
    ratio_we_night_day = r_we_night_day(df)
    ratio_we_morning_noon = r_we_morning_noon(df)
    ratio_we_evening_noon = r_we_evening_noon(df)
    ratio_wd_night_day = r_wd_night_day(df)
    ratio_wd_morning_noon = r_wd_morning_noon(df)
    ratio_wd_evening_noon = r_wd_evening_noon(df)

    max_weekend = s_we_max(df)
    min_weekend = s_we_min(df)
    max_weekday = s_wd_max(df)
    min_weekday = s_wd_min(df)

    correlation_we = s_cor_we(df)
    correlation_wd = s_cor_wd(df)
    correlation_wd_we = s_cor_wd_we(df)

    sm_variety = s_sm_variety(df)
    bg_variety = s_bg_variety(df)
    day_difference = s_day_diff(df)
    variance_we = s_var_we(df)
    variance_wd = s_var_wd(df)
    difference = s_diff(df)
    num_peaks = s_num_peaks(df)
    quartile_1 = s_q1(df)
    quartile_2 = s_q2(df)
    quartile_3 = s_q3(df)

    max_avg = c_max_avg(df)
    min_avg = c_min_avg(df)
    num_zeros = s_number_zeros(df)
    num_small_peaks = s_number_small_peaks(df)
    num_big_peaks = s_number_big_peaks(df)

    morning_avg = c_morning(df)
    noon_avg = c_noon(df)
    afternoon_avg = c_afternoon(df)
    evening_avg = c_evening(df)
    night_avg = c_night(df)
    weekday_avg = c_weekday(df)
    weekend_avg = c_weekend(df)
    var_weekday = c_var_weekday(df)
    morning_wd = c_wd_morning(df)
    noon_wd = c_wd_noon(df)
    afternoon_wd = c_wd_afternoon(df)
    evening_wd = c_wd_evening(df)
    night_wd = c_wd_night(df)
    var_weekend = c_var_weekend(df)
    morning_we = c_we_morning(df)
    noon_we = c_we_noon(df)
    afternoon_we = c_we_afternoon(df)
    evening_we = c_we_evening(df)
    night_we = c_we_night(df)
    week_no_min = c_week_no_min(df)

    time_above_1kw = t_above_1kw(df)
    time_above_2kw = t_above_2kw(df)
    time_above_mean = t_above_mean(df)
    daily_max_time = t_daily_max(df)
    daily_min_time = t_daily_min(df)

    # Variance and correlation calculations
    variance = s_variance(df)
    correlation = s_cor(df)

    # Temporal features like the first time daily consumption exceeds 1 kW
    first_time_above_1kw = t_above_1kw(df)

    # Seasonal and autocorrelation features
    seasonal_var_rem = ts_stl_varRem(df)  # Implement this function if STL decomposition is needed
    autocorr_mean_3h = ts_acf_mean3h(df)
    autocorr_mean_3h_wd = ts_acf_mean3h_weekday(df)

    # Peak detection related features
    wide_peak = t_wide_peaks(df)
    width_peak = t_width_peaks(df)

    # Combine all features into a single DataFrame
    features_list = [
        avg_weekly, max_weekly, min_weekly, ratio_mean_max, ratio_min_mean, ratio_night_mean,
        ratio_morning_noon, ratio_evening_noon, ratio_mean_max_no_min, ratio_evening_noon_no_min,
        ratio_morning_noon_no_min, ratio_day_night_no_min, ratio_var_wd_we, ratio_min_wd_we,
        ratio_max_wd_we, ratio_evening_wd_we, ratio_night_wd_we, ratio_noon_wd_we, ratio_morning_wd_we,
        ratio_afternoon_wd_we, ratio_we_night_day, ratio_we_morning_noon, ratio_we_evening_noon,
        ratio_wd_night_day, ratio_wd_morning_noon, ratio_wd_evening_noon, max_weekend, min_weekend,
        max_weekday, min_weekday, correlation_we, correlation_wd, correlation_wd_we,
        sm_variety, bg_variety, day_difference, variance_we, variance_wd, difference, num_peaks,
        quartile_1, quartile_2, quartile_3, max_avg, min_avg, num_zeros, num_small_peaks,
        num_big_peaks, morning_avg, noon_avg, afternoon_avg, evening_avg, night_avg,
        weekday_avg, weekend_avg, var_weekday, morning_wd, noon_wd, afternoon_wd,
        evening_wd, night_wd, var_weekend, morning_we, noon_we, afternoon_we,
        evening_we, night_we, week_no_min, time_above_1kw, time_above_2kw, time_above_mean,
        daily_max_time, daily_min_time, variance, correlation, first_time_above_1kw, seasonal_var_rem,
        autocorr_mean_3h, autocorr_mean_3h_wd, wide_peak, width_peak
    ]

    # Start with the first DataFrame and join the rest
    features = features_list[0].clone()
    for feature_df in features_list[1:]:
        features = features.join(feature_df, how='outer', left_on=['year', 'week'], right_on=['year', 'week'])

    return features