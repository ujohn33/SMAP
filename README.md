
# SMAP: Smart Meter Analytics Python 

SMAP (Smart Meter Analytics Python) is a Python library for smart meter analysis of consumption and temperature time-series data. It's designed as a Python implementation of the R package [SmartMeterAnalytics](https://cran.r-project.org/web/packages/SmartMeterAnalytics/index.html), providing similar functionality for Python users. SMAP is built using Polars, a dataframe library, optimized for high speed performance on scale. 

## Features

Derived from extensive research and machine learning analysis, SMAP uses smart meter and weather data to derive 77 features, enabling recognition of various household characteristics with high accuracy. These features facilitate detailed analyses of energy consumption behavior and extracting explanatory features of user behavior from the complex high-resolution time-series data. Current implementation aggregates smart meter data with 15 min / 30 min / 1 hr granularity into a curated set of engineered features characterizing various properties of smart meter consumption. A list of implemented features is provided below, and includes features describing statistical properties (mean, variance, quartiles, etc.), peak behavior, temporal behavior, correlation with temperature and various ratios (e.g., weekday/weekend). Correlation with other weather features (wind speed, humidity, etc.) is due to be added in later versions.

| Feature Function       | Description                                                      |
|------------------------|------------------------------------------------------------------|
| `c_week`               | Computes the average consumption for each week.                  |
| `s_max`                | Finds the maximum consumption for each week.                     |
| `s_min`                | Finds the minimum consumption for each week.                     |
| `c_morning`            | Calculates average consumption in the morning (6:00–9:59) weekly.|
| `c_noon`               | Calculates average consumption at noon (10:00–13:59) weekly.     |
| `c_afternoon`          | Calculates average consumption in the afternoon (14:00–17:59) weekly.|
| `c_evening`            | Calculates average consumption in the evening (18:00–21:59) weekly.|
| `c_night`              | Calculates average consumption at night (1:00–5:59) weekly.      |
| `c_weekday`            | Computes average consumption on weekdays (Monday–Friday).        |
| `c_var_weekday`        | Computes variance of consumption on weekdays (Monday–Friday).    |
| `s_wd_min`             | Finds the minimum consumption on weekdays per week.              |
| `s_wd_max`             | Finds the maximum consumption on weekdays per week.              |
| `c_wd_morning`         | Calculates average weekday morning consumption.                  |
| `c_wd_noon`            | Calculates average weekday noon consumption.                     |
| `c_wd_afternoon`       | Calculates average weekday afternoon consumption.                |
| `c_wd_evening`         | Calculates average weekday evening consumption.                  |
| `c_wd_night`           | Calculates average weekday night consumption.                    |
| `c_weekend`            | Calculates average consumption on weekends (Saturday–Sunday).    |
| `c_var_weekend`        | Computes variance of weekend consumption.                        |
| `s_we_min`             | Finds the minimum weekend consumption per week.                  |
| `s_we_max`             | Finds the maximum weekend consumption per week.                  |
| `c_we_morning`         | Calculates average weekend morning consumption.                  |
| `c_we_noon`            | Calculates average weekend noon consumption.                     |
| `c_we_afternoon`       | Calculates average weekend afternoon consumption.                |
| `c_we_evening`         | Calculates average weekend evening consumption.                  |
| `c_we_night`           | Calculates average weekend night consumption.                    |
| `c_week_no_min`        | Computes the difference between average and minimum weekly consumption.|
| `s_max_no_min`         | Computes the difference between maximum and minimum weekly consumption.|
| `c_evening_no_min`     | Computes evening consumption minus the minimum weekly consumption.|
| `c_morning_no_min`     | Computes morning consumption minus the minimum weekly consumption.|
| `c_noon_no_min`        | Computes noon consumption minus the minimum weekly consumption.  |
| `c_night_no_min`       | Computes night consumption minus the minimum weekly consumption. |
| `r_mean_max`           | Ratio between weekly average and maximum consumption.            |
| `r_min_mean`           | Ratio between minimum and weekly average consumption.            |
| `r_night_mean`         | Ratio between night and weekly average consumption.              |
| `r_morning_noon`       | Ratio between morning and noon consumption.                      |
| `r_evening_noon`       | Ratio between evening and noon consumption.                      |
| `r_mean_max_no_min`    | Ratio between average minus minimum and maximum minus minimum weekly consumption.|
| `r_evening_noon_no_min`| Ratio between evening minus minimum and noon minus minimum consumption.|
| `r_morning_noon_no_min`| Ratio between morning minus minimum and noon minus minimum consumption.|
| `r_day_night_no_min`   | Ratio between day minus minimum and night minus minimum consumption.|
| `r_var_wd_we`          | Ratio between variance of weekday and weekend consumption.       |
| `r_min_wd_we`          | Ratio between minimum weekday and weekend consumption.           |
| `r_max_wd_we`          | Ratio between maximum weekday and weekend consumption.           |
| `r_evening_wd_we`      | Ratio between evening weekday and weekend consumption.           |
| `r_night_wd_we`        | Ratio between night weekday and weekend consumption.             |
| `r_noon_wd_we`         | Ratio between noon weekday and weekend consumption.              |
| `r_morning_wd_we`      | Ratio between morning weekday and weekend consumption.           |
| `r_afternoon_wd_we`    | Ratio between afternoon weekday and weekend consumption.         |
| `s_sm_variety`     | Calculates the 20%-quintile of absolute differences in consumption (`cons`) per week and year. |
| `s_bg_variety`     | Calculates the 60%-quintile of absolute differences in consumption (`cons`) per week and year. |
| `s_day_diff`       | Computes the mean of absolute differences in daily consumption during weekdays, grouped by year and week. |
| `s_variance`       | Computes the variance of consumption (`cons`) grouped by year and week. |
| `s_var_wd`         | Computes the variance of weekday consumption grouped by year and week. |
| `s_var_we`         | Computes the variance of weekend consumption grouped by year and week. |
| `s_diff`           | Computes the total absolute difference in consumption for each year and week. |
| `s_cor`            | Computes the correlations between daily consumption patterns across different weekdays within a week. |
| `s_num_peaks`      | Identifies and counts the number of peaks in consumption data per week. |
| `s_q1`             | Calculates the first quartile (Q1) of consumption data for each week. |
| `s_q2`             | Calculates the median (Q2) of consumption data for each week. |
| `s_q3`             | Calculates the third quartile (Q3) of consumption data for each week. |
| `c_max_avg`        | Calculates the average of daily maximum consumption values per week. |
| `c_min_avg`        | Calculates the average of daily minimum consumption values per week. |
| `s_number_zeros`   | Counts the number of zero consumption readings per week. |
| `s_cor_wd`         | Computes average correlations between weekdays' consumption. |
| `s_cor_we`         | Computes correlations between Saturday and Sunday consumption. |
| `s_cor_wd_we`      | Computes correlations between average weekday and weekend consumption. |
| `s_number_small_peaks` | Identifies and counts smaller peaks in consumption within a week. |
| `s_number_big_peaks`   | Identifies and counts larger peaks in consumption within a week. |
| `w_temp_cor_overall`   | Computes the correlation between temperature and consumption on a weekly basis. |
| `w_temp_cor_night`     | Calculates correlation between night-time temperature and consumption for each week. |
| `w_temp_cor_daytime`   | Calculates correlation between daytime temperature and consumption during weekdays. |
| `w_temp_cor_evening`   | Calculates correlation between evening temperature and consumption during weekdays. |
| `w_temp_cor_minima`    | Analyzes the relationship between daily minimum temperatures and minimum consumption. |
| `w_temp_cor_maxima`    | Analyzes the relationship between daily maximum temperatures and maximum consumption. |
| `w_temp_cor_maxmin`    | Analyzes the relationship between daily minimum temperatures and maximum consumption. |
| `w_temp_cor_weekday_weekend` | Calculates the ratio of differences between weekday and weekend average temperatures and consumptions. |
| `t_above_1kw`      | Determines the first hour of the day where consumption exceeds 1 kW on weekdays. |
| `t_above_2kw`      | Determines the first hour of the day where consumption exceeds 2 kW on weekdays. |
| `t_above_mean`     | Identifies the first time each day that consumption exceeds the weekly mean. |
| `t_daily_max`      | Determines the time of day when daily maximum consumption occurs. |
| `t_daily_min`      | Determines the time of day when daily minimum consumption occurs. |
| `ts_stl_varRem`    | Applies STL decomposition and computes the variance of the remainder component. |
| `ts_acf_mean3h`    | Computes the mean autocorrelation of consumption data at 3-hour intervals. |
| `ts_acf_mean3h_weekday` | Computes the mean autocorrelation of weekday consumption data at 3-hour intervals. |
| `t_wide_peaks`     | Calculates the number of wide peaks in consumption data. |
| `t_width_peaks`    | Calculates the average width of peaks in consumption data. |


## Installation

Install SMAP via pip:

```bash
pip install SMAP
```

SMAP requires Python 3.6 or later and Polars 0.10.0 or later.

## Usage

Import SMAP and use its functions to analyze smart meter data effectively:

```python
import SMAP

# Create a DataFrame
data = {
    'dt': ['2023-01-01 19:30', '2023-01-01 20:00', '2023-01-08 13:00', '2023-01-09 20:00'],
    'cons': [15, 25, 35, 45]
}
df = pl.DataFrame(data)

# Analyze evening consumption
result = SMAP.c_evening_min(df)
```

## Contributing

Contributions are welcome! Please read the contributing guidelines before making any changes.

## References
1. Konstantin Hopf et al. (2016). "Predictive Analytics for Energy Efficiency and Energy Retailing." DOI: [10.20378/irbo-54833](https://doi.org/10.20378/irbo-54833).

2. Konstantin Hopf et al. (2018). "Enhancing energy efficiency in the residential sector with smart meter data analytics." Energy, vol. 28, pp. 453-473. DOI: [10.1007/s12525-018-0290-9](https://doi.org/10.1007/s12525-018-0290-9).


## License

SMAP is licensed under the terms of the MIT license.

## Contact

For any questions or feedback, please contact:

Evgenii Genov  
evgenii.genov@vub.be

