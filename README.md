
# Smart Meter Analytics Python Library

SMAP (Smart Meter Analytics Python) is a Python library for smart meter analysis of consumption and temperature time-series data. It's designed as a Python implementation of the R package [SmartMeterAnalytics](https://cran.r-project.org/web/packages/SmartMeterAnalytics/index.html), providing similar functionality for Python users. SMAP is built using Polars, a dataframe library, optimized for high speed performance on scale. 

## Features

Derived from extensive research and machine learning analysis, SMAP uses smart meter and weather data to derive 133 features, enabling recognition of various household characteristics with high accuracy. These features facilitate detailed analyses of energy consumption behavior and extracting explanatory features of user behavior from the complex high-resolution time-series data. Current implementation aggregates smart meter data with 15 min / 30 min / 1 hr granularity into a curated set of engineered features characterizing various properties of smart meter consumption. A list of implemented features is provided below, and includes features describing statistical properties (mean, variance, quartiles, etc.), peak behavior, temporal behavior, correlation with temperature and various ratios (e.g., weekday/weekend). Correlation with other weather features (wind speed, humidity, etc.) is due to be added in later versions.

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


## License

SMAP is licensed under the terms of the MIT license.

## Contact

For any questions or feedback, please contact:

Evgenii Genov  
eugengenov@gmail.com  

