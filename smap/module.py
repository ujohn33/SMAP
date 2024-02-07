import polars as pl
import numpy as np
from typing import List
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from datetime import timedelta


# function inputs a time-series in polars, outputs average consumption for each weak 
def c_week(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average weekly consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekly average consumption
    """
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week and calculate the average consumption
    weekly_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons')
    )
    return weekly_avg


# function inputs a time-series in polars, outputs maximum consumption for each weak 
def s_max(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with maximum weekly consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekly maximum consumption
    """
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week and calculate the maximum consumption
    weekly_max = df.group_by(["year", "week"]).agg(
        pl.col('cons').max().alias('max_cons')
    )
    return weekly_max


# function inputs a time-series in polars, outputs minimum consumption for each weak
def s_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with minimum weekly consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekly minimum consumption
    """
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week and calculate the minimum consumption
    weekly_min = df.group_by(["year", "week"]).agg(
        pl.col('cons').min().alias('min_cons')
    )
    return weekly_min


# function inputs a time-series in polars, outputs average cons in the morning (6:00–9:59) for each weak
def c_morning(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average morning consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with morning average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([6, 7, 8, 9]))
    # Group by year and week and calculate the average consumption
    morning_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_morning')
    )
    return morning_avg


# function inputs a time-series in polars, outputs average cons at noon (10:00–13:59)
def c_noon(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average noon consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with noon average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([10, 11, 12, 13]))
    # Group by year and week and calculate the average consumption
    noon_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_noon')
    )
    return noon_avg


# function inputs a time-series in polars, outputs average cons in the afternoon (14:00–17:59)
def c_afternoon(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average afternoon consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with afternoon average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([14, 15, 16, 17]))
    # Group by year and week and calculate the average consumption
    afternoon_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_afternoon')
    )
    return afternoon_avg


# function inputs a time-series in polars, outputs average cons in at noon (18:00–21:59)
def c_evening(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average evening consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with evening average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([18, 19, 20, 21]))
    # Group by year and week and calculate the average consumption
    evening_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_evening')
    )
    return evening_avg


# function inputs a time-series in polars, outputs average cons in the night (1:00–5:59)
def c_night(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average night consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with night average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([1, 2, 3, 4, 5]))
    # Group by year and week and calculate the average consumption
    night_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_night')
    )
    return night_avg


# function inputs a time-series in polars, outputs average cons on working days (Monday–Friday)
def c_weekday(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average working days consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with working days average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([0, 1, 2, 3, 4]))
    # Group by year and week and calculate the average consumption
    working_days_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_wd')
    )
    return working_days_avg


# function inputs a time-series in polars, outputs variance of cons on working days (Monday–Friday)
def c_var_weekday(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average working days consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with working days average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([0, 1, 2, 3, 4]))
    # Group by year and week and calculate the average consumption
    working_days_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').var().alias('var_cons_wd')
    )
    return working_days_avg


# function inputs a time-series in polars, outputs variance of cons on working days (Monday–Friday)
def s_wd_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with minimum working days consumption per week.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with working days minimum consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([0, 1, 2, 3, 4]))
    # Group by year and week and calculate the average consumption
    working_days_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').min().alias('min_cons_wd')
    )
    return working_days_avg


# function inputs a time-series in polars, outputs maximum of cons on working days (Monday–Friday)
def s_wd_max(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with maximum working days consumption for each week.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with working days maximum consumption for each week
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([0, 1, 2, 3, 4]))
    # Group by year and week and calculate the average consumption
    working_days_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').max().alias('max_cons_wd')
    )
    return working_days_avg


# function inputs a time-series in polars, outputs average cons on weekdays in the morning (6:00–9:59)
def c_wd_morning(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average working days morning consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with working days morning average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([0, 1, 2, 3, 4]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([6, 7, 8, 9]))
    # Group by year and week and calculate the average consumption
    wd_morning_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_wd_morning')
    )
    return wd_morning_avg


# function inputs a time-series in polars, outputs average cons on weekdays at noon (10:00–13:59)
def c_wd_noon(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column 'cons',
    and returns a DataFrame with average working days noon consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with working days noon average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([0, 1, 2, 3, 4]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([10, 11, 12, 13]))
    # Group by year and week and calculate the average consumption
    wd_noon_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_wd_noon')
    )
    return wd_noon_avg


# function inputs a time-series in polars, outputs average cons on weekdays in the afternoon (14:00–17:59)
def c_wd_afternoon(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average working days afternoon
    consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with working days afternoon average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([0, 1, 2, 3, 4]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([14, 15, 16, 17]))
    # Group by year and week and calculate the average consumption
    wd_afternoon_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_wd_afternoon')
    )
    return wd_afternoon_avg


# function inputs a time-series in polars, outputs average cons on weekdays in the evening (18:00–21:59)
def c_wd_evening(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average working days evening
    consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with working days evening average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
        pl.col('dt').dt.hour().alias("hour")  # hour
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([0, 1, 2, 3, 4]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([18, 19, 20, 21]))
    # Group by year and week and calculate the average consumption
    wd_evening_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_wd_evening')
    )
    return wd_evening_avg


# function inputs a time-series in polars, outputs average cons on weekdays in the night (1:00–5:59)
def c_wd_night(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average working days night consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with working days night average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
        pl.col('dt').dt.hour().alias("hour")  # hour
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([0, 1, 2, 3, 4]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([1, 2, 3, 4, 5]))
    # Group by year and week and calculate the average consumption
    wd_night_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_wd_night')
    )
    return wd_night_avg


# function inputs a time-series in polars, outputs average cons on weekends (Saturday–Sunday)
def c_weekend(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average weekend consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekend average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([5, 6]))
    # Group by year and week and calculate the average consumption
    weekend_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_weekend')
    )
    return weekend_avg


# function inputs a time-series in polars, outputs variance cons on weekends (Saturday–Sunday)
def c_var_weekend(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with variance of weekend consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with variance of weekend consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([5, 6]))
    # Group by year and week and calculate the variance consumption
    weekend_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').var().alias('var_cons_weekend')
    )
    return weekend_avg


# function inputs a time-series in polars, outputs minimum cons on weekends
# (Saturday–Sunday) for each week
def s_we_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with minimum weekend consumption per week.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with minimum weekend consumption per week
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([5, 6]))
    # Group by year and week and calculate the average consumption
    weekend_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').min().alias('min_cons_weekend')
    )
    return weekend_avg


# function inputs a time-series in polars, outputs maximum cons on weekends
# (Saturday–Sunday) for each week
def s_we_max(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with maximum weekend consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekend maximum consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([5, 6]))
    # Group by year and week and calculate the maximum consumption
    weekend_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').max().alias('max_cons_weekend')
    )
    return weekend_avg


# function inputs a time-series in polars, outputs average cons on weekends in the morning (6:00–9:59)
def c_we_morning(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average weekend morning consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekend morning average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
        pl.col('dt').dt.hour().alias("hour")  # hour
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([5, 6]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([6, 7, 8, 9]))
    # Group by year and week and calculate the average consumption
    we_morning_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_we_morning')
    )
    return we_morning_avg


# function inputs a time-series in polars, outputs average cons on weekends at noon (10:00–13:59)
def c_we_noon(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average weekend noon consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekend noon average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
        pl.col('dt').dt.hour().alias("hour")  # hour
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([5, 6]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([10, 11, 12, 13]))
    # Group by year and week and calculate the average consumption
    we_noon_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_we_noon')
    )
    return we_noon_avg


# function inputs a time-series in polars, outputs average cons on weekends in the afternoon (14:00–17:59)
def c_we_afternoon(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average weekend afternoon consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekend afternoon average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
        pl.col('dt').dt.hour().alias("hour")  # hour
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([5, 6]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([14, 15, 16, 17]))
    # Group by year and week and calculate the average consumption
    we_afternoon_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_we_afternoon')
    )
    return we_afternoon_avg


# function inputs a time-series in polars, outputs average cons on weekends in the evening (18:00–21:59)
def c_we_evening(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average weekend evening consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekend evening average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
        pl.col('dt').dt.hour().alias("hour")  # hour
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([5, 6]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([18, 19, 20, 21]))
    # Group by year and week and calculate the average consumption
    we_evening_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_we_evening')
    )
    return we_evening_avg


# function inputs a time-series in polars, outputs average cons on weekends in the night (1:00–5:59)
def c_we_night(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average weekend night consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekend night average consumption
    """
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # weekday
        pl.col('dt').dt.hour().alias("hour")  # hour
    ])
    # Filter for working days
    df = df.filter(pl.col('weekday').is_in([5, 6]))
    # Filter for morning hours
    df = df.filter(pl.col('hour').is_in([1, 2, 3, 4, 5]))
    # Group by year and week and calculate the average consumption
    we_night_avg = df.group_by(["year", "week"]).agg(
        pl.col('cons').mean().alias('average_cons_we_night')
    )
    return we_night_avg


# function inputs a time-series in polars, outputs average cons for each week minus the minimum consumption for each week
def c_week_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with average weekly consumption minus the
    minimum weekly consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekly average consumption minus the minimum
        weekly consumption
    """
    # get the weekly average consumption
    weekly_avg = c_week(df)
    # get the minimum weekly consumption
    weekly_min = s_min(df)
    # outer join the weekly_avg and weekly_min DataFrames on the year and week columns
    weekly_avg = weekly_avg.join(weekly_min, on=['year', 'week'], how='outer')
    # subtract the minimum from the average_cons column of weekly_avg and rename to average_cons_min
    weekly_avg = weekly_avg.with_columns(
        (pl.col('average_cons') - pl.col('min_cons')).alias('cons_week_no_min')
    )
    # remove the min_cons and average_cons columns
    weekly_avg = weekly_avg.drop(['min_cons', 'average_cons'])
    return weekly_avg


# function inputs a time-series in polars, outputs max cons for each week minus the minimum consumption for each week
def s_max_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with maximum weekly consumption minus the
    minimum weekly consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with weekly maximum consumption minus the minimum
        weekly consumption
    """
    # get the maximum weekly consumption
    weekly_max = s_max(df)
    # get the minimum weekly consumption
    weekly_min = s_min(df)
    # outer join the weekly_max and weekly_min DataFrames on the year and week columns
    weekly_max = weekly_max.join(weekly_min, on=['year', 'week'], how='outer')
    # subtract the minimum from the max_cons column of weekly_max and rename to max_cons_min
    weekly_max = weekly_max.with_columns(
        (pl.col('max_cons') - pl.col('min_cons')).alias('cons_max_no_min')
    )
    # remove the min_cons and max_cons columns
    weekly_max = weekly_max.drop(['min_cons', 'max_cons'])
    return weekly_max


# function inputs a time-series in polars, outputs the output c_evening minus the minimum consumption
def c_evening_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with evening consumption minus the minimum
    consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with evening consumption minus the minimum
        consumption
    """
    evening_avg = c_evening(df)
    # get the minimum in the cons column of df 
    evening_min = s_min(df)
    # outer join the weekly_max and weekly_min DataFrames on the year and week columns
    evening_avg = evening_avg.join(evening_min, on=['year', 'week'], how='outer')
    # subtract the minimum from the average_cons_evening column of evening_avg and rename to average_cons_evening_min
    evening_avg = evening_avg.with_columns( 
        (pl.col('average_cons_evening') - pl.col('min_cons')).alias('cons_evening_no_min')
    )   
    # remove the min_cons and average_cons_evening columns
    evening_avg = evening_avg.drop(['min_cons', 'average_cons_evening'])
    return evening_avg


# function inputs a time-series in polars, outputs the output c_morning minus the minimum consumption
def c_morning_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with morning consumption minus the minimum
    consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with morning consumption minus the minimum
        consumption
    """
    morning_avg = c_morning(df)
    # get the minimum in the cons column of df 
    morning_min = s_min(df)
    # outer join the weekly_max and weekly_min DataFrames on the year and week columns
    morning_avg = morning_avg.join(morning_min, on=['year', 'week'], how='outer')
    # subtract the minimum from the average_cons_morning column of morning_avg and rename to average_cons_morning_min
    morning_avg = morning_avg.with_columns(
        (pl.col('average_cons_morning') - pl.col('min_cons')).alias('cons_morning_no_min')
    )
    # remove the min_cons and average_cons_morning columns
    morning_avg = morning_avg.drop(['min_cons', 'average_cons_morning'])
    return morning_avg


# function inputs a time-series in polars, outputs the output c_noon minus the minimum consumption
def c_noon_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with noon consumption minus the minimum
    consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with noon consumption minus the minimum
        consumption
    """
    noon_avg = c_noon(df)
    # get the minimum in the cons column of df 
    noon_min = s_min(df)
    # outer join the weekly_max and weekly_min DataFrames on the year and week columns
    noon_avg = noon_avg.join(noon_min, on=['year', 'week'], how='outer')
    # subtract the minimum from the average_cons_noon column of noon_avg and rename to average_cons_noon_min
    noon_avg = noon_avg.with_columns(
        (pl.col('average_cons_noon') - pl.col('min_cons')).alias('cons_noon_no_min')
    )
    # remove the min_cons and average_cons_noon columns
    noon_avg = noon_avg.drop(['min_cons', 'average_cons_noon'])
    return noon_avg


# function inputs a time-series in polars, outputs the output c_night minus the minimum consumption
def c_night_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with night consumption minus the minimum
    consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with night consumption minus the minimum
        consumption
    """
    night_avg = c_night(df)
    # get the minimum in the cons column of df 
    night_min = s_min(df)
    # outer join the weekly_max and weekly_min DataFrames on the year and week columns
    night_avg = night_avg.join(night_min, on=['year', 'week'], how='outer')
    # subtract the minimum from the average_cons_night column of night_avg and rename to average_cons_night_min
    night_avg = night_avg.with_columns(
        (pl.col('average_cons_night') - pl.col('min_cons')).alias('cons_night_no_min')
    )
    # remove the min_cons and average_cons_night columns
    night_avg = night_avg.drop(['min_cons', 'average_cons_night'])
    return night_avg


# function inputs a time-series in polars, outputs the ratio between c_week and max cons
def r_mean_max(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the weekly average
    consumption and the maximum consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the weekly average
        consumption and the maximum consumption
    """
    # get the weekly average consumption
    weekly_avg = c_week(df)
    # get the maximum consumption 
    weekly_max = s_max(df)
    # outer join the weekly_avg and weekly_max DataFrames on the year and week columns
    weekly_avg = weekly_avg.join(weekly_max, on=['year', 'week'], how='outer')
    # divide the weekly average consumption by the maximum consumption and rename to ratio_mean_max
    weekly_avg = weekly_avg.with_columns(
        (pl.col('average_cons') / pl.col('max_cons')).alias('ratio_mean_max')
    )
    # remove the max_cons and average_cons columns
    weekly_avg = weekly_avg.drop(['max_cons', 'average_cons'])
    return weekly_avg


# function inputs a time-series in polars, outputs the ratio between min cons and c_week
def r_min_mean(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the minimum
    consumption and the weekly average consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the minimum consumption and
        the weekly average consumption
    """
    # get the weekly average consumption
    weekly_avg = c_week(df)
    # get the minimum consumption 
    weekly_min = s_min(df)
    # outer join the weekly_avg and weekly_min DataFrames on the year and week columns
    weekly_avg = weekly_avg.join(weekly_min, on=['year', 'week'], how='outer')
    # divide the minimum consumption by the weekly average consumption and rename to ratio_min_mean
    weekly_avg = weekly_avg.with_columns(
        (pl.col('min_cons') / pl.col('average_cons')).alias('ratio_min_mean')
    )
    # remove the min_cons and average_cons columns
    weekly_avg = weekly_avg.drop(['min_cons', 'average_cons'])
    return weekly_avg


# function inputs a time-series in polars, outputs the ratio between c_night and c_week
def r_night_mean(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the night consumption
    and the weekly average consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the night consumption and
        the weekly average consumption
    """
    # get the weekly average consumption
    weekly_avg = c_week(df)
    # get the night consumption
    night_avg = c_night(df)
    # outer join the weekly_avg and night_avg DataFrames on the year and week columns
    weekly_avg = weekly_avg.join(night_avg, on=['year', 'week'], how='outer')
    # fill the missing values with 0
    weekly_avg = weekly_avg.fill_null(0)
    # divide the night consumption by the weekly average consumption and rename to ratio_night_mean
    weekly_avg = weekly_avg.with_columns(
        (pl.col('average_cons_night') / pl.col('average_cons')).alias('ratio_night_mean')
    )
    # remove the average_cons column
    weekly_avg = weekly_avg.drop(['average_cons', 'average_cons_night'])
    return weekly_avg


# function inputs a time-series in polars, outputs the ratio between c_morning and c_noon
def r_morning_noon(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the morning
    consumption and the noon consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the morning consumption and
        the noon consumption
    """
    # get the morning consumption
    morning_avg = c_morning(df)
    # get the noon consumption
    noon_avg = c_noon(df)
    # outer join the morning_avg and noon_avg DataFrames on the year and week columns
    morning_avg = morning_avg.join(noon_avg, on=['year', 'week'], how='outer')
    # fill the missing values with 0
    morning_avg = morning_avg.fill_null(0)
    # divide the morning consumption by the noon consumption and rename to ratio_morning_noon
    morning_avg = morning_avg.with_columns(
        (pl.col('average_cons_morning') / pl.col('average_cons_noon')).alias('ratio_morning_noon')
    )
    # remove the average_cons column
    morning_avg = morning_avg.drop(['average_cons_morning', 'average_cons_noon'])
    return morning_avg


# function inputs a time-series in polars, outputs the ratio between c_evening and c_noon
def r_evening_noon(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the evening
    consumption and the noon consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the evening consumption and
        the noon consumption
    """
    # get the evening consumption
    evening_avg = c_evening(df)
    # get the noon consumption
    noon_avg = c_noon(df)
    # outer join the evening_avg and noon_avg DataFrames on the year and week columns
    evening_avg = evening_avg.join(noon_avg, on=['year', 'week'], how='outer')
    # fill the missing values with 0
    evening_avg = evening_avg.fill_null(0)
    # divide the evening consumption by the noon consumption and rename to ratio_evening_noon
    evening_avg = evening_avg.with_columns(
        (pl.col('average_cons_evening') / pl.col('average_cons_noon')).alias('ratio_evening_noon')
    )
    # remove the average_cons column
    evening_avg = evening_avg.drop(['average_cons_evening', 'average_cons_noon'])
    return evening_avg


# function inputs a time-series in polars, outputs the ratio between c_week and s_max with min cons subtracted from both
def r_mean_max_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the weekly average
    consumption minus the minimum consumption and the maximum consumption minus
    the minimum consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the weekly average
        consumption minus the minimum consumption and the maximum consumption
        minus the minimum consumption
    """
    # get the weekly average consumption minus the minimum consumption
    weekly_avg = c_week_no_min(df)
    # get the maximum consumption minus the minimum consumption
    weekly_max = s_max_no_min(df)
    # outer join the weekly_avg and weekly_max DataFrames on the year and week columns
    weekly_avg = weekly_avg.join(weekly_max, on=['year', 'week'], how='outer')
    # divide the weekly average consumption minus the minimum consumption by the maximum consumption minus the minimum consumption and rename to ratio_mean_max_no_min
    weekly_avg = weekly_avg.with_columns(
        (pl.col('cons_week_no_min') / pl.col('cons_max_no_min')).alias('ratio_mean_max_no_min')
    )
    # remove the max_cons_no_min and cons_no_min columns
    weekly_avg = weekly_avg.drop(['cons_week_no_min', 'cons_max_no_min'])
    return weekly_avg


# function inputs a time-series in polars, outputs the ratio between c_evening and c_noon with min cons subtracted from both
def r_evening_noon_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the evening
    consumption minus the minimum consumption and the noon consumption minus the
    minimum consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the evening consumption
        minus the minimum consumption and the noon consumption minus the minimum
        consumption
    """
    # get the evening consumption minus the minimum consumption
    evening_avg = c_evening_no_min(df)
    # get the noon consumption minus the minimum consumption
    noon_avg = c_noon_no_min(df)
    # outer join the evening_avg and noon_avg DataFrames on the year and week columns
    evening_avg = evening_avg.join(noon_avg, on=['year', 'week'], how='outer')
    # fill the missing values with 0
    evening_avg = evening_avg.fill_null(0)
    # divide the evening consumption minus the minimum consumption by the noon consumption minus the minimum consumption and rename to ratio_evening_noon_no_min
    evening_avg = evening_avg.with_columns(
        (pl.col('cons_evening_no_min') / pl.col('cons_noon_no_min')).alias('ratio_evening_noon_no_min')
    )
    # remove the cons_evening_no_min and cons_noon_no_min columns
    evening_avg = evening_avg.drop(['cons_evening_no_min', 'cons_noon_no_min'])
    return evening_avg


# function inputs a time-series in polars, outputs the ratio between c_morning and c_noon with min cons subtracted from both
def r_morning_noon_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the morning
    consumption minus the minimum consumption and the noon consumption minus the
    minimum consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the morning consumption
        minus the minimum consumption and the noon consumption minus the minimum
        consumption
    """
    # get the morning consumption minus the minimum consumption
    morning_avg = c_morning_no_min(df)
    # get the noon consumption minus the minimum consumption
    noon_avg = c_noon_no_min(df)
    # outer join the morning_avg and noon_avg DataFrames on the year and week columns
    morning_avg = morning_avg.join(noon_avg, on=['year', 'week'], how='outer')
    # fill the missing values with 0
    morning_avg = morning_avg.fill_null(0)
    # divide the morning consumption minus the minimum consumption by the noon consumption minus the minimum consumption and rename to ratio_morning_noon_no_min
    morning_avg = morning_avg.with_columns(
        (pl.col('cons_morning_no_min') / pl.col('cons_noon_no_min')).alias('ratio_morning_noon_no_min')
    )
    # remove the cons_morning_no_min and cons_noon_no_min columns
    morning_avg = morning_avg.drop(['cons_morning_no_min', 'cons_noon_no_min'])
    return morning_avg


# function inputs a time-series in polars, outputs the ratio between c_night and c_week with min cons subtracted from both
def r_day_night_no_min(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the day consumption
    minus the minimum consumption and the night consumption minus the minimum
    consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the day consumption minus
        the minimum consumption and the night consumption minus the minimum
        consumption
    """
    # get the day consumption minus the minimum consumption
    day_avg = c_week_no_min(df)
    # get the night consumption minus the minimum consumption
    night_avg = c_night_no_min(df)
    # outer join the day_avg and night_avg DataFrames on the year and week columns
    day_avg = day_avg.join(night_avg, on=['year', 'week'], how='outer')
    # fill the missing values with 0
    day_avg = day_avg.fill_null(0)
    # divide the day consumption minus the minimum consumption by the night consumption minus the minimum consumption and rename to ratio_day_night_no_min
    day_avg = day_avg.with_columns(
        (pl.col('cons_week_no_min') / pl.col('cons_night_no_min')).alias('ratio_day_night_no_min')
    )
    # remove the cons_day_no_min and cons_night_no_min columns
    day_avg = day_avg.drop(['cons_week_no_min', 'cons_night_no_min'])
    return day_avg


# function inputs a time-series in polars, outputs the ratio between variance of c_weekday and c_weekend
def r_var_wd_we(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the working day
    consumption and the weekend consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the variance of working day
        consumption and the variance of the weekend consumption
    """
    # get the working day consumption
    wd_avg = c_var_weekday(df)
    # get the weekend consumption
    we_avg = c_var_weekend(df)
    # outer join the wd_avg and we_avg DataFrames on the year and week columns
    wd_avg = wd_avg.join(we_avg, on=['year', 'week'], how='outer')
    # divide the working day consumption by the weekend consumption and rename to ratio_var_wd_we
    wd_avg = wd_avg.with_columns(
        (pl.col('var_cons_wd') / pl.col('var_cons_weekend')).alias('ratio_var_wd_we')
    )
    # remove the average_cons_wd and average_cons_weekend columns
    wd_avg = wd_avg.drop(['var_cons_wd', 'var_cons_weekend'])
    return wd_avg


# function inputs a time-series in polars, outputs the Ratio of the minimum
# weekday / weekend day
def r_min_wd_we(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the working day
    consumption and the weekend consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the minimum of working day
        consumption and the minimum of the weekend consumption
    """
    # get the working day consumption
    wd_min = s_wd_min(df)
    # get the weekend consumption
    we_min = s_we_min(df)
    # outer join the wd_avg and we_avg DataFrames on the year and week columns
    wd_min = wd_min.join(we_min, on=['year', 'week'], how='outer')
    # divide the working day minimum consumption by the weekend minimum
    # consumption and rename to ratio_min_wd_we
    wd_min = wd_min.with_columns(
        (pl.col('min_cons_wd') / pl.col('min_cons_weekend')).alias('ratio_min_wd_we')
    )
    # remove the average_cons_wd and average_cons_weekend columns
    wd_min = wd_min.drop(['min_cons_wd', 'min_cons_weekend'])
    return wd_min


# function inputs a time-series in polars, outputs the Ratio of the maximum
# weekday / weekend day
def r_max_wd_we(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the working day
    consumption and the weekend consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between the maximum of working day
        consumption and the maximum of the weekend consumption
    """
    # get the working day consumption
    wd_max = s_wd_max(df)
    # get the weekend consumption
    we_max = s_we_max(df)
    # outer join the wd_avg and we_avg DataFrames on the year and week columns
    wd_max = wd_max.join(we_max, on=['year', 'week'], how='outer')
    # divide the working day maximum consumption by the weekend maximum
    # consumption and rename to ratio_max_wd_we
    wd_max = wd_max.with_columns(
        (pl.col('max_cons_wd') / pl.col('max_cons_weekend')).alias('ratio_max_wd_we')
    )
    # remove the average_cons_wd and average_cons_weekend columns
    wd_max = wd_max.drop(['max_cons_wd', 'max_cons_weekend'])
    return wd_max


# function inputs a time-series in polars, outputs the Ratio of consumption
# during evening, weekday / weekend day
def r_evening_wd_we(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the working day
    evening consumption and the weekend evening consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between working day evening
        consumption and the weekend evening consumption
    """
    # get the working day consumption
    wd_evening = c_wd_evening(df)
    # get the weekend consumption
    we_evening = c_we_evening(df)
    # outer join the wd_avg and we_avg DataFrames on the year and week columns
    wd_evening = wd_evening.join(we_evening, on=['year', 'week'], how='outer')
    # divide the working day maximum consumption by the weekend maximum
    # consumption and rename to ratio_max_wd_we
    wd_evening = wd_evening.with_columns(
        (pl.col('average_cons_wd_evening') / pl.col('average_cons_we_evening')).alias('ratio_evening_wd_we')
    )
    # remove the average_cons_wd and average_cons_weekend columns
    wd_evening = wd_evening.drop(['average_cons_wd_evening', 'average_cons_we_evening'])
    return wd_evening


# function inputs a time-series in polars, outputs the Ratio of consumption at
# night, weekday / weekend
def r_night_wd_we(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the working day
    night consumption and the weekend night consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between working day night
        consumption and the weekend night consumption
    """
    # get the working day consumption
    wd_night = c_wd_night(df)
    # get the weekend consumption
    we_night = c_we_night(df)
    # outer join the wd_avg and we_avg DataFrames on the year and week columns
    wd_night = wd_night.join(we_night, on=['year', 'week'], how='outer')
    # divide the working day maximum consumption by the weekend maximum
    # consumption and rename to ratio_max_wd_we
    wd_night = wd_night.with_columns(
        (pl.col('average_cons_wd_night') / pl.col('average_cons_we_night')).alias('ratio_night_wd_we')
    )
    # remove the average_cons_wd and average_cons_weekend columns
    wd_night = wd_night.drop(['average_cons_wd_night', 'average_cons_we_night'])
    return wd_night


# function inputs a time-series in polars, outputs the Ratio between consumption
# during nonn, weekday / weekend
def r_noon_wd_we(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the working day
    noon consumption and the weekend noon consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between working day noon
        consumption and the weekend noon consumption
    """
    # get the working day consumption
    wd_noon = c_wd_noon(df)
    # get the weekend consumption
    we_noon = c_we_noon(df)
    # outer join the wd_avg and we_avg DataFrames on the year and week columns
    wd_noon = wd_noon.join(we_noon, on=['year', 'week'], how='outer')
    # divide the working day maximum consumption by the weekend maximum
    # consumption and rename to ratio_max_wd_we
    wd_noon = wd_noon.with_columns(
        (pl.col('average_cons_wd_noon') / pl.col('average_cons_we_noon')).alias('ratio_noon_wd_we')
    )
    # remove the average_cons_wd and average_cons_weekend columns
    wd_noon = wd_noon.drop(['average_cons_wd_noon', 'average_cons_we_noon'])
    return wd_noon


# function inputs a time-series in polars, outputs the Ratio between consumption
# during morning, weekday / weekend
def r_morning_wd_we(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the working day
    morning consumption and the weekend morning consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between working day morning
        consumption and the weekend morning consumption
    """
    # get the working day consumption
    wd_morning = c_wd_morning(df)
    # get the weekend consumption
    we_morning = c_we_morning(df)
    # outer join the wd_avg and we_avg DataFrames on the year and week columns
    wd_morning = wd_morning.join(we_morning, on=['year', 'week'], how='outer')
    # divide the working day maximum consumption by the weekend maximum
    # consumption and rename to ratio_max_wd_we
    wd_morning = wd_morning.with_columns(
        (pl.col('average_cons_wd_morning') / pl.col('average_cons_we_morning')).alias('ratio_morning_wd_we')
    )
    # remove the average_cons_wd and average_cons_weekend columns
    wd_morning = wd_morning.drop(['average_cons_wd_morning', 'average_cons_we_morning'])
    return wd_morning


# function inputs a time-series in polars, outputs the Ratio between consumption
# during afternoon, weekday / weekend
def r_afternoon_wd_we(df):
    """
    Takes a DataFrame with a datetime column 'dt' and a consumption column
    'cons', and returns a DataFrame with the ratio between the working day
    afternoon consumption and the weekend afternoon consumption.

    :param df: Polars DataFrame with 'dt' and 'cons' columns
    :return: Polars DataFrame with the ratio between working day afternoon
        consumption and the weekend afternoon consumption
    """
    # get the working day consumption
    wd_afternoon = c_wd_afternoon(df)
    # get the weekend consumption
    we_afternoon = c_we_afternoon(df)
    # outer join the wd_avg and we_avg DataFrames on the year and week columns
    wd_afternoon = wd_afternoon.join(we_afternoon, on=['year', 'week'], how='outer')
    # divide the working day maximum consumption by the weekend maximum
    # consumption and rename to ratio_max_wd_we
    wd_afternoon = wd_afternoon.with_columns(
        (pl.col('average_cons_wd_afternoon') / pl.col('average_cons_we_afternoon')).alias('ratio_afternoon_wd_we')
    )
    # remove the average_cons_wd and average_cons_weekend columns
    wd_afternoon = wd_afternoon.drop(['average_cons_wd_afternoon', 'average_cons_we_afternoon'])
    return wd_afternoon


# function inputs a time-series in polars, outputs the Ratio c_we_night / c_we_weekend
def r_we_night_day(df):
    we_night = c_we_night(df)
    we_weekend = c_weekend(df)
    we_night = we_night.join(we_weekend, on=['year', 'week'], how='outer')
    we_night = we_night.with_columns(
        (pl.col('average_cons_we_night') / pl.col('average_cons_weekend')).alias('ratio_we_night_day')
    )
    we_night = we_night.drop(['average_cons_we_night', 'average_cons_weekend'])
    return we_night


# function inputs a time-series in polars, outputs the Ratio c_we_morning / c_we_ noon
def r_we_morning_noon(df):
    we_morning = c_we_morning(df)
    we_noon = c_we_noon(df)
    we_morning = we_morning.join(we_noon, on=['year', 'week'], how='outer')
    we_morning = we_morning.with_columns(
        (pl.col('average_cons_we_morning') / pl.col('average_cons_we_noon')).alias('ratio_we_morning_noon')
    )
    we_morning = we_morning.drop(['average_cons_we_morning', 'average_cons_we_noon'])
    return we_morning


def r_we_evening_noon(df):
    we_evening = c_we_evening(df)
    we_noon = c_we_noon(df)
    we_evening = we_evening.join(we_noon, on=['year', 'week'], how='outer')
    we_evening = we_evening.with_columns(
        (pl.col('average_cons_we_evening') / pl.col('average_cons_we_noon')).alias('ratio_we_evening_noon')
    )
    we_evening = we_evening.drop(['average_cons_we_evening', 'average_cons_we_noon'])
    return we_evening


def r_wd_night_day(df):
    wd_night = c_wd_night(df)
    wd_day = c_wd_noon(df)
    wd_night = wd_night.join(wd_day, on=['year', 'week'], how='outer')
    wd_night = wd_night.with_columns(
        (pl.col('average_cons_wd_night') / pl.col('average_cons_wd_noon')).alias('ratio_wd_night_day')
    )
    wd_night = wd_night.drop(['average_cons_wd_night', 'average_cons_wd_noon'])
    return wd_night


def r_wd_morning_noon(df):
    wd_morning = c_wd_morning(df)
    wd_noon = c_wd_noon(df)
    wd_morning = wd_morning.join(wd_noon, on=['year', 'week'], how='outer')
    wd_morning = wd_morning.with_columns(
        (pl.col('average_cons_wd_morning') / pl.col('average_cons_wd_noon')).alias('ratio_wd_morning_noon')
    )
    wd_morning = wd_morning.drop(['average_cons_wd_morning', 'average_cons_wd_noon'])
    return wd_morning


def r_wd_evening_noon(df):
    wd_evening = c_wd_evening(df)
    wd_noon = c_wd_noon(df)
    wd_evening = wd_evening.join(wd_noon, on=['year', 'week'], how='outer')
    wd_evening = wd_evening.with_columns(
        (pl.col('average_cons_wd_evening') / pl.col('average_cons_wd_noon')).alias('ratio_wd_evening_noon')
    )
    wd_evening = wd_evening.drop(['average_cons_wd_evening', 'average_cons_wd_noon'])
    return wd_evening


def s_sm_variety(df):
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
    ])
    # Calculate the difference in 'cons' and then the 20%-quintile for each group
    df_variety = df.group_by(["year", "week"]).agg(
        [
            pl.col('cons').diff().abs().alias('sm_variety'),
            #pl.col('sm_variety').quantile(0.20).alias('20pct_sm_variety')
        ]
    )
    return df_variety


# 60%-quintile of the deviation from the previous measured value
def s_bg_variety(df):
    # Add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
    ])
    # Calculate the difference in 'cons' and then the 20%-quintile for each group
    df_variety = df.group_by(["year", "week"]).agg(
        [
            pl.col('cons').diff().abs().quantile(0.60).alias('20pct_sm_variety')
        ]
    )
    return df_variety


# Deviation of measured values on weekdays
def s_day_diff(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),  # year
        pl.col('dt').dt.week().alias("week"),  # week
        pl.col('dt').dt.weekday().alias("weekday"),  # day of the week
    ])
    # Filter for weekdays (Monday=0, ..., Sunday=6)
    weekday_df = df.filter(pl.col('weekday') < 5)
    # Group by year and week, calculate standard deviation of measured values
    # Replace 'value_column' with the name of your measured values column
    result_df = weekday_df.group_by(["year", "week"]).agg(
        pl.col('cons').diff().abs().mean().alias('weekday_diff')
    )
    return result_df


# Variance
def s_variance(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week, and calculate the variance of 'cons'
    result_df = df.group_by(["year", "week"]).agg(
        pl.col('cons').var().alias('cons_variance')
    )
    return result_df


# Variance on weekdays
def s_var_wd(df):
    # Add columns for the year, week, and day of the week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Filter for weekdays (Monday=0, ..., Sunday=6)
    weekday_df = df.filter(pl.col('weekday') < 5)
    # Group by year and week, and calculate the variance of 'cons'
    result_df = weekday_df.group_by(["year", "week"]).agg(
        pl.col('cons').var().alias('cons_varianc_wd')
    )
    return result_df


# Variance on weekends
def s_var_we(df):
    # Add columns for the year, week, and day of the week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Filter for weekends (Saturday=5, Sunday=6)
    weekend_df = df.filter(pl.col('weekday') >= 5)
    # Group by year and week, and calculate the variance of 'cons'
    result_df = weekend_df.group_by(["year", "week"]).agg(
        pl.col('cons').var().alias('cons_variance_we')
    )
    return result_df


# Total of differences from predecessor (absolute value)
def s_diff(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week, calculate the total absolute difference of 'cons'
    result_df = df.group_by(["year", "week"]).agg(
        pl.col('cons').diff().abs().sum().alias('total_abs_diff')
    )
    return result_df


def s_cor(df):
    # Add columns for year, week, weekday, and time
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.time().alias("time")  # Use time directly
    ])
    # Reshape data: create a column for each weekday's consumption
    pivot_df = df.pivot(
        index=['year', 'week', 'time'],
        columns='weekday',
        values='cons'
    ).fill_null(pl.lit(0))  # Fill missing values
    # Rename columns for clarity
    day_cols = [f'cons_weekday{i}' for i in range(7)]
    pivot_df.columns = ['year', 'week', 'time'] + day_cols
    # Compute correlations for each pair of days
    correlations = []
    correlation_cols = []
    for i in range(1,6):  # Correlate each day with the next
        correlation_col = pl.corr(f'cons_weekday{i}', f'cons_weekday{i+1}').alias(f'day{i}_day{i+1}_correlation')
        correlations.append(correlation_col)
        correlation_cols.append(f'day{i}_day{i+1}_correlation')
    # Group by year, week, and time, and aggregate the correlations
    result_df = pivot_df.group_by(['year', 'week']).agg(correlations)
    mean_correlation = result_df.select(
        col_mean = pl.concat_list(correlation_cols).list.mean()
    )
    # Final DataFrame with year, week, and mean_correlation
    final_df = result_df.select(['year', 'week']).with_columns(mean_correlation)
    final_df = final_df.rename({"col_mean": "mean_cor"})
    return final_df


def s_num_peaks(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias('year'),
        pl.col('dt').dt.week().alias('week')
    ])

    # Function to apply to each group
    def calc_peaks(cons: List[pl.Series]):
        # Check if the group has enough data
        if len(cons[0]) < 240:
            return 0
        # Extract the data
        data = cons[0].to_numpy()[:240]
        # Apply lowess smoothing
        smoothed = lowess(data, np.arange(len(data)), frac=0.02)[:, 1]
        # Calculate differences, apply sign, and find second differences
        diff_smoothed = np.diff(smoothed)
        sign_diff = np.diff(np.sign(diff_smoothed))
        # Count peaks
        num_peaks = np.sum(sign_diff == 2)
        return num_peaks

    # Apply the function to each group and collect results
    result_df = df.group_by(['year', 'week']).agg([
        pl.apply(pl.col('cons'), calc_peaks).alias('num_peaks')
    ])
    return result_df


def s_q1(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week, calculate the lower quartile of 'cons'
    result_df = df.group_by(["year", "week"]).agg(
        pl.col('cons').quantile(0.25).alias('lower_quartile')
    )
    return result_df


def s_q2(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week, calculate the lower quartile of 'cons'
    result_df = df.group_by(["year", "week"]).agg(
        pl.col('cons').quantile(0.5).alias('median')
    )
    return result_df


def s_q3(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week, calculate the lower quartile of 'cons'
    result_df = df.group_by(["year", "week"]).agg(
        pl.col('cons').quantile(0.75).alias('upper_quartile')
    )
    return result_df


def c_max_avg(df):
    # Add columns for the year, week, and day
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.day().alias("day")
    ])
    # Calculate daily maxima and then average these for each week
    result_df = df.group_by(["year", "week", "day"]).agg(
        pl.col('cons').max().alias('daily_max')
    ).group_by(["year", "week"]).agg(
        pl.col('daily_max').mean().alias('weekly_avg_max')
    )
    return result_df


def c_min_avg(df):
    # Add columns for the year, week, and day
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.day().alias("day")
    ])
    # Calculate daily minima and then average these for each week
    result_df = df.group_by(["year", "week", "day"]).agg(
        pl.col('cons').min().alias('daily_min')
    ).group_by(["year", "week"]).agg(
        pl.col('daily_min').mean().alias('weekly_avg_min')
    )
    return result_df


# Number of zero values
def s_number_zeros(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week, count the number of zeros in 'cons'
    result_df = df.group_by(["year", "week"]).agg(
        (pl.col('cons') == 0).sum().alias('zero_count')
    )
    return result_df


# Average Correlation between weekdays
def s_cor_wd(df):
    # Add columns for year, week, weekday, and time
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.time().alias("time")  # Use time directly
    ])
    # Reshape data: create a column for each weekday's consumption
    pivot_df = df.pivot(
        index=['year', 'week', 'time'],
        columns='weekday',
        values='cons'
    ).fill_null(pl.lit(0))  # Fill missing values
    # Rename columns for clarity
    day_cols = [f'cons_weekday{i}' for i in range(7)]
    pivot_df.columns = ['year', 'week', 'time'] + day_cols
    # Compute correlations for each pair of days
    correlations = []
    correlation_cols = []
    for i in range(1,4):  # Correlate each day with the next
        correlation_col = pl.corr(f'cons_weekday{i}', f'cons_weekday{i+1}').alias(f'day{i}_day{i+1}_correlation')
        correlations.append(correlation_col)
        correlation_cols.append(f'day{i}_day{i+1}_correlation')
    # Group by year, week, and time, and aggregate the correlations
    result_df = pivot_df.group_by(['year', 'week']).agg(correlations)
    mean_correlation = result_df.select(
        col_mean = pl.concat_list(correlation_cols).list.mean()
    )
    # Final DataFrame with year, week, and mean_correlation
    final_df = result_df.select(['year', 'week']).with_columns(mean_correlation)
    final_df = final_df.rename({"col_mean": "mean_cor_wd"})
    return final_df

# Correlation between Sat and Sun
def s_cor_we(df):
    # Add columns for year, week, weekday, and time
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.time().alias("time")  # Use time directly
    ])
    # Reshape data: create a column for each weekday's consumption
    pivot_df = df.pivot(
        index=['year', 'week', 'time'],
        columns='weekday',
        values='cons'
    ).fill_null(pl.lit(0))  # Fill missing values
    # Rename columns for clarity
    day_cols = [f'cons_weekday{i}' for i in range(7)]
    pivot_df.columns = ['year', 'week', 'time'] + day_cols
    # Compute correlations for each pair of days
    correlations = []
    correlation_cols = []
    for i in range(5,6):  # Correlate each day with the next
        correlation_col = pl.corr(f'cons_weekday{i}', f'cons_weekday{i+1}').alias(f'day{i}_day{i+1}_correlation')
        correlations.append(correlation_col)
        correlation_cols.append(f'day{i}_day{i+1}_correlation')
    # Group by year, week, and time, and aggregate the correlations
    result_df = pivot_df.group_by(['year', 'week']).agg(correlations)
    mean_correlation = result_df.select(
        col_mean = pl.concat_list(correlation_cols).list.mean()
    )
    # Final DataFrame with year, week, and mean_correlation
    final_df = result_df.select(['year', 'week']).with_columns(mean_correlation)
    final_df = final_df.rename({"col_mean": "mean_cor_we"})
    return final_df


def s_cor_wd_we(df):
    # Add columns for year, week, and weekday
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.time().alias("time")
    ])
    # Group by year and week
    df_weekday = df.filter(
        pl.col("weekday") < 5
    )
    df_weekend = df.filter(
        pl.col("weekday") >= 5
    )
    # Use conditional aggregation to calculate averages for weekdays and weekends
    result_df_weekday = df_weekday.group_by(['year', 'week', 'time']).agg(pl.mean('cons').alias('weekday_avg'))
    result_df_weekend = df_weekend.group_by(['year', 'week', 'time']).agg(pl.mean('cons').alias('weekend_avg'))
    result_df = result_df_weekday.join(result_df_weekend, on=["year", "week", 'time'], how='outer')
    # Calculate the correlation between weekday and weekend averages
    result_df = result_df.group_by(['year', 'week']).agg(
        pl.corr('weekday_avg', 'weekend_avg').alias('correlation')
    )
    return result_df


def s_number_small_peaks(df):
    # add 'year' and 'week' columns
    df = df.with_columns([
        pl.col('dt').dt.year().alias('year'),
        pl.col('dt').dt.week().alias('week')
    ])
    weekdays = 24*5*4

    # Function to apply to each group
    def calc_small_peaks(cons: List[pl.Series]):
        # Check if the group has enough data
        if len(cons[0]) < 240:
            return None
        # Extract the data
        data = cons[0].to_numpy()[:weekdays]
        # Apply lowess smoothing
        smoothed = lowess(data, np.arange(len(data)), frac=0.02)[:, 1]
        # Calculate differences, apply sign, and find second differences
        diff_smoothed = np.diff(smoothed)
        sign_diff = np.diff(np.sign(diff_smoothed))
        # Count peaks
        num_peaks = np.sum(sign_diff == 2)
        return num_peaks

    # Apply the function to each group and collect results
    result_df = df.group_by(['year', 'week']).agg([
        pl.apply(pl.col('cons'), calc_small_peaks).alias('num_small_peaks')
    ])
    return result_df


def s_number_big_peaks(df):
    # add 'year' and 'week' columns
    df = df.with_columns([
        pl.col('dt').dt.year().alias('year'),
        pl.col('dt').dt.week().alias('week')
    ])
    weekdays = 24*5*4

    # Function to apply to each group
    def calc_big_peaks(cons: List[pl.Series]):
        # Check if the group has enough data
        if len(cons[0]) < 240:
            return None
        # Extract the data
        data = cons[0].to_numpy()[:weekdays]
        # Apply lowess smoothing
        smoothed = lowess(data, np.arange(len(data)), frac=0.05)[:, 1]
        # Calculate differences, apply sign, and find second differences
        diff_smoothed = np.diff(smoothed)
        sign_diff = np.diff(np.sign(diff_smoothed))
        # Count peaks
        num_peaks = np.sum(sign_diff == 2)
        return num_peaks

    # Apply the function to each group and collect results
    result_df = df.group_by(['year', 'week']).agg([
        pl.apply(pl.col('cons'), calc_big_peaks).alias('num_big_peaks')
    ])
    return result_df


def w_temp_cor_overall(df, weather_col='temp'):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])

    # Group by year and week, calculate the correlation between temperature and power consumption
    result_df = df.group_by(["year", "week"]).agg(
        pl.corr(weather_col, 'cons').alias('temp_cons_cor')
    )
    return result_df


# Function to apply to each group for linear regression
def calc_linear_relationship(args: List[pl.Series]):
    if args[0].is_null().any() or args[1].is_null().any():
        return None
    # Prepare data for linear regression
    X = sm.add_constant(args[0].to_numpy())  # Independent variable (temperature)
    y = args[1].to_numpy()  # Dependent variable (consumption)
    # Fit linear regression model and extract the coefficient
    model = sm.OLS(y, X, missing='drop').fit()
    return model.params[1]  # Return the slope coefficient


def w_temp_cor_daily(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Group by year and week and apply the linear regression function
    result_df = df.group_by(["year", "week"]).agg(pl.apply(exprs=["temp", "cons"], function=calc_linear_relationship).alias('temp_cons_cor'))
    return result_df


def w_temp_cor_night(df):
    # add columns for the year, week, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for night hours (0:00 - 5:59)
    df_night = df.filter((pl.col('hour') >= 0) & (pl.col('hour') < 6))
    # Group by year and week and apply the linear regression function
    result_df = df_night.group_by(["year", "week"]).agg(pl.apply(exprs=["temp", "cons"], function=calc_linear_relationship).alias('temp_cons_cor_night'))
    return result_df


def w_temp_cor_daytime(df):
    # Add columns for the year, week, day, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for daytime hours (6:00 - 17:59) from Monday to Friday (0-4)
    df_daytime = df.filter((pl.col('hour') >= 6) & (pl.col('hour') <= 17) & (pl.col('weekday') < 5))
    # Group by year, week, and day and apply the linear regression function
    result_df = df_daytime.group_by(["year", "week"]).agg(pl.apply(exprs=["temp", "cons"], function=calc_linear_relationship).alias('temp_cons_cor_daytime'))
    return result_df


def w_temp_cor_evening(df):
    # Add columns for the year, week, day, and hour
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for evening hours (18:00 - 23:59) from Monday to Friday (0-4)
    df_evening = df.filter((pl.col('hour') >= 18) & (pl.col('hour') <= 23))
    # Group by year, week, and day and apply the linear regression function
    result_df = df_evening.group_by(["year", "week"]).agg(pl.apply(exprs=["temp", "cons"], function=calc_linear_relationship).alias('temp_cons_cor_evening'))
    return result_df
    

def w_temp_cor_minima(df):
    # Add columns for the year, week, and day
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Group by year, week, and day and calculate the daily minimum for 'cons' and 'temp'
    daily_min_df = df.group_by(["year", "week", "weekday"]).agg([
        pl.col('cons').min().alias('daily_min_cons'),
        pl.col('temp').min().alias('daily_min_temp')
    ])

    # Group by year and week and calculate linear relationship between daily minimums
    def calc_linear_relationship_minima(args: List[pl.Series]):
        X = sm.add_constant(args[0].to_numpy())  # Independent variable (temperature)
        y = args[1].to_numpy()  # Dependent variable (consumption)
        # Fit linear regression model and extract the coefficient
        model = sm.OLS(y, X, missing='drop').fit()
        return model.params[1]  # Return the slope coefficient
    
    # Apply the linear regression function to each group
    result_df = daily_min_df.group_by(["year", "week"]).agg(pl.apply(exprs=["daily_min_temp", "daily_min_cons"], function=calc_linear_relationship_minima).alias('min_temp_cons_correlation'))
    return result_df


def w_temp_cor_maxima(df):
    # Add columns for the year, week, and day
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Group by year, week, and day and calculate the daily minimum for 'cons' and 'temp'
    daily_min_df = df.group_by(["year", "week", "weekday"]).agg([
        pl.col('cons').max().alias('daily_max_cons'),
        pl.col('temp').max().alias('daily_max_temp')
    ])

    # Group by year and week and calculate linear relationship between daily minimums
    def calc_linear_relationship_maxima(args: List[pl.Series]):
        if args[0].is_null().any() or args[1].is_null().any():
            return None
        X = sm.add_constant(args[0].to_numpy())  # Independent variable (temperature)
        y = args[1].to_numpy()  # Dependent variable (consumption)
        # Fit linear regression model and extract the coefficient
        model = sm.OLS(y, X, missing='drop').fit()
        return model.params[1]  # Return the slope coefficient
    
    # Apply the linear regression function to each group
    result_df = daily_min_df.group_by(["year", "week"]).agg(pl.apply(exprs=["daily_max_temp", "daily_max_cons"], function=calc_linear_relationship_maxima).alias('max_temp_cons_correlation'))
    return result_df


def w_temp_cor_maxmin(df):
    # Add columns for the year, week, and day
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Group by year, week, and day, and calculate the daily max for power consumption and min for temperature
    daily_values_df = df.group_by(["year", "week", "weekday"]).agg([
        pl.col('cons').max().alias('daily_max_cons'),
        pl.col('temp').min().alias('daily_min_temp')
    ])

    # Group by year and week and calculate linear relationship between daily minimums
    def calc_linear_relationship_maxmin(args: List[pl.Series]):
        if args[0].is_null().any() or args[1].is_null().any():
            return None
        X = sm.add_constant(args[0].to_numpy())  # Independent variable (temperature)
        y = args[1].to_numpy()  # Dependent variable (consumption)
        # Fit linear regression model and extract the coefficient
        model = sm.OLS(y, X, missing='drop').fit()
        return model.params[1]  # Return the slope coefficient
    
    # Apply the linear regression function to each group
    result_df = daily_values_df.group_by(["year", "week"]).agg(pl.apply(exprs=["daily_min_temp", "daily_max_cons"], function=calc_linear_relationship_maxmin).alias('maxmin_temp_cons_correlation'))
    return result_df


def w_temp_cor_weekday_weekend(df):
    # Add columns for the year, week, and weekday/weekend
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        (pl.col('dt').dt.weekday() < 5).alias("is_weekday")
    ])
    # Group by year, week, and is_weekday; calculate average 'cons' and 'temp'
    avg_df = df.group_by(["year", "week", "is_weekday"]).agg([
        pl.mean('cons').alias('avg_cons'),
        pl.mean('temp').alias('avg_temp')
    ])
    # Reshape the data to have separate columns for weekdays and weekends
    pivot_df = avg_df.pivot(index=["year", "week"], columns="is_weekday", values=["avg_cons", "avg_temp"])
    # Calculate the ratio (c_wd - c_we) / (t_wd - t_we) for each week
    ratio_df = pivot_df.with_columns([
        ((pl.col("avg_cons_is_weekday_true") - pl.col("avg_cons_is_weekday_false")) / 
         (pl.col("avg_temp_is_weekday_true") - pl.col("avg_temp_is_weekday_false"))).alias("weekday_weekend_ratio")
    ]).select(['year', 'week', 'weekday_weekend_ratio'])
    return ratio_df


def t_above_1kw(df):
    # Add columns for the year, week, and weekday
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for weekdays
    weekday_df = df.filter(pl.col('weekday') < 5)
    # Group by year, week, day, and hour, calculate the average consumption
    hourly_avg_df = weekday_df.filter(pl.col('cons') > 1).group_by(["year", "week", "weekday"]).agg(
        pl.min('hour').alias('daily_t_above_1kW_hour')
    )
    first_exceeding_df = hourly_avg_df.group_by(['year', 'week']).agg(
        pl.mean('daily_t_above_1kW_hour').alias('first_exceeding_1kW_hour')
    )
    return first_exceeding_df


def t_above_2kw(df):
    # Add columns for the year, week, and weekday
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for weekdays
    weekday_df = df.filter(pl.col('weekday') < 5)
    # Group by year, week, day, and hour, calculate the average consumption
    hourly_avg_df = weekday_df.filter(pl.col('cons') > 2).group_by(["year", "week", "weekday"]).agg(
        pl.min('hour').alias('daily_t_above_1kW_hour')
    )
    first_exceeding_df = hourly_avg_df.group_by(['year', 'week']).agg(
        pl.mean('daily_t_above_1kW_hour').alias('first_exceeding_1kW_hour')
    )
    return first_exceeding_df


def t_above_mean(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Calculate the weekly mean for power consumption
    weekly_mean_df = df.group_by(["year", "week"]).agg(
        pl.mean('cons').alias('weekly_mean_cons')
    )
    # Join the original df with the weekly means
    joined_df = df.join(weekly_mean_df, on=["year", "week"])
    # Count the number of points above the mean per week
    joined_df = joined_df.filter(pl.col('cons') > pl.col('weekly_mean_cons')).group_by(["year", "week", "weekday"]).agg(
        pl.first('hour').alias('daily_first_time_above_mean')
    )
    t_above_mean_df = joined_df.group_by(['year', 'week']).agg(
        pl.mean('daily_first_time_above_mean').alias('first_time_above_mean')
    )
    return t_above_mean_df


def t_daily_max(df):
    # Add columns for the year, week, and weekday
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for weekdays
    weekday_df = df.filter(pl.col('weekday') < 5)
    # Group by year, week, day, and calculate the daily maximum consumption
    daily_max_df = weekday_df.group_by(["year", "week", "weekday"]).agg(
        pl.max('cons').alias('daily_max')
    )
    # Merge the daily maximum dataframe with the weekly averages
    weekday_df = weekday_df.join(daily_max_df, on=["year", "week", "weekday"])
    # Find the first day and time when the maximum consumption reaches/exceeds the weekly average
    daily_first_exceeding_df = weekday_df.filter(pl.col('cons') == pl.col('daily_max')).group_by(["year", "week", "weekday"]).agg(
        pl.first('hour').alias('daily_time_at_max')
    )
    first_exceeding_df = daily_first_exceeding_df.group_by(['year', 'week']).agg(
        pl.mean('daily_time_at_max').alias('time_at_max')
    )
    return first_exceeding_df


def t_daily_min(df):
    # Add columns for the year, week, and weekday
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday"),
        pl.col('dt').dt.hour().alias("hour")
    ])
    # Filter for weekdays
    weekday_df = df.filter(pl.col('weekday') < 5)
    # Group by year, week, day, and calculate the daily minimum consumption
    daily_min_df = weekday_df.group_by(["year", "week", "weekday"]).agg(
        pl.min('cons').alias('daily_min')
    )
    # Merge the daily maximum dataframe with the weekly averages
    weekday_df = weekday_df.join(daily_min_df, on=["year", "week", "weekday"])
    # Find the first day and time when the maximum consumption reaches/exceeds the weekly average
    daily_first_exceeding_df = weekday_df.filter(pl.col('cons') == pl.col('daily_min')).group_by(["year", "week", "weekday"]).agg(
        pl.first('hour').alias('daily_time_at_min')
    )
    first_exceeding_df = daily_first_exceeding_df.group_by(['year', 'week']).agg(
        pl.mean('daily_time_at_min').alias('time_at_min')
    )
    return first_exceeding_df


def ts_stl_varRem(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])
    # Sort by 'dt'
    df = df.sort('dt')
    # Group by year and week, and check for missing values in each group
    # missing_check_df = df.group_by(["year", "week"]).agg(
    #     pl.any(pl.col('cons').is_null()).alias('has_missing')
    # )
    # # Join the original df with the missing values check
    # df = df.join(missing_check_df, on=["year", "week"])

    # Apply STL decomposition and calculate variance, but output NaN if missing values are present
    def calc_stl_variance(args: List[pl.Series]):
        if args[0].is_null().any():
            return None
        ts_data = args[0].to_numpy()
        stl = STL(ts_data, period=52)  # Adjust period as needed
        result = stl.fit()
        remainder = result.resid
        return np.var(remainder)

    result_df = df.group_by(["year", "week"]).agg(pl.apply([pl.col('cons')],calc_stl_variance).alias("mean_residual_stl"))
    return result_df


def ts_acf_mean3h(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])

    # Function to apply to each group
    def calc_autocorrelation(args: List[pl.Series]):
        # Check for missing values
        if args[0].is_null().any():
            return None
        # Calculate autocorrelation with lag up to 12
        autocorr_values = acf(args[0].to_numpy(), nlags=12, fft=True, missing='conservative')
        # Compute the mean of the autocorrelation values (excluding the first value at lag=0)
        mean_autocorr = np.mean(autocorr_values[1:])
        return mean_autocorr

    # Group by year and week and apply the autocorrelation function
    result_df = df.group_by(["year", "week"]).agg(pl.apply([pl.col('cons')],calc_autocorrelation).alias("mean_autocorrelation"))
    return result_df


def ts_acf_mean3h_weekday(df):
    # Add columns for the year, week, and weekday
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week"),
        pl.col('dt').dt.weekday().alias("weekday")
    ])
    # Filter for weekdays
    weekday_df = df.filter(pl.col('weekday') < 5)

    # Function to apply to each group
    def calc_autocorrelation(args: List[pl.Series]):
        # Check for missing values
        if args[0].is_null().any():
            return None
        # Calculate autocorrelation with lag up to 12
        autocorr_values = acf(args[0].to_numpy(), nlags=12, fft=True, missing='conservative')
        # Compute the mean of the autocorrelation values (excluding the first value at lag=0)
        mean_autocorr = np.mean(autocorr_values[1:])
        return mean_autocorr

    # Group by year and week and apply the autocorrelation function
    result_df = weekday_df.group_by(["year", "week"]).agg(pl.apply([pl.col('cons')],calc_autocorrelation).alias("mean_autocorrelation_wd"))
    return result_df


def t_wide_peaks(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])

    # Function to calculate peak metrics for each group
    def number_wide_peaks(args: List[pl.Series]):
        if args[0].is_null().any():
            return None
        # Identify peaks
        max_val = args[0].max()
        peaks = args[0] > (0.5 * max_val)
        # Find non-peaks and calculate peak widths
        non_peaks = [i for i, val in enumerate(peaks, start=1) if not val]
        non_peaks.append(len(peaks) + 1)  # Append a sentinel value
        lv = 0
        d_peaks = [2]  # Initialize with 2 as in the R code
        for i in non_peaks:
            temp = sum(peaks[lv:i])
            if temp > 1:
                d_peaks.append(temp)
            lv = i
        # Calculate metrics
        N_peaks = sum(d_peaks)
        #mean_d_peaks = np.mean(d_peaks)
        return N_peaks

    # Group by year and week and apply the autocorrelation function
    result_df = df.group_by(["year", "week"]).agg(pl.apply([pl.col('cons')],number_wide_peaks).alias("t_wide_peaks"))
    return result_df


def t_width_peaks(df):
    # Add columns for the year and week
    df = df.with_columns([
        pl.col('dt').dt.year().alias("year"),
        pl.col('dt').dt.week().alias("week")
    ])

    # Function to calculate peak metrics for each group
    def width_peaks(args: List[pl.Series]):
        if args[0].is_null().any():
            return None
        # Identify peaks
        max_val = args[0].max()
        peaks = args[0] > (0.5 * max_val)
        # Find non-peaks and calculate peak widths
        non_peaks = [i for i, val in enumerate(peaks, start=1) if not val]
        non_peaks.append(len(peaks) + 1)  # Append a sentinel value
        lv = 0
        d_peaks = [2]  # Initialize with 2 as in the R code
        for i in non_peaks:
            temp = sum(peaks[lv:i])
            if temp > 1:
                d_peaks.append(temp)
            lv = i
        # Calculate metrics
        mean_d_peaks = np.mean(d_peaks)
        return mean_d_peaks

    # Group by year and week and apply the autocorrelation function
    result_df = df.group_by(["year", "week"]).agg(pl.apply([pl.col('cons')],width_peaks).alias("t_width_peaks"))
    return result_df