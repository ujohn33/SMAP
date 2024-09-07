# smap/decorators.py

import polars as pl

def replace_na_with_defaults_decorator(rep_zero: dict, rep_min1: dict):
    def decorator(func):
        def wrapper(df, replace_NA_with_defaults=False, *args, **kwargs):
            func_name = func.__name__
            
            # If the replace_NA_with_defaults flag is True, handle missing values before applying the function
            if replace_NA_with_defaults:
                # Handle columns in rep_zero
                if func_name in rep_zero:
                    for col in rep_zero[func_name]:
                        if col in df.columns:
                            df = df.with_columns(
                                pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col)).alias(col)
                            )
                            # Use fill_nan to replace NaN values with 0
                            df = df.with_columns(
                                pl.col(col).fill_nan(0).fill_null(0).alias(col)
                            )
                
                # Handle columns in rep_min1
                if func_name in rep_min1:
                    for col in rep_min1[func_name]:
                        if col in df.columns:
                            df = df.with_columns(
                                pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col)).alias(col)
                            )
                            # Use fill_nan to replace NaN values with -1
                            df = df.with_columns(
                                pl.col(col).fill_nan(-1).fill_null(-1).alias(col)
                            )
            
            # Apply the original function after handling missing values
            result = func(df, *args, **kwargs)
            
            return result
        
        return wrapper
    return decorator