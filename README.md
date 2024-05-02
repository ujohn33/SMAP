
# SMAP Library

SMAP is a Python library for smart meter analysis of consumption and temperature time-series data. It is a recreation of an R package, designed to provide similar functionality in Python.

## Features

SMAP provides a range of functions for analyzing time-series data, including:

- Calculating weekly average consumption
- Calculating evening consumption minus the minimum consumption
- Calculating peak metrics for each group

## Installation

You can install SMAP using pip:

```bash
pip install SMAP
```

SMAP requires Python 3.6 or later and Polars 0.10.0 or later.

## Usage

To use the functions in SMAP, import the library and call the desired function with the appropriate parameters. For example:

```python
import SMAP

# Create a DataFrame
data = {
    'dt': ['2023-01-01 19:30', '2023-01-01 20:00', '2023-01-08 13:00', '2023-01-09 20:00'],
    'cons': [15, 25, 35, 45]
}
df = pl.DataFrame(data)

# Call a function
result = SMAP.c_evening_min(df)
```

## Contributing

Contributions are welcome! Please read the contributing guidelines before making any changes.

## License

SMAP is licensed under the terms of the MIT license.

## Contact

If you have any questions or feedback, please contact the author:

Evgenii Genov  
eugengenov@gmail.com  
For more information, please visit the SMAP GitHub repository.
