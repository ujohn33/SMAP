
# SMAP Library

SMAP is a Python library for smart meter analysis of consumption and temperature time-series data. It's designed as a Python implementation of the R package [SmartMeterAnalytics](https://cran.r-project.org/web/packages/SmartMeterAnalytics/index.html), providing similar functionality for Python users.

## Features

Derived from extensive research and machine learning analysis, SMAP uses smart meter and weather data to derive 133 features, enabling recognition of various household characteristics with high accuracy. These features facilitate detailed analyses and interventions for energy efficiency improvements. Here are some key functionalities:

- Weekly average consumption calculations.
- Evening minus minimum consumption calculations.
- Peak metrics calculation for each group.
- Predicting household characteristics with high accuracy, including the type of water and space heating, the age of appliances, and the presence of photovoltaic systems.

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

## License

SMAP is licensed under the terms of the MIT license.

## Contact

For any questions or feedback, please contact:

Evgenii Genov  
eugengenov@gmail.com  

Visit the SMAP [GitHub repository](https://github.com/your-repo/smap) for more information.
