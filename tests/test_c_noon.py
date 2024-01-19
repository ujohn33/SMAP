import polars as pl
import pytest
from smap import c_noon


def test_c_noon():
    # Sample data
    data = {
        'dt': ['2023-01-01 10:30', '2023-01-01 11:00', '2023-01-08 13:00', '2023-01-09 12:00'],
        'cons': [15, 25, 35, 45]
    }
    df = pl.DataFrame(data)

    # Expected output
    expected_data = {
        'year': [2023, 2023],
        'week': [52, 2],
        'average_cons': [20.0, 40.0]  # Average for each week
    }
    expected_df = pl.DataFrame(expected_data)

    # Run the c_noon function
    result_df = c_noon(df)

    # Assert the result is as expected
    assert result_df.frame_equal(expected_df)

# Run the test
pytest.main()

