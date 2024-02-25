import pandas as pd
from pandas.testing import assert_frame_equal

import config
from pathlib import Path
import load_option_data_01 as l1
import filter_option_data_01 as f1
import filter_option_data_02 as f2

WRDS_USERNAME = Path(config.WRDS_USERNAME)
DATA_DIR = Path(config.DATA_DIR)
TEST_START = "1996-01-04"
TEST_END = "2012-01-31"

""" Unit tests for the following:

"""
def test_calc_mnyns():
    """Consider the scenario below to test the calc_moneyness function 
       accurately calculates moneyness using the strike price and
       underlying price.
    """
    input = pd.DataFrame(
        data={
            "strike_price": [100, 100, 170, 200, 200, 200],
            "sec_price": [100, 200, 200, 100, 200, 250]
        }
    )

    expected_output = pd.DataFrame(
          data = {
              "strike_price": [100, 100, 170, 200, 200, 200],
              "sec_price": [100, 200, 200, 100, 200, 250],
              "mnyns": [1.0, 0.5, 0.85, 2.0, 1.0, 0.8]
          }
    )

    assert_frame_equal(
        f1.calc_moneyness(input).round(4),
        expected_output.round(4)
    )

def test_l2_filters_validity():
    """Test that the data has the correct shape after applying filters from filter_option_data_02.
    """
    # call function to load data
    df = l1.load_all_optm_data(DATA_DIR)
    df = f1.appendixBfilter_level1(df)[0]

    # apply level 2 filters
    df = f2.apply_l2_filters(df)

    # expected result >> based on paper
    expected = 1_076_744

    # assert that the number of rows is the same as the original
    assert df.shape[0] == expected, "The number of rows should return 1,076,744"