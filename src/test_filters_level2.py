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

