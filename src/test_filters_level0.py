import pandas as pd
from pandas.testing import assert_frame_equal

import config
from pathlib import Path
import load_option_data_01 as l1

WRDS_USERNAME = Path(config.WRDS_USERNAME)
DATA_DIR = Path(config.DATA_DIR)

""" Unit tests for the following:
    1). Determing if load of data from wrds using load_all_optm_data function was pulled correctly.
    2). Confirming option distribution is expected.

"""
def test_load_option_data_shape():
    """Test that the data has the correct shape after pulling from wrds using load function.
    """
    # call function to load data
    df = l1.load_all_optm_data(DATA_DIR)
    # expected result
    expected = 3410580
    # assert that the number of rows is the same as the original
    assert df.shape[0] == expected, "The number of rows should return 3,410,580"

def test_load_option_data_columns():
    """Test that the data has the correct columns after pulling from wrds using load function.
    """
    # call function to load data
    df = l1.load_all_optm_data(DATA_DIR)
    # expected result
    expected_columns = [
                        'secid', 'date', 'open', 'close', 'cp_flag',
                        'exdate', 'impl_volatility', 'tb_m3', 'volume',
                        'open_interest', 'best_bid', 'best_offer', 'strike_price', 
                        'contract_size'
                        ]
    # assert that the columns are the same as the original
    assert all(col in df.columns for col in expected_columns)

def test_load_option_data_dae_valadity():
    """Test that the data has the correct date range after pulling from wrds using load function.
    """
    # call function to load data
    df = l1.load_all_optm_data(DATA_DIR)
    # expected result
    expected = pd.to_datetime(['1996-01-04', '2012-01-31'])
    # assert that the date range is the same as the original
    assert df['date'].min() == expected[0], "The date range should start from 1996-01-01"
    assert df['date'].max() == expected[1], "The date range should end at 2012-01-31"

def test_load_option_type_dist():
    """Test that the option type distribution is the same as the original.
    """
    # call function to load data
    df = l1.load_all_optm_data(DATA_DIR)
    # expected result 'P': 1706360, 'C': 1704220
    expected = {'P': 1706360, 'C': 1704220}
    # assert that the option type distribution is the same as the original
    assert df['cp_flag'].value_counts().to_dict() == expected
