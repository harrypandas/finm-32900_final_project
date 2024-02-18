import pandas as pd
from pandas.testing import assert_frame_equal

import config
DATA_DIR = config.DATA_DIR

def test_starting_data_shape(df):
    # test that the data is the same shape as the original
    assert df.shape[0] == 3410580

def test_starting_option_type_dist(df):
    # test that the option type distribution is the same as the original
    assert df['cp_flag'].value_counts().to_dict() == {'P': 1706360, 'C': 1704220}
