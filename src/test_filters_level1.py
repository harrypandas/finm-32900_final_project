import pandas as pd
from pandas.testing import assert_frame_equal
import filter_option_data_01 as f1
import config
DATA_DIR = config.DATA_DIR

def test_identical_filter(df):
    og_count = df.shape[0]
    df = f1.delete_identical_filter(df)
    assert df.shape[0] == (og_count-1)


