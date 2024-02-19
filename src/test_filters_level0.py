import pandas as pd
from pandas.testing import assert_frame_equal

import config
from pathlib import Path
import load_option_data_01 as l1

WRDS_USERNAME = Path(config.WRDS_USERNAME)
DATA_DIR = Path(config.DATA_DIR)

def test_starting_data_shape():
    # test that the data is the same shape as the original
    df = l1.load_all_optm_data(DATA_DIR)
    assert df.shape[0] == 3410580

def test_starting_option_type_dist():
    # test that the option type distribution is the same as the original
    df = l1.load_all_optm_data(DATA_DIR)
    assert df['cp_flag'].value_counts().to_dict() == {'P': 1706360, 'C': 1704220}
